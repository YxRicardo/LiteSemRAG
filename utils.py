from __future__ import annotations
import numpy as np
import hdbscan
from collections import defaultdict
from functools import lru_cache
import torch, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pympler import asizeof
import math
import re
from typing import Sequence, Optional, Set

def mrr_for_one_query_titles(
    ranked_titles: Sequence[str],
    gold_titles: Sequence[str],
    k: Optional[int] = 10,
    normalize: bool = True,
) -> float:
    """
    MRR@k for one query, where both retrieved items and gold labels are title strings.

    - Multi-gold: the first occurrence of ANY gold title determines reciprocal rank.
    - If no gold appears in top-k, returns 0.0
    """

    def norm(s: str, do_norm: bool = normalize) -> str:
        s = str(s).strip()
        if do_norm:
            s = s.lower()
            s = " ".join(s.split())  # collapse multiple spaces/tabs/newlines
        return s

    gold_set: Set[str] = {norm(t) for t in gold_titles if t is not None and str(t).strip()}
    if not gold_set:
        return 0.0

    end = len(ranked_titles) if k is None else min(k, len(ranked_titles))
    for idx in range(end):
        t = ranked_titles[idx]
        if t is None:
            continue
        if norm(t) in gold_set:
            return 1.0 / (idx + 1)  # 1-indexed rank
    return 0.0

time_stamp = datetime.now().strftime("%m-%d-%H-%M")
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
DEFAULT_WIKIDATA_HEADERS = {
    "User-Agent": "LiteSemRAG/1.0 (https://www.wikidata.org/)"
}
ENGLISH_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "trillion", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
    "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth",
    "billionth", "trillionth",
}

NON_WIKIPEDIA_SITELINK_SITES = {
    "commonswiki",
    "foundationwiki",
    "incubatorwiki",
    "mediawikiwiki",
    "metawiki",
    "outreachwiki",
    "specieswiki",
    "wikidatawiki",
    "wikimaniawiki",
}


def _import_wikidata_deps():
    import pandas as pd
    import requests

    return requests, pd


def _safe_get_json(url, params=None, headers=None, timeout=30):
    """Safely call an HTTP JSON endpoint and return parsed JSON (or empty dict on failure)."""
    requests, _ = _import_wikidata_deps()
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers or DEFAULT_WIKIDATA_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return {}
    except ValueError as exc:
        print(f"Invalid JSON response: {exc}")
        return {}


def _coerce_aliases_for_search(value):
    """Convert aliases from wbsearchentities result into a clean comma-separated string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, str):
        return value
    return str(value)


def _safe_string_series(series):
    """Convert a pandas Series to a string-valued Series safe for .str accessors."""
    return series.map(lambda value: "" if value is None else str(value))


def _series_casefold_equals(series, target: str):
    target_casefold = str(target).casefold()
    return series.map(lambda value: ("" if value is None else str(value)).casefold() == target_casefold)


def _series_casefold_contains(series, target: str):
    target_casefold = str(target).casefold()
    return series.map(lambda value: target_casefold in ("" if value is None else str(value)).casefold())


def _series_startswith_lowercase(series):
    return series.map(
        lambda value: (str(value)[0].islower() if value is not None and str(value) else False)
    )


def _series_non_empty_mask(series):
    return series.map(lambda value: ("" if value is None else str(value)).strip() != "")


def _series_exclude_name_descriptions(series):
    blocked_phrases = ("family name", "given name")
    return series.map(
        lambda value: not any(phrase in str(value).casefold() for phrase in blocked_phrases)
        if value is not None else True
    )


def _should_skip_wikidata_term(term: str) -> bool:
    normalized_term = str(term).strip()
    if not normalized_term:
        return True

    if len(normalized_term.split()) >= 3:
        return True

    if re.search(r"\d", normalized_term):
        return True

    word_tokens = re.findall(r"[a-z]+", normalized_term.casefold())
    return any(token in ENGLISH_NUMBER_WORDS for token in word_tokens)


def extract_best_wikipedia_title(entity, language="en"):
    """Return the best Wikipedia title from a Wikidata entity's sitelinks."""
    if not isinstance(entity, dict):
        return ""

    sitelinks = entity.get("sitelinks", {})
    if not isinstance(sitelinks, dict) or not sitelinks:
        return ""

    preferred_site = f"{language}wiki"
    preferred = sitelinks.get(preferred_site)
    if isinstance(preferred, dict):
        return preferred.get("title", "") or ""

    for site_name, site_info in sitelinks.items():
        if not site_name.endswith("wiki") or not isinstance(site_info, dict):
            continue
        if site_name in NON_WIKIPEDIA_SITELINK_SITES:
            continue
        title = site_info.get("title", "")
        if title:
            return title

    return ""


def _truncate_to_sentences(text, sentences):
    if not text or sentences is None or sentences <= 0:
        return text or ""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [part for part in parts if part]
    if not parts:
        return ""
    return " ".join(parts[:sentences])


def fetch_wikipedia_intro(title, language="en", headers=None, timeout=30, sentences=3):
    """Fetch a short introductory summary for a Wikipedia title."""
    if not isinstance(title, str) or not title.strip():
        return ""

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1,
        "redirects": 1,
        "titles": title.strip(),
    }
    if sentences is not None:
        params["exsentences"] = int(sentences)

    url = f"https://{language}.wikipedia.org/w/api.php"
    data = _safe_get_json(url, params=params, headers=headers, timeout=timeout)
    pages = data.get("query", {}).get("pages", {}) if isinstance(data, dict) else {}
    if not isinstance(pages, dict):
        return ""

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        extract = page.get("extract", "")
        if extract:
            return _truncate_to_sentences(extract, sentences)

    return ""


def fetch_detailed_descriptions_for_entities(
    entity_ids,
    language="en",
    headers=None,
    timeout=30,
    sentences=3,
):
    """Fetch Wikipedia-backed detailed descriptions keyed by Wikidata entity ID."""
    if not entity_ids:
        return {}

    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(str(entity_id).strip() for entity_id in entity_ids if str(entity_id).strip()),
        "languages": language,
        "props": "sitelinks",
    }
    if not params["ids"]:
        return {}

    data = _safe_get_json(WIKIDATA_API_URL, params=params, headers=headers, timeout=timeout)
    entities = data.get("entities", {}) if isinstance(data, dict) else {}
    if not isinstance(entities, dict):
        return {}

    detailed_descriptions = {}
    for entity_id, entity in entities.items():
        wikipedia_title = extract_best_wikipedia_title(entity, language=language)
        if not wikipedia_title:
            detailed_descriptions[entity_id] = ""
            continue
        detailed_descriptions[entity_id] = fetch_wikipedia_intro(
            wikipedia_title,
            language=language,
            headers=headers,
            timeout=timeout,
            sentences=sentences,
        )
    return detailed_descriptions


def filter_entity_ids_with_wikipedia_sitelinks(entity_ids, language="en", headers=None, timeout=30):
    """Return entity IDs that have a Wikipedia sitelink, preferring the requested language."""
    if not entity_ids:
        return set()

    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(str(entity_id).strip() for entity_id in entity_ids if str(entity_id).strip()),
        "languages": language,
        "props": "sitelinks",
    }
    if not params["ids"]:
        return set()

    data = _safe_get_json(WIKIDATA_API_URL, params=params, headers=headers, timeout=timeout)
    entities = data.get("entities", {}) if isinstance(data, dict) else {}
    if not isinstance(entities, dict):
        return set()

    valid_entity_ids = set()
    preferred_site = f"{language}wiki"
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        sitelinks = entity.get("sitelinks", {})
        if not isinstance(sitelinks, dict) or not sitelinks:
            continue
        if preferred_site in sitelinks or any(site_name.endswith("wiki") for site_name in sitelinks):
            valid_entity_ids.add(entity_id)

    return valid_entity_ids


def search_wikidata(
    term,
    language="en",
    limit=10,
    exact_match_text=False,
    exact_match_first=False,
    include_detailed_description=False,
    drop_missing_detailed_description=False,
    detailed_description_sentences=3,
    filter_name=True,
    label_contains_text=True,
):
    """
    Search Wikidata entities using the same interface as explore_wikidata.ipynb.
    """
    _, pd = _import_wikidata_deps()

    if not isinstance(term, str) or not term.strip():
        print("Please provide a non-empty search term.")
        columns = ["id", "label", "description", "match_text", "aliases", "concepturi"]
        if include_detailed_description:
            columns.insert(3, "detailed_description")
        return pd.DataFrame(columns=columns)

    normalized_term = term.strip()
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "uselang": language,
        "search": normalized_term,
        "limit": int(limit),
    }

    data = _safe_get_json(WIKIDATA_API_URL, params=params)
    items = data.get("search", []) if isinstance(data, dict) else []

    rows = []
    for item in items:
        match = item.get("match") if isinstance(item, dict) else None
        match_text = ""
        if isinstance(match, dict):
            match_text = match.get("text", "")

        rows.append(
            {
                "id": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "match_text": match_text,
                "aliases": _coerce_aliases_for_search(item.get("aliases")),
                "concepturi": item.get("concepturi", ""),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["id", "label", "description", "match_text", "aliases", "concepturi"],
    )

    if filter_name and not df.empty:
        df = df[_series_exclude_name_descriptions(df["description"])].reset_index(drop=True)

    if label_contains_text and not df.empty:
        df = df[_series_casefold_contains(df["label"], normalized_term)].reset_index(drop=True)

    if exact_match_first and not df.empty:
        exact_match_mask = _series_casefold_equals(df["match_text"], normalized_term)
        lowercase_initial_mask = _series_startswith_lowercase(df["match_text"])
        df = df.assign(
            _exact_match_rank=exact_match_mask.astype(int),
            _lowercase_initial_rank=lowercase_initial_mask.astype(int),
        )
        df = (
            df.sort_values(
                ["_exact_match_rank", "_lowercase_initial_rank"],
                ascending=[False, False],
                kind="stable",
            )
            .drop(columns=["_exact_match_rank", "_lowercase_initial_rank"])
            .reset_index(drop=True)
        )

    if exact_match_text and not df.empty:
        df = df[_series_casefold_equals(df["match_text"], normalized_term)].reset_index(drop=True)

    if include_detailed_description:
        detailed_descriptions = {}
        if not df.empty:
            detailed_descriptions = fetch_detailed_descriptions_for_entities(
                df["id"].tolist(),
                language=language,
                headers=DEFAULT_WIKIDATA_HEADERS,
                timeout=30,
                sentences=detailed_description_sentences,
            )
        df.insert(
            df.columns.get_loc("description") + 1,
            "detailed_description",
            [detailed_descriptions.get(entity_id, "") for entity_id in df["id"]],
        )
        if drop_missing_detailed_description:
            df = df[_series_non_empty_mask(df["detailed_description"])].reset_index(drop=True)

    return df


def load_wikidata_definition_candidates(
    query_text: str,
    use_detailed_description: bool = True,
    exact_match_text: bool = False,
    exact_match_first: bool = False,
    limit: int = 5,
    filter_name: bool = True,
    require_detailed_description: bool = False,
    label_contains_text: bool = True,
):
    _, pd = _import_wikidata_deps()

    include_detailed_description = use_detailed_description or require_detailed_description
    candidates_df = search_wikidata(
        query_text,
        limit=limit,
        exact_match_text=exact_match_text,
        exact_match_first=exact_match_first,
        include_detailed_description=include_detailed_description,
        detailed_description_sentences=3,
        drop_missing_detailed_description=include_detailed_description,
        filter_name=filter_name,
        label_contains_text=label_contains_text,
    )

    if candidates_df.empty:
        if include_detailed_description:
            raise ValueError(
                f"search_wikidata returned no detailed_description candidates for span={query_text!r}."
            )
        raise ValueError(
            f"search_wikidata returned no description candidates for span={query_text!r}."
        )

    definition_column = "detailed_description" if use_detailed_description else "description"
    candidates_df = candidates_df[_series_non_empty_mask(candidates_df[definition_column])].copy()
    candidates_df = candidates_df.drop_duplicates(subset=["id", definition_column]).reset_index(drop=True)
    if candidates_df.empty:
        raise ValueError(
            f"No usable {definition_column} candidates remained for span={query_text!r}."
        )

    return candidates_df, definition_column


def definition_to_hypothesis(definition: str) -> str:
    cleaned_definition = definition.strip()
    if cleaned_definition.endswith((".", "!", "?")):
        cleaned_definition = cleaned_definition[:-1]
    return f"It refers to {cleaned_definition}."


def extract_cross_encoder_scores(raw_scores, model) -> np.ndarray:
    score_array = np.asarray(raw_scores)
    if score_array.ndim == 1:
        return score_array.astype(float)

    id2label = getattr(model.model.config, "id2label", {}) or {}
    entailment_index = None
    for label_index, label_name in id2label.items():
        if str(label_name).lower() == "entailment":
            entailment_index = int(label_index)
            break

    if entailment_index is None:
        entailment_index = score_array.shape[1] - 1

    return score_array[:, entailment_index].astype(float)


def build_wikidata_candidate_bank(candidates_df, definition_column: str) -> list[dict]:
    candidate_bank = []
    for row in candidates_df.itertuples(index=False):
        definition = str(getattr(row, definition_column)).strip()
        candidate_bank.append(
            {
                "entity_id": row.id,
                "label": row.label,
                "description": row.description,
                "definition_source": definition_column,
                "definition": definition,
                "hypothesis": definition_to_hypothesis(definition),
            }
        )
    return candidate_bank


@lru_cache(maxsize=4096)
def _get_wikidata_term_info_cached(term, language="en", filter_name=True):
    """Return exact-match Wikidata row count plus description list using a lightweight lookup."""
    if _should_skip_wikidata_term(term):
        return 0, ("an entity",)

    try:
        results = search_wikidata(
            term,
            language=language,
            exact_match_text=True,
            include_detailed_description=False,
            drop_missing_detailed_description=False,
            filter_name=filter_name,
        )
    except Exception as exc:
        print(f"Wikidata lookup failed for {term!r}: {exc}")
        return 0, ("an entity",)

    if results.empty:
        return 0, ("an entity",)

    try:
        valid_entity_ids = filter_entity_ids_with_wikipedia_sitelinks(
            results["id"].tolist(),
            language=language,
            headers=DEFAULT_WIKIDATA_HEADERS,
            timeout=30,
        )
    except Exception as exc:
        print(f"Wikidata sitelink lookup failed for {term!r}: {exc}")
        return 0, ("an entity",)

    if not valid_entity_ids:
        return 0, ("an entity",)

    filtered_results = results[results["id"].isin(valid_entity_ids)].reset_index(drop=True)
    row_count = len(filtered_results.index)
    if row_count == 0:
        return 0, ("an entity",)

    descriptions = []
    for description in filtered_results["description"].tolist():
        normalized_description = str(description).strip() if description is not None else ""
        descriptions.append(normalized_description or "an entity")

    return row_count, tuple(descriptions)


def get_wikidata_term_info(term, language="en", filter_name=True):
    normalized_term = " ".join(str(term).strip().split())
    return _get_wikidata_term_info_cached(normalized_term, language=language, filter_name=filter_name)


@lru_cache(maxsize=4096)
def is_multi_semantic_by_wikidata(term, language="en", filter_name=True):
    """Return True when a term resolves to more than one detailed exact-match Wikidata entity."""
    row_count, _ = get_wikidata_term_info(term, language=language, filter_name=filter_name)
    return row_count > 1

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

# 其实在 100 样本规模：
#
# 👉 HDBSCAN 不一定是最优
# 👉 有时层次聚类（Agglomerative）更清晰
# from sklearn.cluster import AgglomerativeClustering
#
# clustering = AgglomerativeClustering(
#     n_clusters=3,
#     affinity='euclidean',
#     linkage='ward'
# )


def hdbscan_cluster(
    embeds_list,
    min_cluster_size=10,
    percentile=0.9,
    merge_chunks=False
    ):
    """
    embeds_list: List[(embedding_tensor, chunk_node)]
    merge_chunks: 是否按照 chunk_node_id 聚合
    """

    # Kept for backward compatibility with existing callers.
    _ = percentile
    original_count = len(embeds_list)

    merged_embeds = []
    merged_to_original = []

    # =========================
    # 1️⃣ 聚合 or 不聚合
    # =========================
    if merge_chunks:

        chunk_groups = defaultdict(list)

        for idx, (embed, chunk_node) in enumerate(embeds_list):
            chunk_id = chunk_node.chunk_node_id
            chunk_groups[chunk_id].append((idx, embed))

        for items in chunk_groups.values():

            indices = []
            embeds = []

            for idx, embed in items:
                indices.append(idx)
                embeds.append(embed.detach().cpu().numpy())

            embeds = np.array(embeds)

            merged_embed = np.mean(embeds, axis=0)

            merged_embeds.append(merged_embed)
            merged_to_original.append(indices)

    else:

        for idx, (embed, _) in enumerate(embeds_list):

            merged_embeds.append(embed.detach().cpu().numpy())
            merged_to_original.append([idx])

    merged_embeds = np.array(merged_embeds)
    merged_count = len(merged_embeds)

    # =========================
    # 2️⃣ min_cluster_size 缩放
    # =========================
    if merge_chunks:
        k = merged_count / original_count
    else:
        k = 1

    scaled_min_cluster_size = max(2, math.ceil(min_cluster_size * k))

    # =========================
    # 3️⃣ L2 normalize
    # =========================
    norms = np.linalg.norm(merged_embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = merged_embeds / norms

    # =========================
    # 4️⃣ PCA
    # =========================
    pca_n_components = min(50, X_norm.shape[0], X_norm.shape[1])
    pca = PCA(n_components=pca_n_components)
    X_reduced = pca.fit_transform(X_norm)

    # =========================
    # 5️⃣ HDBSCAN
    # =========================
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=scaled_min_cluster_size,
        # min_samples=max(2, scaled_min_cluster_size // 2),
        metric="euclidean",
        # cluster_selection_method='eom',
        # cluster_selection_epsilon=0.05
    )

    cluster_labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    # =========================
    # 6️⃣ 映射回原始 index
    # =========================
    clusters = defaultdict(list)

    for merged_idx, label in enumerate(cluster_labels):

        original_indices = merged_to_original[merged_idx]

        clusters[label].extend(original_indices)

    # =========================
    # 7️⃣ cluster centers
    # =========================
    cluster_centers = {}

    for label in clusters.keys():

        merged_indices = [
            i for i, l in enumerate(cluster_labels) if l == label
        ]

        cluster_embeds = X_norm[merged_indices]

        center = np.mean(cluster_embeds, axis=0)

        norm = np.linalg.norm(center)
        if norm > 0:
            center /= norm

        cluster_centers[label] = center

    return n_clusters, clusters, cluster_centers

def get_anomaly_threshold(values, percentile):
    q = 1 - percentile
    return float(np.percentile(values, q * 100))

# def get_anomaly_threshold(center, embeds_list, percentile):
#     embeds = torch.stack(embeds_list)
#
#     center = torch.nn.functional.normalize(center, p=2, dim=0)
#     embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
#
#     similarities = embeds @ center
#
#     threshold = torch.quantile(similarities, 1 - percentile)
#
#     return threshold.item()

# def hdbscan_cluster(embeds_list, min_size=5):
#     X = np.array(embeds_list)
#
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=min_size,
#         metric='cosine'
#     )
#
#     cluster_labels = clusterer.fit_predict(X)
#     n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
#
#     clusters = defaultdict(list)
#
#     for idx, label in enumerate(cluster_labels):
#         clusters[label].append(idx)
#
#     clusters.pop(-1, None)
#     cluster_centers = {}
#
#     for label, indices in clusters.items():
#         cluster_embeds = X[indices]
#         center = np.mean(cluster_embeds, axis=0)
#         norm = np.linalg.norm(center)
#         if norm > 0:
#             center = center / norm
#
#         cluster_centers[label] = center
#     return n_clusters, clusters, cluster_centers

def e5_average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# def get_s_mean(embeds_buffer):
#     embeds_mat = np.stack(embeds_buffer, axis=0)
#     mu = l2_normalize(embeds_mat.mean(axis=0))
#     E_norm = embeds_mat / (np.linalg.norm(embeds_mat, axis=1, keepdims=True) + 1e-12)
#     s_mean = float(E_norm @ mu).mean()
#     return s_mean

def get_s_mean(embeds_buffer):
    embeds_mat = torch.stack(embeds_buffer)  # [N, 1024]
    mu = F.normalize(embeds_mat.mean(dim=0), dim=0)
    E_norm = F.normalize(embeds_mat, dim=1)

    s_mean = torch.matmul(E_norm, mu).mean()
    return s_mean.item()


def sorted_indices(values: Sequence[float]) -> list[int]:
    """
    Sorts the list of values and returns the indices of the sorted order.

    Args:
        values (Sequence[float]): The input list of values to be sorted.

    Returns:
        list[int]: Indices of the values in sorted order.
    """
    return sorted(range(len(values)), key=lambda i: values[i])

def average_embeds(tensors, eps=1e-12):
    x = torch.stack(tensors, dim=0)
    x = F.normalize(x, p=2, dim=1)
    m = x.mean(dim=0)
    return F.normalize(m, p=2, dim=0)


def proto_embed_sim(proto_node):
    proto_embed = proto_node.embed
    chunk_node_embed = torch.stack(proto_node.chunk_node_embed)
    proto_embed = proto_embed.unsqueeze(0)  # [1, d]
    sim = F.cosine_similarity(chunk_node_embed, proto_embed, dim=1)

    return sim

def proto_node_combine_sim(sim, proto_node, r=3):
    """
    sim: list/iterable of similarity floats (same length as proto_node.chunk_node)
    proto_node.chunk_node: list of chunk nodes (may contain duplicates)
    r: top-r to average
    """
    bucket = defaultdict(list)

    # collect sims per chunk
    for s, chunk in zip(sim, proto_node.chunk_node_list):
        bucket[chunk].append(float(s))

    # unique chunks + top-r mean
    new_chunks = []
    combined_sim = []
    for chunk, sims in bucket.items():
        sims.sort(reverse=True)
        top = sims[:r]
        combined_sim.append(sum(top) / len(top))
        new_chunks.append(chunk)

    proto_node.chunk_node_list = new_chunks
    return combined_sim


def inspect_prototypes(embed, prototype_list):
    B = torch.stack([prototype.embed for prototype in prototype_list])  # [N, D]
    A = embed.unsqueeze(0)  # [1, D]
    similarities = F.cosine_similarity(A, B, dim=1)
    max_val, max_idx = torch.max(similarities, dim=0)
    return max_val, max_idx


def plot_embeddings(embed, token, clusters):
    folder_name = time_stamp
    current_dir = os.getcwd()
    base_dir = os.path.join(current_dir, "model_embed_dis")
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    embeddings = torch.stack(embed)  # shape: (N, D)
    embeddings_np = embeddings.detach().cpu().numpy()

    # tsne = TSNE(n_components=2, random_state=42)
    # reduced = tsne.fit_transform(embeddings_np)

    pca = PCA(n_components=min(50, embeddings_np.shape[1]))
    reduced = pca.fit_transform(embeddings_np)

    N = len(embed)

    # 为每个点建立cluster标签
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

    # 颜色映射
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        indices = [j for j, l in enumerate(labels) if l == cid]
        if len(indices) == 0:
            continue

        points = reduced[indices]

        if cid == -1:
            plt.scatter(points[:, 0], points[:, 1], color="gray", label="unassigned")
        else:
            plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=f"cluster {cid}")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"{token} PCA Visualization")

    plt.legend()
    plt.savefig(f"{folder_path}/{token}_PCA_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings_np)

    N = len(embed)

    # 为每个点建立cluster标签
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

    # 颜色映射
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))

    for i, cid in enumerate(unique_clusters):
        indices = [j for j, l in enumerate(labels) if l == cid]
        if len(indices) == 0:
            continue

        points = reduced[indices]

        if cid == -1:
            plt.scatter(points[:, 0], points[:, 1], color="gray", label="unassigned")
        else:
            plt.scatter(points[:, 0], points[:, 1], color=cmap(i), label=f"cluster {cid}")

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"{token} t-SNE Visualization")

    plt.legend()
    plt.savefig(f"{folder_path}/{token}_TSNE_embedding.png", dpi=300, bbox_inches="tight")
    plt.close()



def print_size_mb(obj, precision=2):
    size_bytes = asizeof.asizeof(obj)
    size_mb = size_bytes / (1024 * 1024)
    print(f"memory size: {size_mb:.{precision}f} MB")



def max_cosine_similarity_index(query_tensor, tensor_list):
    """
    query_tensor: shape (d,)
    tensor_list:  list of tensors, each shape (d,)
    return: index of tensor with max cosine similarity
    """

    if len(tensor_list) == 0:
        return None

    matrix = torch.stack(tensor_list)  # shape: (n, d)

    query_norm = F.normalize(query_tensor.unsqueeze(0), dim=1)  # (1, d)
    matrix_norm = F.normalize(matrix, dim=1)  # (n, d)

    similarities = torch.mm(query_norm, matrix_norm.t()).squeeze(0)  # (n,)

    max_index = torch.argmax(similarities).item()

    return max_index

import json
from pathlib import Path

def load_hotpot_distractor(file_path):
    """
    Load HotpotQA distractor setting.

    Returns:
        samples (list of dict)
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for item in data:
        sample = {}

        # 基本字段
        sample["id"] = item["_id"]
        sample["question"] = item["question"]
        sample["answer"] = item["answer"]

        # supporting facts
        # 格式: [["Page_Title", sentence_id], ...]
        supporting_pages = set()
        supporting_sentences = {}

        for title, sent_id in item["supporting_facts"]:
            supporting_pages.add(title)
            supporting_sentences.setdefault(title, []).append(sent_id)

        sample["supporting_pages"] = list(supporting_pages)
        sample["supporting_sentences"] = supporting_sentences

        # context (10篇文章)
        # 格式: [["Page_Title", ["sent1", "sent2", ...]], ...]
        documents = []

        for title, sentences in item["context"]:
            doc = {
                "title": title,
                "sentences": sentences,
                "text": " ".join(sentences)  # 拼成整篇文章
            }
            documents.append(doc)

        sample["documents"] = documents

        samples.append(sample)

    return samples

def build_global_document_list(file_path):
    """
    从 HotpotQA distractor 数据构建全局去重文档列表

    返回:
        documents: list[dict]
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_store = {}

    for item in data:
        for title, sentences in item["context"]:

            # 拼接成完整文本
            text = " ".join(sentences).strip()

            # 用 title 做唯一键（HotpotQA 基本可行）
            if title not in doc_store:
                doc_store[title] = {
                    "doc_id": title,
                    "title": title,
                    "text": text,
                    "sentences": sentences
                }
            else:
                # 如果出现重复 title 但文本更长，可以选择替换
                if len(text) > len(doc_store[title]["text"]):
                    doc_store[title]["text"] = text
                    doc_store[title]["sentences"] = sentences

    documents = list(doc_store.values())

    print(f"Total unique documents: {len(documents)}")

    return documents


import json
import pickle
from pathlib import Path


def build_hotpot_retrieval_dataset(file_path, num_samples=None):
    """
    构建 HotpotQA retrieval dataset，并带缓存机制

    返回:
        documents, samples
    """

    file_path = Path(file_path)

    # -----------------------------
    # cache 目录
    # -----------------------------

    cache_dir = Path("./hotpot_QA")
    cache_dir.mkdir(exist_ok=True)

    tag = "all" if num_samples is None else str(num_samples)

    documents_cache = cache_dir / f"hotpot_documents_{tag}.pkl"
    samples_cache = cache_dir / f"hotpot_samples_{tag}.pkl"

    # -----------------------------
    # 如果缓存存在，直接读取
    # -----------------------------

    if documents_cache.exists() and samples_cache.exists():
        print("Loading cached dataset...")

        with open(documents_cache, "rb") as f:
            documents = pickle.load(f)

        with open(samples_cache, "rb") as f:
            samples = pickle.load(f)

        print(f"Loaded {len(documents)} documents")
        print(f"Loaded {len(samples)} samples")

        return documents, samples

    # -----------------------------
    # 否则重新构建
    # -----------------------------

    print("Building dataset from raw file...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

    # -----------------------------
    # Step 1: 构建全局去重文档库
    # -----------------------------

    title_to_doc = {}

    for item in data:
        for title, sentences in item["context"]:

            text = " ".join(sentences).strip()

            if title not in title_to_doc:
                title_to_doc[title] = {
                    "title": title,
                    "text": text,
                    "sentences": sentences
                }
            else:
                if len(text) > len(title_to_doc[title]["text"]):
                    title_to_doc[title]["text"] = text
                    title_to_doc[title]["sentences"] = sentences

    documents = []
    title_to_id = {}

    for idx, (title, doc) in enumerate(title_to_doc.items()):
        doc_entry = {
            "doc_id": idx,
            "title": title,
            "text": doc["text"],
            "sentences": doc["sentences"]
        }

        documents.append(doc_entry)
        title_to_id[title] = idx

    print(f"Total unique documents: {len(documents)}")

    # -----------------------------
    # Step 2: 构建 samples
    # -----------------------------

    samples = []

    for item in data:

        gold_titles = set([title for title, _ in item["supporting_facts"]])

        gold_doc_ids = []

        for title in gold_titles:
            if title in title_to_id:
                gold_doc_ids.append(title_to_id[title])

        sample_entry = {
            "sample_id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "gold_doc_ids": gold_doc_ids
        }

        samples.append(sample_entry)

    print(f"Total samples: {len(samples)}")

    # -----------------------------
    # 保存缓存
    # -----------------------------

    with open(documents_cache, "wb") as f:
        pickle.dump(documents, f)

    with open(samples_cache, "wb") as f:
        pickle.dump(samples, f)

    print(f"Dataset cached to: {cache_dir}")

    return documents, samples
