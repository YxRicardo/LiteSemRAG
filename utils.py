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

# Compute MRR for one query using retrieved titles and gold titles.
def mrr_for_one_query_titles(
    ranked_titles: Sequence[str],
    gold_titles: Sequence[str],
    k: Optional[int] = 10,
    normalize: bool = True,
) -> float:

    # Normalize a title for case-insensitive comparison.
    def norm(s: str, do_norm: bool = normalize) -> str:
        s = str(s).strip()
        if do_norm:
            s = s.lower()
            s = " ".join(s.split())                                          
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
            return 1.0 / (idx + 1)                  
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


# Import optional Wikidata dependencies only when they are needed.
def _import_wikidata_deps():
    import pandas as pd
    import requests

    return requests, pd


# Fetch JSON from an HTTP endpoint and return an empty object on failure.
def _safe_get_json(url, params=None, headers=None, timeout=30):
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


# Convert Wikidata aliases into a clean comma-separated string.
def _coerce_aliases_for_search(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, str):
        return value
    return str(value)


# Convert a pandas Series into string values safe for string operations.
def _safe_string_series(series):
    return series.map(lambda value: "" if value is None else str(value))


# Build a case-insensitive equality mask for a pandas Series.
def _series_casefold_equals(series, target: str):
    target_casefold = str(target).casefold()
    return series.map(lambda value: ("" if value is None else str(value)).casefold() == target_casefold)


# Build a case-insensitive containment mask for a pandas Series.
def _series_casefold_contains(series, target: str):
    target_casefold = str(target).casefold()
    return series.map(lambda value: target_casefold in ("" if value is None else str(value)).casefold())


# Build a mask for non-empty string values in a pandas Series.
def _series_non_empty_mask(series):
    return series.map(lambda value: ("" if value is None else str(value)).strip() != "")


# Return whether a label starts with an uppercase character.
def _starts_with_uppercase_label(value) -> bool:
    text = "" if value is None else str(value).strip()
    return bool(text) and text[0].isupper()


# Count whitespace-separated words in a value.
def _word_count(value) -> int:
    return len(("" if value is None else str(value)).strip().split())


# Extract lowercase alphanumeric word tokens from a value.
def _word_tokens(value) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", ("" if value is None else str(value)).casefold())


# Return whether a value contains the target word sequence in order.
def _contains_ordered_word_sequence(value, target: str, allow_at_start: bool = True) -> bool:
    value_tokens = _word_tokens(value)
    target_tokens = _word_tokens(target)
    if not value_tokens or not target_tokens or len(target_tokens) > len(value_tokens):
        return False
    window_size = len(target_tokens)
    return any(
        (allow_at_start or index > 0)
        and value_tokens[index:index + window_size] == target_tokens
        for index in range(len(value_tokens) - window_size + 1)
    )


# Return whether an alias list exactly contains the target text.
def _alias_exact_match(value, target: str) -> bool:
    target_text = str(target).strip()
    if not target_text:
        return False
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        aliases = [str(alias).strip() for alias in value]
    else:
        aliases = [alias.strip() for alias in str(value).split(",")]
    return any(alias == target_text for alias in aliases if alias)


# Build a mask that removes generic name-description rows.
def _series_exclude_name_descriptions(series):
    blocked_phrases = ("family name", "given name")
    return series.map(
        lambda value: not any(phrase in str(value).casefold() for phrase in blocked_phrases)
        if value is not None else True
    )


# Return whether a Wikidata search term is too weak or noisy to query.
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


# Return the best Wikipedia title from a Wikidata entity sitelink map.
def extract_best_wikipedia_title(entity, language="en"):
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


# Limit text to a requested number of sentence-like segments.
def _truncate_to_sentences(text, sentences):
    if not text or sentences is None or sentences <= 0:
        return text or ""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [part for part in parts if part]
    if not parts:
        return ""
    return " ".join(parts[:sentences])


# Fetch a short introductory summary for a Wikipedia title.
def fetch_wikipedia_intro(title, language="en", headers=None, timeout=30, sentences=3):
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


# Fetch Wikipedia-backed descriptions keyed by Wikidata entity ID.
def fetch_detailed_descriptions_for_entities(
    entity_ids,
    language="en",
    headers=None,
    timeout=30,
    sentences=3,
):
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


# Return entity IDs that have Wikipedia sitelinks in the preferred language.
def filter_entity_ids_with_wikipedia_sitelinks(entity_ids, language="en", headers=None, timeout=30):
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


# Search Wikidata and return normalized candidate rows with descriptions.
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
        df = df.assign(
            _exact_match_rank=exact_match_mask.astype(int),
        )
        df = (
            df.sort_values(
                ["_exact_match_rank"],
                ascending=[False],
                kind="stable",
            )
            .drop(columns=["_exact_match_rank"])
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


# Select usable Wikidata candidates using span-aware label and alias rules.
def _select_wikidata_candidates_by_span_rules(
    candidates_df,
    query_text: str,
    definition_column: str,
    max_candidate_count: int,
):
    if candidates_df.empty:
        return candidates_df

    max_candidate_count = max(1, int(max_candidate_count))
    definition_mask = _series_non_empty_mask(candidates_df[definition_column])
    label_uppercase_mask = candidates_df["label"].map(_starts_with_uppercase_label)
    is_instance_query = bool(label_uppercase_mask.all())

    if is_instance_query:
        label_exact_mask = _series_casefold_equals(candidates_df["label"], query_text)
        selected = candidates_df[definition_mask & label_exact_mask].copy()
        return selected.head(max_candidate_count).reset_index(drop=True)

    query_word_count = _word_count(query_text)
    label_exact_mask = _series_casefold_equals(candidates_df["label"], query_text)
    label_contains_mask = candidates_df["label"].map(
        lambda value: _contains_ordered_word_sequence(
            value,
            query_text,
            allow_at_start=False,
        )
    )
    alias_exact_mask = candidates_df["aliases"].map(
        lambda value: _alias_exact_match(value, query_text)
    )
    label_word_count_mask = candidates_df["label"].map(
        lambda value: _word_count(value) <= query_word_count + 1
    )
    selected = candidates_df[
        definition_mask
        & ~label_uppercase_mask
        & (label_exact_mask | label_contains_mask | alias_exact_mask)
        & label_word_count_mask
    ].copy()
    if selected.empty:
        return selected.reset_index(drop=True)

    label_exact_mask = _series_casefold_equals(selected["label"], query_text)
    selected = (
        selected.assign(_label_exact_rank=label_exact_mask.astype(int))
        .sort_values(["_label_exact_rank"], ascending=[False], kind="stable")
        .drop(columns=["_label_exact_rank"])
        .head(max_candidate_count)
        .reset_index(drop=True)
    )
    return selected


# Load Wikidata definition candidates suitable for semantic-description scoring.
def load_wikidata_definition_candidates(
    query_text: str,
    use_detailed_description: bool = True,
    exact_match_text: bool = False,
    exact_match_first: bool = False,
    limit: int = 5,
    filter_name: bool = True,
    require_detailed_description: bool = False,
    label_contains_text: bool = True,
    target_candidate_count: int | None = None,
):
    max_candidate_count = max(1, int(target_candidate_count or limit))
    search_limit = int(limit)
    include_detailed_description = use_detailed_description
    candidates_df = search_wikidata(
        query_text,
        limit=search_limit,
        exact_match_text=False,
        exact_match_first=False,
        include_detailed_description=include_detailed_description,
        detailed_description_sentences=3,
        drop_missing_detailed_description=False,
        filter_name=False,
        label_contains_text=False,
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
    candidates_df = _select_wikidata_candidates_by_span_rules(
        candidates_df,
        query_text=query_text,
        definition_column=definition_column,
        max_candidate_count=max_candidate_count,
    )
    if candidates_df.empty:
        raise ValueError(
            f"No usable {definition_column} candidates remained for span={query_text!r}."
        )
    candidates_df = candidates_df.drop_duplicates(subset=["id", definition_column]).reset_index(drop=True)

    return candidates_df, definition_column


# Convert a definition into a natural-language NLI hypothesis.
def definition_to_hypothesis(definition: str) -> str:
    cleaned_definition = definition.strip()
    if cleaned_definition.endswith((".", "!", "?")):
        cleaned_definition = cleaned_definition[:-1]
    return f"It refers to {cleaned_definition}."


# Normalize CrossEncoder prediction outputs into a flat score array.
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


# Build candidate dictionaries used by semantic-description prediction.
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


# Fetch and cache lightweight Wikidata info for an exact term lookup.
@lru_cache(maxsize=4096)
def _get_wikidata_term_info_cached(term, language="en", filter_name=True):
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


# Return cached Wikidata lookup information for a term.
def get_wikidata_term_info(term, language="en", filter_name=True):
    normalized_term = " ".join(str(term).strip().split())
    return _get_wikidata_term_info_cached(normalized_term, language=language, filter_name=filter_name)


# Return whether Wikidata suggests multiple detailed meanings for a term.
@lru_cache(maxsize=4096)
def is_multi_semantic_by_wikidata(term, language="en", filter_name=True):
    row_count, _ = get_wikidata_term_info(term, language=language, filter_name=filter_name)
    return row_count > 1

# Return an L2-normalized NumPy vector.
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n + eps)

               
 
                  
                            
                                                     
 
                                       
                   
                           
                    
   


# Cluster embeddings with HDBSCAN and return labels, members, and centers.
def hdbscan_cluster(
    embeds_list,
    min_cluster_size=10,
    percentile=0.9,
    merge_chunks=False
    ):

                                                            
    _ = percentile
    original_count = len(embeds_list)

    merged_embeds = []
    merged_to_original = []

                               
                   
                               
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

                               
                             
                               
    if merge_chunks:
        k = merged_count / original_count
    else:
        k = 1

    scaled_min_cluster_size = max(2, math.ceil(min_cluster_size * k))

                               
                      
                               
    norms = np.linalg.norm(merged_embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = merged_embeds / norms

                               
             
                               
    pca_n_components = min(50, X_norm.shape[0], X_norm.shape[1])
    pca = PCA(n_components=pca_n_components)
    X_reduced = pca.fit_transform(X_norm)

                               
                 
                               
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=scaled_min_cluster_size,
                                                           
        metric="euclidean",
                                         
                                        
    )

    cluster_labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

                               
                     
                               
    clusters = defaultdict(list)

    for merged_idx, label in enumerate(cluster_labels):

        original_indices = merged_to_original[merged_idx]

        clusters[label].extend(original_indices)

                               
                         
                               
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

# Compute a similarity threshold from edge weights using a percentile.
def get_anomaly_threshold(values, percentile):
    q = 1 - percentile
    return float(np.percentile(values, q * 100))

                                                             
                                       
 
                                                                
                                                                
 
                                    
 
                                                              
 
                             

                                               
                               
 
                                  
                                    
                         
       
 
                                               
                                                                                
 
                                  
 
                                                  
                                     
 
                            
                          
 
                                             
                                     
                                                  
                                       
                      
                                    
 
                                         
                                                  

# Average-pool transformer hidden states using an attention mask.
def e5_average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


                                
                                                  
                                                
                                                                                       
                                        
                   

# Compute mean cosine concentration for a set of embeddings.
def get_s_mean(embeds_buffer):
    embeds_mat = torch.stack(embeds_buffer)             
    mu = F.normalize(embeds_mat.mean(dim=0), dim=0)
    E_norm = F.normalize(embeds_mat, dim=1)

    s_mean = torch.matmul(E_norm, mu).mean()
    return s_mean.item()


# Return indices that would sort a sequence of values.
def sorted_indices(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i])

# Average and normalize a list of tensor embeddings.
def average_embeds(tensors, eps=1e-12):
    x = torch.stack(tensors, dim=0)
    x = F.normalize(x, p=2, dim=1)
    m = x.mean(dim=0)
    return F.normalize(m, p=2, dim=0)


# Compute cosine similarities between a semantic node embedding and its member embeddings.
def sem_embed_sim(sem_node):
    sem_embed = sem_node.embed
    chunk_node_embed = torch.stack(sem_node.chunk_node_embed)
    sem_embed = sem_embed.unsqueeze(0)          
    sim = F.cosine_similarity(chunk_node_embed, sem_embed, dim=1)

    return sim

# Combine duplicate chunk similarities for a semantic node.
def sem_node_combine_sim(sim, sem_node, r=3):
    bucket = defaultdict(list)

                            
    for s, chunk in zip(sim, sem_node.chunk_node_list):
        bucket[chunk].append(float(s))

                                
    new_chunks = []
    combined_sim = []
    for chunk, sims in bucket.items():
        sims.sort(reverse=True)
        top = sims[:r]
        combined_sim.append(sum(top) / len(top))
        new_chunks.append(chunk)

    sem_node.chunk_node_list = new_chunks
    return combined_sim


# Return the most similar semantic node for an embedding.
def inspect_sem_nodes(embed, sem_node_list):
    B = torch.stack([sem_node.embed for sem_node in sem_node_list])          
    A = embed.unsqueeze(0)          
    similarities = F.cosine_similarity(A, B, dim=1)
    max_val, max_idx = torch.max(similarities, dim=0)
    return max_val, max_idx


# Plot clustered embeddings with PCA and t-SNE visualizations.
def plot_embeddings(embed, token, clusters):
    folder_name = time_stamp
    current_dir = os.getcwd()
    base_dir = os.path.join(current_dir, "model_embed_dis")
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    embeddings = torch.stack(embed)                 
    embeddings_np = embeddings.detach().cpu().numpy()

                                                  
                                                 

    pca = PCA(n_components=min(50, embeddings_np.shape[1]))
    reduced = pca.fit_transform(embeddings_np)

    N = len(embed)

                     
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

          
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

                     
    labels = [-1] * N
    for cid, idx_list in clusters.items():
        for idx in idx_list:
            if idx < N:
                labels[idx] = cid

    unique_clusters = sorted(set(labels))

    plt.figure()

          
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



# Print an approximate object size in megabytes.
def print_size_mb(obj, precision=2):
    size_bytes = asizeof.asizeof(obj)
    size_mb = size_bytes / (1024 * 1024)
    print(f"memory size: {size_mb:.{precision}f} MB")



# Return the index of the tensor most similar to a query tensor.
def max_cosine_similarity_index(query_tensor, tensor_list):

    if len(tensor_list) == 0:
        return None

    matrix = torch.stack(tensor_list)                 

    query_norm = F.normalize(query_tensor.unsqueeze(0), dim=1)          
    matrix_norm = F.normalize(matrix, dim=1)          

    similarities = torch.mm(query_norm, matrix_norm.t()).squeeze(0)        

    max_index = torch.argmax(similarities).item()

    return max_index

import json
from pathlib import Path

# Load HotpotQA distractor examples into a simplified structure.
def load_hotpot_distractor(file_path):

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for item in data:
        sample = {}

              
        sample["id"] = item["_id"]
        sample["question"] = item["question"]
        sample["answer"] = item["answer"]

                          
                                                
        supporting_pages = set()
        supporting_sentences = {}

        for title, sent_id in item["supporting_facts"]:
            supporting_pages.add(title)
            supporting_sentences.setdefault(title, []).append(sent_id)

        sample["supporting_pages"] = list(supporting_pages)
        sample["supporting_sentences"] = supporting_sentences

                         
                                                            
        documents = []

        for title, sentences in item["context"]:
            doc = {
                "title": title,
                "sentences": sentences,
                "text": " ".join(sentences)          
            }
            documents.append(doc)

        sample["documents"] = documents

        samples.append(sample)

    return samples

# Build a deduplicated document list from HotpotQA context records.
def build_global_document_list(file_path):

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_store = {}

    for item in data:
        for title, sentences in item["context"]:

                     
            text = " ".join(sentences).strip()

                                         
            if title not in doc_store:
                doc_store[title] = {
                    "doc_id": title,
                    "title": title,
                    "text": text,
                    "sentences": sentences
                }
            else:
                                           
                if len(text) > len(doc_store[title]["text"]):
                    doc_store[title]["text"] = text
                    doc_store[title]["sentences"] = sentences

    documents = list(doc_store.values())

    print(f"Total unique documents: {len(documents)}")

    return documents


import json
import pickle
from pathlib import Path


# Build or load cached HotpotQA documents and retrieval samples.
def build_hotpot_retrieval_dataset(file_path, num_samples=None):

    file_path = Path(file_path)

                                   
              
                                   

    cache_dir = Path("./hotpot_QA")
    cache_dir.mkdir(exist_ok=True)

    tag = "all" if num_samples is None else str(num_samples)

    documents_cache = cache_dir / f"hotpot_documents_{tag}.pkl"
    samples_cache = cache_dir / f"hotpot_samples_{tag}.pkl"

                                   
                 
                                   

    if documents_cache.exists() and samples_cache.exists():
        print("Loading cached dataset...")

        with open(documents_cache, "rb") as f:
            documents = pickle.load(f)

        with open(samples_cache, "rb") as f:
            samples = pickle.load(f)

        print(f"Loaded {len(documents)} documents")
        print(f"Loaded {len(samples)} samples")

        return documents, samples

                                   
            
                                   

    print("Building dataset from raw file...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]

                                   
                       
                                   

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

                                   
          
                                   

    with open(documents_cache, "wb") as f:
        pickle.dump(documents, f)

    with open(samples_cache, "wb") as f:
        pickle.dump(samples, f)

    print(f"Dataset cached to: {cache_dir}")

    return documents, samples


# Convert a list/array/tensor of embeddings into a contiguous (N, D) float32 numpy array.
def _embeddings_to_numpy_2d(embeddings):
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
    elif (
        isinstance(embeddings, (list, tuple))
        and len(embeddings) > 0
        and isinstance(embeddings[0], torch.Tensor)
    ):
        arr = torch.stack([e.detach().cpu().flatten() for e in embeddings]).numpy()
    else:
        arr = np.asarray(embeddings)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


# Select N representative indices via farthest-first traversal.
#
# Returns indices in selection order. The first index is the global medoid
# (start="medoid") or a seeded random point (start="random"). Each subsequent
# index is the unselected point whose minimum distance to the selected set is
# maximal. Distances are updated incrementally; only the first medoid step
# materializes a full (N, N) matrix.
def farthest_first_traversal(
    embeddings,
    N,
    metric="cosine",
    start="medoid",
    random_state=None,
    return_min_distances=False,
):
    arr = _embeddings_to_numpy_2d(embeddings)
    num_points = arr.shape[0]
    if num_points == 0 or N <= 0:
        return ([], []) if return_min_distances else []
    if N >= num_points:
        indices = list(range(num_points))
        return (indices, []) if return_min_distances else indices

    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        work = arr / norms

        def dist_to(i):
            return 1.0 - work @ work[i]
    elif metric == "euclidean":
        work = arr

        def dist_to(i):
            diff = work - work[i]
            return np.sqrt(np.einsum("ij,ij->i", diff, diff))
    else:
        raise ValueError(f"unsupported metric: {metric}")

    if start == "medoid":
        if metric == "cosine":
            dist_matrix = 1.0 - work @ work.T
        else:
            sq = np.einsum("ij,ij->i", work, work)
            dist_matrix = np.sqrt(
                np.maximum(sq[:, None] + sq[None, :] - 2.0 * work @ work.T, 0.0)
            )
        first = int(np.argmin(dist_matrix.sum(axis=1)))
    elif start == "random":
        rng = np.random.default_rng(random_state)
        first = int(rng.integers(0, num_points))
    else:
        raise ValueError(f"unsupported start: {start}")

    selected = [first]
    min_dist = dist_to(first).astype(np.float32, copy=False)
    selected_mask = np.zeros(num_points, dtype=bool)
    selected_mask[first] = True
    min_dist[first] = -np.inf
    step_distances = []

    while len(selected) < N:
        next_idx = int(np.argmax(min_dist))
        next_min = float(min_dist[next_idx])
        if not np.isfinite(next_min):
            break  # remaining points are duplicates of selected ones
        selected.append(next_idx)
        step_distances.append(next_min)
        selected_mask[next_idx] = True
        new_dist = dist_to(next_idx)
        np.minimum(min_dist, new_dist, out=min_dist)
        min_dist[next_idx] = -np.inf

    return (selected, step_distances) if return_min_distances else selected

    return documents, samples
