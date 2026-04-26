from __future__ import annotations

from typing import Any

import requests


WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"

DEFAULT_HEADERS = {
    "User-Agent": "WikidataExplorerNotebook/1.0 (https://www.wikidata.org/)"
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


def _safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Safely call an HTTP JSON endpoint and return parsed JSON (or empty dict on failure)."""
    try:
        response = requests.get(url, params=params, headers=headers or DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return {}
    except ValueError as exc:
        print(f"Invalid JSON response: {exc}")
        return {}


def _normalize_extract_text(text: str) -> str:
    """Collapse Wikipedia extract whitespace into one readable paragraph."""
    if not isinstance(text, str):
        return ""
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    return " ".join(parts)


def _clean_entity_ids(entity_ids: list[str] | tuple[str, ...]) -> list[str]:
    cleaned_ids: list[str] = []
    for entity_id in entity_ids:
        if not isinstance(entity_id, str):
            continue
        normalized = entity_id.strip()
        if normalized and normalized not in cleaned_ids:
            cleaned_ids.append(normalized)
    return cleaned_ids


def fetch_entities(
    entity_ids: list[str] | tuple[str, ...],
    language: str = "en",
    props: str = "sitelinks",
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, dict[str, Any]]:
    """Fetch a batch of Wikidata entities keyed by entity ID."""
    cleaned_ids = _clean_entity_ids(entity_ids)
    if not cleaned_ids:
        return {}

    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(cleaned_ids),
        "languages": language,
        "props": props,
    }
    data = _safe_get_json(WIKIDATA_API_URL, params=params, headers=headers, timeout=timeout)
    entities = data.get("entities", {}) if isinstance(data, dict) else {}
    if not isinstance(entities, dict):
        return {}

    results: dict[str, dict[str, Any]] = {}
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        if entity.get("missing") == "":
            continue
        results[entity_id] = entity
    return results


def extract_best_wikipedia_title(entity: dict[str, Any] | None, language: str = "en") -> str:
    """Pick the best linked Wikipedia page title for an entity."""
    if not isinstance(entity, dict):
        return ""

    sitelinks = entity.get("sitelinks", {})
    if not isinstance(sitelinks, dict):
        return ""

    preferred_sites = [f"{language}wiki"]
    if language != "en":
        preferred_sites.append("enwiki")

    for site in preferred_sites:
        info = sitelinks.get(site)
        if isinstance(info, dict):
            title = info.get("title", "")
            if isinstance(title, str) and title.strip():
                return title.strip()

    for site_name, info in sitelinks.items():
        if not isinstance(info, dict):
            continue
        if not site_name.endswith("wiki"):
            continue
        if site_name in NON_WIKIPEDIA_SITELINK_SITES:
            continue
        title = info.get("title", "")
        if isinstance(title, str) and title.strip():
            return title.strip()

    return ""


def fetch_wikipedia_intro(
    title: str,
    language: str = "en",
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    sentences: int = 3,
) -> str:
    """Fetch a readable lead summary for a Wikipedia title via the Action API."""
    if not isinstance(title, str) or not title.strip():
        return ""

    params: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title.strip(),
        "redirects": 1,
        "explaintext": 1,
        "exintro": 1,
    }
    if isinstance(sentences, int) and sentences > 0:
        params["exsentences"] = sentences

    url = WIKIPEDIA_API_URL_TEMPLATE.format(language=language)
    data = _safe_get_json(url, params=params, headers=headers, timeout=timeout)
    pages = data.get("query", {}).get("pages", {}) if isinstance(data, dict) else {}
    if not isinstance(pages, dict):
        return ""

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        extract = page.get("extract", "")
        normalized = _normalize_extract_text(extract)
        if normalized:
            return normalized
    return ""


def fetch_detailed_descriptions_for_entities(
    entity_ids: list[str] | tuple[str, ...],
    language: str = "en",
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    sentences: int = 3,
) -> dict[str, str]:
    """Fetch Wikipedia-backed detailed descriptions for a list of Wikidata entity IDs."""
    cleaned_ids = _clean_entity_ids(entity_ids)
    if not cleaned_ids:
        return {}

    entities = fetch_entities(
        cleaned_ids,
        language=language,
        props="sitelinks",
        headers=headers,
        timeout=timeout,
    )

    description_by_id: dict[str, str] = {}
    description_by_title: dict[str, str] = {}

    for entity_id in cleaned_ids:
        entity = entities.get(entity_id)
        title = extract_best_wikipedia_title(entity, language=language)
        if not title:
            description_by_id[entity_id] = ""
            continue

        if title not in description_by_title:
            description_by_title[title] = fetch_wikipedia_intro(
                title,
                language=language,
                headers=headers,
                timeout=timeout,
                sentences=sentences,
            )
        description_by_id[entity_id] = description_by_title[title]

    return description_by_id
