"""LLM-driven Wikidata definition filtering, merging, and frequency reranking.

Given a term, this module fetches a fixed number of candidate senses from
Wikidata (no client-side filtering), then asks an LLM (local server or the
OpenAI API) to:

1. Drop candidates that are clearly unrelated to the term itself.
2. Merge candidates whose meanings substantially overlap.
3. Reorder the surviving senses from most to least common in everyday usage.

LLM judgments use the rich Wikipedia-derived ``detailed_description``; the
returned senses use the short Wikidata ``description`` (or a concise rewrite
when several candidates were merged). Results are cached in SQLite keyed
solely by the normalized term: once a term has been answered, the cached
answer is reused regardless of model, language, or candidate count.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_llm import LocalLLMClient, LocalLLMConfig
from utils import search_wikidata


DEFAULT_CACHE_PATH = Path("cache/wikidata_definition_filter_cache.sqlite3")
DEFAULT_NUM_CANDIDATES = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

DEFAULT_API_MODEL = "gpt-4o-mini"
DEFAULT_API_KEY_FILE = "API_KEY"


SYSTEM_PROMPT = """You are a precise lexicographer with strict semantic judgment.

You distinguish carefully between:
- core meanings of a word
- entities that the word directly refers to in common usage
- things merely named using the word

You prefer general, conceptually clean definitions over narrow or domain-specific ones.

You are conservative:
- do not drop important meanings
- do not merge unrelated meanings
- do not keep irrelevant named entities

You always return valid JSON only.
"""

USER_PROMPT_TEMPLATE = """Target term: {term}

Candidate senses:
{candidate_block}

---

Examples of correct behavior:

### Example 1
Target term: apple

Candidates:
0. apple — edible fruit of the apple tree
1. Apple — American multinational technology company
2. Apple Music — music streaming service
3. Apple Records — record label
4. Apple — a unisex given name

Correct output:
{{
  "definitions": [
    {{"source_indices": [0], "rewritten": ""}},
    {{"source_indices": [1], "rewritten": ""}}
  ]
}}

Explanation:
- Keep fruit (core meaning)
- Keep Apple Inc. (commonly referred to as "Apple")
- Drop sub-brands and named entities that cannot be called just "apple"
- Drop given name

---

### Example 2
Target term: director

Candidates:
0. film director — directs a film
1. theatrical director — directs a stage production
2. director — a manager in a company

Correct output:
{{
  "definitions": [
    {{
      "source_indices": [0, 1],
      "rewritten": "a person who directs the artistic or dramatic aspects of a performance or production"
    }},
    {{
      "source_indices": [2],
      "rewritten": ""
    }}
  ]
}}

Explanation:
- Merge film and theatre directors (same core function, different domains)
- Keep business meaning separate

---

### Example 3
Target term: amazon

Candidates:
0. Amazon — tropical rainforest
1. Amazon — American e-commerce company
2. Amazon River — river in South America
3. Amazon Prime — subscription service

Correct output:
{{
  "definitions": [
    {{"source_indices": [0], "rewritten": ""}},
    {{"source_indices": [2], "rewritten": ""}},
    {{"source_indices": [1], "rewritten": ""}}
  ]
}}

Explanation:
- Keep core meanings
- Keep major entity commonly referred to by the term
- Drop sub-brands

---

Now perform the task.

Tasks:

1. DROP candidates that do not define the term itself.

   DROP:
   - things merely named after the term (songs, films, minor works)
   - sub-brands or extended names (e.g., "Apple Music", "Amazon Prime")
   - entities that require additional words to identify (cannot be referred to by the term alone)
   - given names or family names (e.g., "unisex given name", "surname")

2. KEEP a proper noun ONLY if the term can be used ALONE to refer to it in natural language.

   Examples:
   - "Apple" → Apple Inc. (KEEP)
   - "Apple Music" → NOT KEEP
   - "Apple Records" → NOT KEEP

3. MERGE meanings when they share the same core concept.

   IMPORTANT:
   - If meanings differ only by domain (film, theatre, music, etc.), MERGE them
   - Prefer broader, more general definitions over narrow ones

   DO NOT merge:
   - different conceptual roles (e.g., artistic role vs business role)
   - unrelated meanings

4. REORDER by importance in everyday usage:

   - core dictionary meaning first
   - then widely known entities directly referred to by the term
   - then less common meanings

---

CRITICAL FINAL CHECK:

- Remove anything that cannot be referred to by the term alone
- Remove all name-related meanings (given name, surname)
- Ensure no major, widely known meaning is missing
- Ensure no unrelated meanings are merged
- Prefer general definitions over domain-specific ones

---

Respond with JSON only:

{{
  "definitions": [
    {{
      "source_indices": [<int>, ...],
      "rewritten": "<one-sentence definition or empty>"
    }}
  ]
}}
"""


@dataclass
class CandidateSense:
    index: int
    entity_id: str
    label: str
    description: str
    detailed_description: str


@dataclass
class FilteredDefinition:
    definition: str
    source_entity_ids: list[str]
    source_labels: list[str]
    is_merged: bool
    is_rewritten: bool


@dataclass
class FilterResult:
    term: str
    language: str
    num_candidates: int
    candidates: list[CandidateSense]
    definitions: list[FilteredDefinition]
    from_cache: bool
    raw_llm_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wikidata fetching
# ---------------------------------------------------------------------------

def fetch_wikidata_candidates(
    term: str,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    language: str = "en",
) -> list[CandidateSense]:
    """Fetch raw Wikidata candidates with no client-side filtering or reordering."""
    if not isinstance(term, str) or not term.strip():
        return []

    df = search_wikidata(
        term.strip(),
        language=language,
        limit=int(num_candidates),
        exact_match_text=False,
        exact_match_first=False,
        include_detailed_description=True,
        drop_missing_detailed_description=False,
        detailed_description_sentences=3,
        filter_name=False,
        label_contains_text=False,
    )

    candidates: list[CandidateSense] = []
    if df is None or df.empty:
        return candidates

    for idx, row in enumerate(df.itertuples(index=False)):
        candidates.append(
            CandidateSense(
                index=idx,
                entity_id=str(getattr(row, "id", "") or ""),
                label=str(getattr(row, "label", "") or ""),
                description=str(getattr(row, "description", "") or ""),
                detailed_description=str(getattr(row, "detailed_description", "") or ""),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# LLM client construction (local vs OpenAI API)
# ---------------------------------------------------------------------------

def build_llm_client(
    use_api: bool = False,
    *,
    model: str | None = None,
    api_key_file: str = DEFAULT_API_KEY_FILE,
) -> LocalLLMClient:
    """Construct an LLM client.

    ``use_api=False`` (default) uses the local OpenAI-compatible server defined
    by environment variables (see ``LocalLLMConfig.from_env``).

    ``use_api=True`` uses the OpenAI API with the key loaded from
    ``api_key_file`` (defaults to ``API_KEY`` in the project root) and the
    model defaults to ``gpt-4o-mini``.
    """
    if use_api:
        config = LocalLLMConfig.from_env(
            provider="openai",
            model=model or DEFAULT_API_MODEL,
            api_key_file=api_key_file,
        )
    else:
        config = LocalLLMConfig.from_env(provider="local", model=model)
    return LocalLLMClient(config)


# ---------------------------------------------------------------------------
# Cache (keyed on the term only)
# ---------------------------------------------------------------------------

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS wikidata_definition_filter_cache (
    term TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _normalize_term(term: str) -> str:
    return " ".join(str(term).strip().lower().split())


def _connect_cache(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(_CACHE_SCHEMA)
    conn.commit()
    return conn


def _cache_lookup(cache_path: Path, *, term: str) -> dict[str, Any] | None:
    conn = _connect_cache(cache_path)
    try:
        cursor = conn.execute(
            "SELECT payload FROM wikidata_definition_filter_cache WHERE term=?",
            (_normalize_term(term),),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None


def _cache_write(cache_path: Path, *, term: str, payload: dict[str, Any]) -> None:
    conn = _connect_cache(cache_path)
    try:
        conn.execute(
            "INSERT INTO wikidata_definition_filter_cache (term, payload, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(term) DO UPDATE SET "
            "payload=excluded.payload, updated_at=excluded.updated_at",
            (
                _normalize_term(term),
                json.dumps(payload, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM prompting and parsing
# ---------------------------------------------------------------------------

def _format_candidate_block(candidates: list[CandidateSense]) -> str:
    lines: list[str] = []
    for cand in candidates:
        text_for_judgment = cand.detailed_description.strip() or cand.description.strip()
        if not text_for_judgment:
            text_for_judgment = "(no description available)"
        lines.append(f"[{cand.index}] label: {cand.label}")
        lines.append(f"     description: {text_for_judgment}")
    return "\n".join(lines)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response.")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = re.sub(r"^json\s*", "", candidate, flags=re.IGNORECASE)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))


def _build_definitions(
    parsed: dict[str, Any],
    candidates: list[CandidateSense],
) -> list[FilteredDefinition]:
    by_index = {cand.index: cand for cand in candidates}
    raw_items = parsed.get("definitions")
    if not isinstance(raw_items, list):
        raise ValueError(f"LLM response missing 'definitions' list: {parsed!r}")

    definitions: list[FilteredDefinition] = []
    seen_indices: set[int] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_indices = item.get("source_indices") or []
        if not isinstance(raw_indices, list):
            continue
        indices: list[int] = []
        for value in raw_indices:
            try:
                idx_int = int(value)
            except (TypeError, ValueError):
                continue
            if idx_int in by_index and idx_int not in seen_indices:
                indices.append(idx_int)
                seen_indices.add(idx_int)
        if not indices:
            continue

        rewritten = str(item.get("rewritten", "") or "").strip()
        sources = [by_index[i] for i in indices]
        is_merged = len(sources) > 1

        if is_merged:
            definition_text = rewritten or " / ".join(
                src.description.strip() for src in sources if src.description.strip()
            )
            is_rewritten = bool(rewritten)
        else:
            definition_text = sources[0].description.strip()
            if not definition_text and rewritten:
                definition_text = rewritten
            is_rewritten = False

        definitions.append(
            FilteredDefinition(
                definition=definition_text,
                source_entity_ids=[src.entity_id for src in sources],
                source_labels=[src.label for src in sources],
                is_merged=is_merged,
                is_rewritten=is_rewritten,
            )
        )
    return definitions


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class WikidataDefinitionFilter:
    """Fetch Wikidata candidates and filter/merge/rerank them with an LLM."""

    def __init__(
        self,
        llm_client: LocalLLMClient | None = None,
        *,
        use_api: bool = False,
        api_model: str | None = None,
        api_key_file: str = DEFAULT_API_KEY_FILE,
        config: LocalLLMConfig | None = None,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if llm_client is None:
            if config is not None:
                llm_client = LocalLLMClient(config)
            else:
                llm_client = build_llm_client(
                    use_api=use_api,
                    model=api_model,
                    api_key_file=api_key_file,
                )
        self.llm_client = llm_client
        self.cache_path = Path(cache_path)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    @property
    def provider(self) -> str:
        return self.llm_client.config.provider

    @property
    def model(self) -> str:
        return self.llm_client.config.model

    def filter_definitions(
        self,
        term: str,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        language: str = "en",
        use_cache: bool = True,
    ) -> FilterResult:
        """Return filtered, merged, frequency-ranked senses for ``term``.

        The cache is keyed solely by the normalized term. When ``use_cache`` is
        ``True`` and any cached entry exists for the term, it is returned as-is
        regardless of the current model/language/num_candidates. When
        ``use_cache`` is ``False``, the LLM is re-run and the cached entry for
        this term is overwritten.
        """
        if not isinstance(term, str) or not term.strip():
            raise ValueError("term must be a non-empty string.")

        normalized = term.strip()

        if use_cache:
            cached = _cache_lookup(self.cache_path, term=normalized)
            if cached is not None:
                return _result_from_cache_payload(cached, from_cache=True)

        candidates = fetch_wikidata_candidates(
            normalized,
            num_candidates=num_candidates,
            language=language,
        )

        if not candidates:
            result = FilterResult(
                term=normalized,
                language=language,
                num_candidates=int(num_candidates),
                candidates=[],
                definitions=[],
                from_cache=False,
                raw_llm_response="",
                metadata={"note": "no_wikidata_candidates"},
            )
            _cache_write(
                self.cache_path,
                term=normalized,
                payload=_result_to_cache_payload(result),
            )
            return result

        prompt = USER_PROMPT_TEMPLATE.format(
            term=normalized,
            candidate_block=_format_candidate_block(candidates),
        )

        chat_kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        model_lower = self.model.lower()
        if not model_lower.startswith(("gpt-5", "o1", "o3", "o4")):
            chat_kwargs["temperature"] = self.temperature

        raw_response = self.llm_client.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            **chat_kwargs,
        )

        try:
            parsed = _extract_json_object(raw_response)
            definitions = _build_definitions(parsed, candidates)
            metadata: dict[str, Any] = {}
        except (ValueError, json.JSONDecodeError) as exc:
            definitions = []
            metadata = {"parse_error": str(exc)}

        metadata["provider"] = self.provider
        metadata["model"] = self.model

        result = FilterResult(
            term=normalized,
            language=language,
            num_candidates=int(num_candidates),
            candidates=candidates,
            definitions=definitions,
            from_cache=False,
            raw_llm_response=raw_response,
            metadata=metadata,
        )

        _cache_write(
            self.cache_path,
            term=normalized,
            payload=_result_to_cache_payload(result),
        )
        return result


def _result_to_cache_payload(result: FilterResult) -> dict[str, Any]:
    return {
        "term": result.term,
        "language": result.language,
        "num_candidates": result.num_candidates,
        "candidates": [cand.__dict__ for cand in result.candidates],
        "definitions": [defn.__dict__ for defn in result.definitions],
        "raw_llm_response": result.raw_llm_response,
        "metadata": result.metadata,
    }


def _result_from_cache_payload(payload: dict[str, Any], *, from_cache: bool) -> FilterResult:
    candidates = [CandidateSense(**c) for c in payload.get("candidates", [])]
    definitions = [FilteredDefinition(**d) for d in payload.get("definitions", [])]
    return FilterResult(
        term=payload.get("term", ""),
        language=payload.get("language", "en"),
        num_candidates=int(payload.get("num_candidates", 0)),
        candidates=candidates,
        definitions=definitions,
        from_cache=from_cache,
        raw_llm_response=payload.get("raw_llm_response", ""),
        metadata=payload.get("metadata", {}) or {},
    )
