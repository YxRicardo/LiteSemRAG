"""LLM-driven Wikidata definition filtering, merging, and reranking.

Given a term, this module fetches a fixed number of candidate senses from
Wikidata, applies the same span-aware candidate rules used by the non-LLM
semantic-description path, then asks an LLM (local server or the OpenAI API)
to produce the same cleaned candidate set shape as
``jupyter_notebooks/wikidata_llm_candidate_merge_experiment.ipynb``:

1. Drop candidates that are not plausible senses of the term.
2. Merge candidates whose hypotheses would behave the same for retrieval.
3. Keep senses separate when they imply meaningfully different contexts.

Results are cached in SQLite keyed solely by the normalized term: once a term
has been answered, the cached answer is reused regardless of model, language,
or candidate count.
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
from utils import build_wikidata_candidate_bank, load_wikidata_definition_candidates


DEFAULT_CACHE_PATH = Path("cache/wikidata_definition_filter_cache.sqlite3")
DEFAULT_NUM_CANDIDATES = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

DEFAULT_API_MODEL = "gpt-5.4-mini"
DEFAULT_API_KEY_FILE = "API_KEY"
WIKIDATA_CANDIDATE_FILTER_VERSION = "merge_prompt_v4_label_window_plus2"


SYSTEM_PROMPT = """You are a careful lexical semantics annotator.
Your task is to clean candidate word senses from Wikidata for a semantic retrieval system.
Merge candidates when they refer to the same or nearly the same meaning, even if their labels or wording differ.
Keep candidates separate when they would lead to different retrieval behavior in context.
Prefer concise, concrete sense descriptions.
Return only valid JSON."""

USER_PROMPT_TEMPLATE = """Merge near-duplicate candidate senses for the target word.

Rules:
1. Merge candidates only when their hypotheses mean the same or nearly the same thing for retrieval.
2. Keep candidates separate when they describe meaningfully different contextual senses.
3. Discard candidates that are too vague, redundant, or not a plausible sense of the target word.
4. Output only valid JSON with keys: word, merged_senses, discarded_candidates, notes.

Each merged_senses item must contain:
- sense_id: a short stable id such as s1, s2, s3
- canonical_label: short label for the merged sense
- merged_description: one sentence describing the merged meaning
- source_candidate_ids: list of integer candidate_id values that were merged
- merge_rationale: one short sentence

Each discarded_candidates item must contain:
- candidate_id
- reason

Candidate data:
{candidate_payload}"""


@dataclass
class CandidateSense:
    index: int
    entity_id: str
    label: str
    description: str
    detailed_description: str
    definition: str = ""
    hypothesis: str = ""


@dataclass
class FilteredDefinition:
    definition: str
    source_entity_ids: list[str]
    source_labels: list[str]
    is_merged: bool
    is_rewritten: bool
    canonical_label: str = ""
    source_candidate_ids: list[int] = field(default_factory=list)
    merge_rationale: str = ""


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
    """Fetch Wikidata candidates using the shared candidate-bank path.

    This intentionally mirrors ``wikidata_llm_candidate_merge_experiment.ipynb``:
    first call ``utils.load_wikidata_definition_candidates`` without LLM
    filtering, then convert the returned rows with
    ``utils.build_wikidata_candidate_bank`` so the LLM sees the same
    hypotheses as the experiment notebook.
    """
    if not isinstance(term, str) or not term.strip():
        return []

    normalized_term = term.strip()
    if language != "en":
        raise ValueError(
            "WikidataDefinitionFilter currently mirrors the English shared "
            "candidate-bank path used by wikidata_llm_candidate_merge_experiment.ipynb."
        )

    candidates: list[CandidateSense] = []
    try:
        candidates_df, definition_column = load_wikidata_definition_candidates(
            normalized_term,
            use_detailed_description=False,
            exact_match_text=False,
            exact_match_first=False,
            limit=int(num_candidates),
            filter_name=True,
            require_detailed_description=False,
            target_candidate_count=int(num_candidates),
            use_llm_filter=False,
        )
    except ValueError:
        return candidates

    candidate_bank = build_wikidata_candidate_bank(candidates_df, definition_column)
    if not candidate_bank:
        return candidates

    for idx, candidate in enumerate(candidate_bank):
        candidates.append(
            CandidateSense(
                index=idx,
                entity_id=str(candidate.get("entity_id", "") or ""),
                label=str(candidate.get("label", "") or ""),
                description=str(candidate.get("description", "") or ""),
                detailed_description="",
                definition=str(candidate.get("definition", "") or ""),
                hypothesis=str(candidate.get("hypothesis", "") or ""),
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
    model defaults to ``gpt-5.4-mini``.
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

def _definition_to_hypothesis(definition: str) -> str:
    cleaned = str(definition or "").strip()
    if cleaned.endswith((".", "!", "?")):
        cleaned = cleaned[:-1]
    if not cleaned:
        cleaned = "a candidate sense"
    return f"It refers to {cleaned}."


def _compact_candidates_for_prompt(candidates: list[CandidateSense]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for cand in candidates:
        hypothesis = cand.hypothesis.strip()
        if not hypothesis:
            text_for_judgment = cand.definition.strip() or cand.description.strip()
            if not text_for_judgment:
                text_for_judgment = "(no description available)"
            hypothesis = _definition_to_hypothesis(text_for_judgment)
        compact.append(
            {
                "candidate_id": int(cand.index) + 1,
                "label": cand.label,
                "hypothesis": hypothesis,
            }
        )
    return compact


def _build_merge_prompt(term: str, candidates: list[CandidateSense]) -> str:
    payload = {
        "word": term,
        "candidate_senses": _compact_candidates_for_prompt(candidates),
    }
    return USER_PROMPT_TEMPLATE.format(
        candidate_payload=json.dumps(payload, ensure_ascii=False, indent=2)
    )


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
    by_candidate_id = {cand.index + 1: cand for cand in candidates}

    raw_merged_senses = parsed.get("merged_senses")
    if isinstance(raw_merged_senses, list):
        definitions: list[FilteredDefinition] = []
        used_candidate_ids: set[int] = set()
        for item in raw_merged_senses:
            if not isinstance(item, dict):
                continue
            definition_text = str(item.get("merged_description", "") or "").strip()
            if not definition_text:
                definition_text = str(item.get("definition", "") or "").strip()
            if not definition_text:
                continue

            raw_ids = item.get("source_candidate_ids") or []
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            source_candidate_ids: list[int] = []
            sources: list[CandidateSense] = []
            for value in raw_ids:
                try:
                    candidate_id = int(value)
                except (TypeError, ValueError):
                    continue
                source = by_candidate_id.get(candidate_id)
                if source is None or candidate_id in used_candidate_ids:
                    continue
                source_candidate_ids.append(candidate_id)
                sources.append(source)
                used_candidate_ids.add(candidate_id)

            canonical_label = str(item.get("canonical_label", "") or "").strip()
            merge_rationale = str(item.get("merge_rationale", "") or "").strip()
            definitions.append(
                FilteredDefinition(
                    definition=definition_text,
                    source_entity_ids=[src.entity_id for src in sources],
                    source_labels=[src.label for src in sources],
                    is_merged=len(sources) > 1,
                    is_rewritten=True,
                    canonical_label=canonical_label,
                    source_candidate_ids=source_candidate_ids,
                    merge_rationale=merge_rationale,
                )
            )
        return definitions

    raw_items = parsed.get("definitions")
    if not isinstance(raw_items, list):
        raise ValueError(
            f"LLM response missing 'merged_senses' or 'definitions' list: {parsed!r}"
        )

    definitions: list[FilteredDefinition] = []
    seen_indices: set[int] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        direct_definition = str(item.get("definition", "") or "").strip()
        if direct_definition:
            definitions.append(
                FilteredDefinition(
                    definition=direct_definition,
                    source_entity_ids=[],
                    source_labels=[],
                    is_merged=False,
                    is_rewritten=True,
                    canonical_label=str(item.get("canonical_label", "") or "").strip(),
                )
            )
            continue

        # Backward-compatible parser for older cached or experimental outputs.
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
                canonical_label=str(item.get("canonical_label", "") or "").strip(),
                source_candidate_ids=[i + 1 for i in indices],
                merge_rationale=str(item.get("merge_rationale", "") or "").strip(),
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
        write_cache: bool = True,
    ) -> FilterResult:
        """Return filtered, merged, frequency-ranked senses for ``term``.

        The cache is keyed solely by the normalized term. When ``use_cache`` is
        ``True`` and any cached entry exists for the term, it is returned as-is
        regardless of the current model/language/num_candidates. When
        ``use_cache`` is ``False``, the LLM is re-run and the cached entry for
        this term is overwritten unless ``write_cache`` is also ``False``.
        """
        if not isinstance(term, str) or not term.strip():
            raise ValueError("term must be a non-empty string.")

        normalized = term.strip()

        if use_cache:
            cached = _cache_lookup(self.cache_path, term=normalized)
            if cached is not None:
                cached_result = _result_from_cache_payload(cached, from_cache=True)
                if (
                    cached_result.metadata.get("wikidata_candidate_filter")
                    == WIKIDATA_CANDIDATE_FILTER_VERSION
                ):
                    return cached_result

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
                metadata={
                    "note": "no_wikidata_candidates",
                    "wikidata_candidate_filter": WIKIDATA_CANDIDATE_FILTER_VERSION,
                },
            )
            if write_cache:
                _cache_write(
                    self.cache_path,
                    term=normalized,
                    payload=_result_to_cache_payload(result),
                )
            return result

        prompt = _build_merge_prompt(normalized, candidates)

        chat_kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        model_lower = self.model.lower()
        if not model_lower.startswith(("gpt-5", "o1", "o3", "o4")):
            chat_kwargs["temperature"] = self.temperature

        total_tokens_before = self.llm_client.total_tokens
        raw_response = self.llm_client.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            **chat_kwargs,
        )
        usage = self.llm_client.last_usage
        api_wait_wall_time = self.llm_client.last_api_wait_wall_time
        total_tokens_after = self.llm_client.total_tokens

        try:
            parsed = _extract_json_object(raw_response)
            definitions = _build_definitions(parsed, candidates)
            metadata: dict[str, Any] = {
                "discarded_candidates": parsed.get("discarded_candidates", []),
                "notes": parsed.get("notes", ""),
                "prompt_style": "wikidata_llm_candidate_merge_experiment",
            }
        except (ValueError, json.JSONDecodeError) as exc:
            definitions = []
            metadata = {"parse_error": str(exc)}

        metadata["provider"] = self.provider
        metadata["model"] = self.model
        metadata["wikidata_candidate_filter"] = WIKIDATA_CANDIDATE_FILTER_VERSION
        metadata["candidate_count_after_rule_filter"] = len(candidates)
        metadata["token_usage"] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "total_tokens_delta": total_tokens_after - total_tokens_before,
        }
        metadata["api_wait_wall_time"] = api_wait_wall_time

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

        if write_cache:
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
