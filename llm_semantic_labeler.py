"""LLM-based semantic label selection with SQLite caching.

This module is designed as a notebook-friendly helper for validating span
semantics against Wikidata candidates. Cache entries are scoped to the target
span, nearby context words, model identity, and the exact candidate bank, so
changing candidate semantics automatically invalidates old judgments.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from local_llm import LocalLLMClient, LocalLLMConfig


DEFAULT_CACHE_PATH = Path("cache/llm_semantic_label_cache.sqlite3")
DEFAULT_CONTEXT_WORD_WINDOW = 12
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256

SYSTEM_PROMPT = """You are a precise semantic disambiguation assistant.
Choose exactly one Wikidata candidate that best matches the target span in the given context.
Return valid JSON only.
"""

USER_PROMPT_TEMPLATE = """Target span: {span_text}
Context excerpt: {context_text}
Local target text: {matched_text}

Candidate meanings:
{candidate_block}

Instructions:
1. Pick the single best candidate for the target span in this context.
2. Prefer semantic fit to the local context over surface similarity.
3. Return JSON only in this shape:
{{"selected_index": <integer>, "reason": "<short explanation>"}}
"""

BATCH_SYSTEM_PROMPT = """You are a precise semantic disambiguation assistant.
Choose exactly one Wikidata candidate that best matches the target span in each given context.
Return valid JSON only.
"""

BATCH_USER_PROMPT_TEMPLATE = """Candidate meanings:
{candidate_block}

Records to judge:
{record_block}

Instructions:
1. For each record, pick the single best candidate for matched_text in that context.
2. Prefer semantic fit to the local context over surface similarity.
3. Return JSON only in this shape:
{{"judgments": [{{"record_id": <integer>, "selected_index": <integer>, "reason": "<short explanation>"}}]}}
4. Include exactly one judgment for every input record_id and do not add extra record_id values.
5. Keep each reason under 12 words.
"""


@dataclass(frozen=True)
class SemanticCacheKey:
    normalized_span: str
    context_signature: str
    provider: str
    model: str
    candidate_bank_hash: str


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _normalize_text(text: str) -> str:
    return _normalize_space(text).lower()


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _find_token_span_from_local_span(
    context_text: str,
    local_span: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if local_span is None:
        return None
    start_char, end_char = local_span
    tokens = list(re.finditer(r"[A-Za-z0-9]+", context_text))
    overlap_indices = [
        idx
        for idx, match in enumerate(tokens)
        if not (match.end() <= start_char or match.start() >= end_char)
    ]
    if not overlap_indices:
        return None
    return overlap_indices[0], overlap_indices[-1] + 1


def build_context_signature(
    span_text: str,
    context_text: str,
    *,
    local_span: tuple[int, int] | None = None,
    context_word_window: int = DEFAULT_CONTEXT_WORD_WINDOW,
) -> str:
    normalized_span = _normalize_text(span_text)
    context_text = _normalize_space(context_text)
    if not context_text:
        return normalized_span

    token_matches = list(re.finditer(r"[A-Za-z0-9]+", context_text))
    token_words = [match.group(0).lower() for match in token_matches]
    if not token_words:
        return normalized_span

    token_span = _find_token_span_from_local_span(context_text, local_span)
    if token_span is None:
        span_tokens = _tokenize_words(span_text)
        if span_tokens:
            span_len = len(span_tokens)
            start_idx = None
            for idx in range(len(token_words) - span_len + 1):
                if token_words[idx:idx + span_len] == span_tokens:
                    start_idx = idx
                    break
            if start_idx is not None:
                token_span = (start_idx, start_idx + span_len)

    if token_span is None:
        clipped = token_words[: max(1, context_word_window * 2 + 1)]
        return f"{normalized_span} || {' '.join(clipped)}"

    start_idx, end_idx = token_span
    left = max(0, start_idx - max(0, int(context_word_window)))
    right = min(len(token_words), end_idx + max(0, int(context_word_window)))
    window_tokens = token_words[left:right]
    return f"{normalized_span} || {' '.join(window_tokens)}"


def _candidate_block(candidate_bank: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for idx, candidate in enumerate(candidate_bank):
        lines.append(
            f"[{idx}] entity_id={candidate.get('entity_id')} | "
            f"label={candidate.get('label')} | "
            f"description={candidate.get('description')} | "
            f"definition={candidate.get('definition')}"
        )
    return "\n".join(lines)


def _record_block(records: Iterable[Mapping[str, Any]]) -> str:
    blocks = []
    for record in records:
        blocks.append(
            "\n".join(
                [
                    f"record_id: {record['record_id']}",
                    f"span_text: {_normalize_space(record.get('span_text') or '')}",
                    f"matched_text: {_normalize_space(record.get('matched_text') or record.get('span_text') or '')}",
                    f"context: {_normalize_space(record.get('context_text') or '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def candidate_bank_hash(candidate_bank: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable hash for the ordered candidate semantics shown to LLM."""
    canonical_candidates = []
    for candidate in candidate_bank:
        canonical_candidates.append(
            {
                "entity_id": str(candidate.get("entity_id") or ""),
                "label": str(candidate.get("label") or ""),
                "description": _normalize_space(candidate.get("description") or ""),
                "definition": _normalize_space(candidate.get("definition") or ""),
                "hypothesis": _normalize_space(candidate.get("hypothesis") or ""),
            }
        )
    payload = json.dumps(
        canonical_candidates,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    if not text:
        raise ValueError("LLM returned empty text.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    excerpt = text[:1000]
    if len(text) > 1000:
        excerpt += "...<truncated>"
    raise ValueError(f"LLM output did not contain a valid JSON object: {excerpt!r}")


def _parse_selected_index(payload: Mapping[str, Any], candidate_bank: list[Mapping[str, Any]]) -> int:
    if "selected_index" in payload:
        selected_index = int(payload["selected_index"])
        if 0 <= selected_index < len(candidate_bank):
            return selected_index

    selected_entity_id = payload.get("selected_entity_id")
    if selected_entity_id is not None:
        selected_entity_id = str(selected_entity_id).strip()
        for idx, candidate in enumerate(candidate_bank):
            if str(candidate.get("entity_id")).strip() == selected_entity_id:
                return idx

    raise ValueError(
        f"LLM selection did not resolve to a valid candidate. payload={payload!r}"
    )


def _parse_batched_judgments(
    payload: Mapping[str, Any],
    batch_records: list[Mapping[str, Any]],
    candidate_bank: list[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError(f"LLM response missing judgments list. payload={payload!r}")

    by_record_id = {}
    for item in judgments:
        if not isinstance(item, Mapping):
            continue
        record_id = int(item.get("record_id"))
        selected_index = int(item.get("selected_index"))
        if not 0 <= selected_index < len(candidate_bank):
            raise ValueError(
                f"LLM selected invalid candidate index {selected_index}. payload={payload!r}"
            )
        reason = item.get("reason")
        by_record_id[record_id] = {
            "selected_index": selected_index,
            "selected_candidate": dict(candidate_bank[selected_index]),
            "reason": str(reason).strip() if reason is not None else None,
        }

    missing = [
        int(record["record_id"])
        for record in batch_records
        if int(record["record_id"]) not in by_record_id
    ]
    if missing:
        raise ValueError(f"LLM response missing judgments for record_id={missing}.")
    return by_record_id


def _ensure_cache_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_semantic_label_cache_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_span TEXT NOT NULL,
            context_signature TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            candidate_bank_hash TEXT NOT NULL,
            span_text TEXT NOT NULL,
            context_text TEXT NOT NULL,
            matched_text TEXT,
            selected_index INTEGER,
            selected_entity_id TEXT,
            selected_label TEXT,
            selected_description TEXT,
            raw_response TEXT NOT NULL,
            reason TEXT,
            created_at_utc TEXT NOT NULL,
            UNIQUE(normalized_span, context_signature, provider, model, candidate_bank_hash)
        )
        """
    )
    connection.commit()


def _open_cache(cache_path: Path | str = DEFAULT_CACHE_PATH) -> sqlite3.Connection:
    resolved_path = Path(cache_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path)
    _ensure_cache_schema(connection)
    return connection


def _lookup_cache_row(
    connection: sqlite3.Connection,
    cache_key: SemanticCacheKey,
) -> sqlite3.Row | None:
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        SELECT *
        FROM llm_semantic_label_cache_v2
        WHERE normalized_span = ?
          AND context_signature = ?
          AND provider = ?
          AND model = ?
          AND candidate_bank_hash = ?
        """,
        (
            cache_key.normalized_span,
            cache_key.context_signature,
            cache_key.provider,
            cache_key.model,
            cache_key.candidate_bank_hash,
        ),
    ).fetchone()
    return row


def _save_cache_row(
    connection: sqlite3.Connection,
    cache_key: SemanticCacheKey,
    *,
    span_text: str,
    context_text: str,
    matched_text: str,
    selected_index: int,
    selected_entity_id: str | None,
    selected_label: str | None,
    selected_description: str | None,
    raw_response: str,
    reason: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO llm_semantic_label_cache_v2 (
            normalized_span,
            context_signature,
            provider,
            model,
            candidate_bank_hash,
            span_text,
            context_text,
            matched_text,
            selected_index,
            selected_entity_id,
            selected_label,
            selected_description,
            raw_response,
            reason,
            created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(normalized_span, context_signature, provider, model, candidate_bank_hash)
        DO UPDATE SET
            span_text = excluded.span_text,
            context_text = excluded.context_text,
            matched_text = excluded.matched_text,
            selected_index = excluded.selected_index,
            selected_entity_id = excluded.selected_entity_id,
            selected_label = excluded.selected_label,
            selected_description = excluded.selected_description,
            raw_response = excluded.raw_response,
            reason = excluded.reason,
            created_at_utc = excluded.created_at_utc
        """,
        (
            cache_key.normalized_span,
            cache_key.context_signature,
            cache_key.provider,
            cache_key.model,
            cache_key.candidate_bank_hash,
            span_text,
            context_text,
            matched_text,
            int(selected_index),
            selected_entity_id,
            selected_label,
            selected_description,
            raw_response,
            reason,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    connection.commit()


def _resolve_cached_candidate(
    row: sqlite3.Row,
    candidate_bank: list[Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]] | None:
    cached_entity_id = row["selected_entity_id"]
    if cached_entity_id:
        for idx, candidate in enumerate(candidate_bank):
            if str(candidate.get("entity_id")) == str(cached_entity_id):
                return idx, candidate

    cached_index = row["selected_index"]
    if cached_index is not None:
        cached_index = int(cached_index)
        if 0 <= cached_index < len(candidate_bank):
            return cached_index, candidate_bank[cached_index]

    return None


def choose_wikidata_candidate_with_llm(
    *,
    span_text: str,
    context_text: str,
    candidate_bank: Iterable[Mapping[str, Any]],
    matched_text: str | None = None,
    local_span: tuple[int, int] | None = None,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    context_word_window: int = DEFAULT_CONTEXT_WORD_WINDOW,
    config: LocalLLMConfig | None = None,
    client: LocalLLMClient | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    candidate_bank_list = [dict(candidate) for candidate in candidate_bank]
    if not candidate_bank_list:
        raise ValueError("candidate_bank must be non-empty.")

    llm_client = client or LocalLLMClient(config)
    active_config = llm_client.config
    bank_hash = candidate_bank_hash(candidate_bank_list)
    normalized_span = _normalize_text(span_text)
    context_signature = build_context_signature(
        span_text,
        context_text,
        local_span=local_span,
        context_word_window=context_word_window,
    )
    cache_key = SemanticCacheKey(
        normalized_span=normalized_span,
        context_signature=context_signature,
        provider=active_config.provider,
        model=active_config.model,
        candidate_bank_hash=bank_hash,
    )

    with _open_cache(cache_path) as connection:
        cache_row = _lookup_cache_row(connection, cache_key)
        if cache_row is not None:
            cached_resolution = _resolve_cached_candidate(cache_row, candidate_bank_list)
            if cached_resolution is not None:
                cached_index, cached_candidate = cached_resolution
                return {
                    "selected_index": cached_index,
                    "selected_candidate": dict(cached_candidate),
                    "selected_entity_id": cached_candidate.get("entity_id"),
                    "selected_label": cached_candidate.get("label"),
                    "selected_description": cached_candidate.get("description"),
                    "reason": cache_row["reason"],
                    "raw_response": cache_row["raw_response"],
                    "cache_hit": True,
                    "context_signature": context_signature,
                    "candidate_bank_hash": bank_hash,
                    "provider": active_config.provider,
                    "model": active_config.model,
                }

        prompt = USER_PROMPT_TEMPLATE.format(
            span_text=_normalize_space(span_text),
            context_text=_normalize_space(context_text),
            matched_text=_normalize_space(matched_text or span_text),
            candidate_block=_candidate_block(candidate_bank_list),
        )
        raw_response = llm_client.complete(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = _extract_json_object(raw_response)
        selected_index = _parse_selected_index(payload, candidate_bank_list)
        selected_candidate = dict(candidate_bank_list[selected_index])
        reason = payload.get("reason")
        if reason is not None:
            reason = str(reason).strip()

        _save_cache_row(
            connection,
            cache_key,
            span_text=_normalize_space(span_text),
            context_text=_normalize_space(context_text),
            matched_text=_normalize_space(matched_text or span_text),
            selected_index=selected_index,
            selected_entity_id=str(selected_candidate.get("entity_id") or ""),
            selected_label=str(selected_candidate.get("label") or ""),
            selected_description=str(selected_candidate.get("description") or ""),
            raw_response=raw_response,
            reason=reason,
        )

    return {
        "selected_index": selected_index,
        "selected_candidate": selected_candidate,
        "selected_entity_id": selected_candidate.get("entity_id"),
        "selected_label": selected_candidate.get("label"),
        "selected_description": selected_candidate.get("description"),
        "reason": reason,
        "raw_response": raw_response,
        "cache_hit": False,
        "context_signature": context_signature,
        "candidate_bank_hash": bank_hash,
        "provider": active_config.provider,
        "model": active_config.model,
    }


def choose_wikidata_candidates_with_llm_batch(
    *,
    records: Iterable[Mapping[str, Any]],
    candidate_bank: Iterable[Mapping[str, Any]],
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    context_word_window: int = DEFAULT_CONTEXT_WORD_WINDOW,
    config: LocalLLMConfig | None = None,
    client: LocalLLMClient | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_batch_size: int = 10,
) -> list[dict[str, Any]]:
    record_list = [dict(record) for record in records]
    candidate_bank_list = [dict(candidate) for candidate in candidate_bank]
    if not record_list:
        return []
    if not candidate_bank_list:
        raise ValueError("candidate_bank must be non-empty.")

    llm_client = client or LocalLLMClient(config)
    active_config = llm_client.config
    bank_hash = candidate_bank_hash(candidate_bank_list)
    batch_size = max(1, int(max_batch_size or 1))

    prepared_records = []
    for record_id, record in enumerate(record_list):
        span_text = _normalize_space(record.get("span_text") or "")
        context_text = _normalize_space(record.get("context_text") or "")
        matched_text = _normalize_space(record.get("matched_text") or span_text)
        local_span = record.get("local_span")
        if local_span is not None:
            local_span = tuple(local_span)
        context_signature = build_context_signature(
            span_text,
            context_text,
            local_span=local_span,
            context_word_window=context_word_window,
        )
        cache_key = SemanticCacheKey(
            normalized_span=_normalize_text(span_text),
            context_signature=context_signature,
            provider=active_config.provider,
            model=active_config.model,
            candidate_bank_hash=bank_hash,
        )
        prepared_records.append(
            {
                "record_id": record_id,
                "span_text": span_text,
                "context_text": context_text,
                "matched_text": matched_text,
                "local_span": local_span,
                "context_signature": context_signature,
                "cache_key": cache_key,
            }
        )

    results_by_id: dict[int, dict[str, Any]] = {}
    uncached_records = []
    with _open_cache(cache_path) as connection:
        for prepared in prepared_records:
            cache_row = _lookup_cache_row(connection, prepared["cache_key"])
            if cache_row is not None:
                cached_resolution = _resolve_cached_candidate(cache_row, candidate_bank_list)
                if cached_resolution is not None:
                    cached_index, cached_candidate = cached_resolution
                    results_by_id[prepared["record_id"]] = {
                        "selected_index": cached_index,
                        "selected_candidate": dict(cached_candidate),
                        "selected_entity_id": cached_candidate.get("entity_id"),
                        "selected_label": cached_candidate.get("label"),
                        "selected_description": cached_candidate.get("description"),
                        "reason": cache_row["reason"],
                        "raw_response": cache_row["raw_response"],
                        "cache_hit": True,
                        "context_signature": prepared["context_signature"],
                        "candidate_bank_hash": bank_hash,
                        "provider": active_config.provider,
                        "model": active_config.model,
                    }
                    continue
            uncached_records.append(prepared)

        candidate_block = _candidate_block(candidate_bank_list)
        for batch_start in range(0, len(uncached_records), batch_size):
            batch_records = uncached_records[batch_start:batch_start + batch_size]
            prompt = BATCH_USER_PROMPT_TEMPLATE.format(
                candidate_block=candidate_block,
                record_block=_record_block(batch_records),
            )
            raw_response = llm_client.complete(
                prompt,
                system_prompt=BATCH_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            payload = _extract_json_object(raw_response)
            batch_judgments = _parse_batched_judgments(
                payload,
                batch_records,
                candidate_bank_list,
            )
            for prepared in batch_records:
                judgment = batch_judgments[prepared["record_id"]]
                selected_index = judgment["selected_index"]
                selected_candidate = dict(judgment["selected_candidate"])
                reason = judgment.get("reason")
                _save_cache_row(
                    connection,
                    prepared["cache_key"],
                    span_text=prepared["span_text"],
                    context_text=prepared["context_text"],
                    matched_text=prepared["matched_text"],
                    selected_index=selected_index,
                    selected_entity_id=str(selected_candidate.get("entity_id") or ""),
                    selected_label=str(selected_candidate.get("label") or ""),
                    selected_description=str(selected_candidate.get("description") or ""),
                    raw_response=raw_response,
                    reason=reason,
                )
                results_by_id[prepared["record_id"]] = {
                    "selected_index": selected_index,
                    "selected_candidate": selected_candidate,
                    "selected_entity_id": selected_candidate.get("entity_id"),
                    "selected_label": selected_candidate.get("label"),
                    "selected_description": selected_candidate.get("description"),
                    "reason": reason,
                    "raw_response": raw_response,
                    "cache_hit": False,
                    "context_signature": prepared["context_signature"],
                    "candidate_bank_hash": bank_hash,
                    "provider": active_config.provider,
                    "model": active_config.model,
                }

    return [results_by_id[record_id] for record_id in range(len(prepared_records))]
