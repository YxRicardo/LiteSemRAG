"""LLM-based semantic label selection with SQLite caching.

This module is designed as a notebook-friendly helper for validating span
semantics against Wikidata candidates. The cache key intentionally focuses on
the target span and its nearby context words so the same semantic choice can be
reused across notebooks without repeating LLM calls.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class SemanticCacheKey:
    normalized_span: str
    context_signature: str
    provider: str
    model: str


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


def _extract_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    if not text:
        raise ValueError("LLM returned empty text.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM output did not contain a JSON object: {text!r}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM JSON was not an object: {text!r}")
    return parsed


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


def _ensure_cache_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_semantic_label_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_span TEXT NOT NULL,
            context_signature TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
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
            UNIQUE(normalized_span, context_signature, provider, model)
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
        FROM llm_semantic_label_cache
        WHERE normalized_span = ?
          AND context_signature = ?
          AND provider = ?
          AND model = ?
        """,
        (
            cache_key.normalized_span,
            cache_key.context_signature,
            cache_key.provider,
            cache_key.model,
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
        INSERT INTO llm_semantic_label_cache (
            normalized_span,
            context_signature,
            provider,
            model,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(normalized_span, context_signature, provider, model)
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
        "provider": active_config.provider,
        "model": active_config.model,
    }
