from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from text_processing import normalize_text


PHRASE_TYPE_ATOMIC = "atomic"
PHRASE_TYPE_COMPOSITIONAL = "compositional"
PHRASE_TYPE_SINGLE_TOKEN = "single_token"
PHRASE_TYPE_UNKNOWN = "unknown"

DEFAULT_IGNORED_ATOMIC_NER_LABELS = {
    "CARDINAL",
    "DATE",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "TIME",
}

DEFAULT_FIXED_COLLOCATIONS = {
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "neural network",
    "random forest",
    "support vector machine",
    "white house",
    "new york",
    "united states",
    "united kingdom",
    "american airlines",
    "world war",
    "world war i",
    "world war ii",
}


@dataclass(frozen=True)
class PhraseAnalysis:
    phrase_text: str
    phrase_text_norm: str
    phrase_type: str
    head_text: str | None = None
    head_text_norm: str | None = None
    head_start: int | None = None
    head_end: int | None = None
    modifier_texts: list[str] = field(default_factory=list)
    modifier_texts_norm: list[str] = field(default_factory=list)
    modifier_spans: list[tuple[int, int]] = field(default_factory=list)
    atomic_modifier_spans: list[dict] = field(default_factory=list)
    dependency_labels: list[tuple[str, str, str]] = field(default_factory=list)
    atomic_reason: str | None = None


class PhraseAnalyzer:
    """Classify extracted spans as atomic, compositional, or single-token phrases.

    The analyzer keeps LiteSemRAG's normalized surface forms for lookup keys, but
    derives heads and modifiers from the original chunk text and character span.
    """

    MODIFIER_DEPS = {
        "amod",
        "compound",
        "nummod",
        "poss",
        "appos",
        "acl",
        "relcl",
    }

    def __init__(
        self,
        nlp,
        title_aliases: Iterable[str] | None = None,
        fixed_collocations: Iterable[str] | None = None,
        ignored_atomic_ner_labels: Iterable[str] | None = None,
    ):
        self.nlp = nlp
        self.title_aliases = self._normalize_lookup(title_aliases)
        self.fixed_collocations = set(DEFAULT_FIXED_COLLOCATIONS)
        self.fixed_collocations.update(self._normalize_lookup(fixed_collocations))
        if ignored_atomic_ner_labels is None:
            ignored_atomic_ner_labels = DEFAULT_IGNORED_ATOMIC_NER_LABELS
        self.ignored_atomic_ner_labels = {
            str(label).upper() for label in ignored_atomic_ner_labels
        }

    @staticmethod
    def _normalize_lookup(values: Iterable[str] | None) -> set[str]:
        if values is None:
            return set()
        return {normalize_text(str(value).strip()) for value in values if str(value).strip()}

    def analyze(
        self,
        span_text: str,
        chunk_text: str | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        doc=None,
    ) -> PhraseAnalysis:
        """Analyze one span.

        Args:
            span_text: The LiteSemRAG span surface, usually already normalized.
            chunk_text: Original chunk text containing the span.
            span_start/span_end: Character offsets in ``chunk_text``. When present,
                these offsets are preferred over ``span_text`` for dependency parsing.
            doc: Optional spaCy Doc for ``chunk_text`` to avoid reparsing.
        """
        raw_text = self._resolve_raw_text(span_text, chunk_text, span_start, span_end)
        phrase_text_norm = normalize_text(span_text or raw_text)
        doc = self._ensure_doc(chunk_text, raw_text, doc)
        span = self._resolve_spacy_span(doc, raw_text, span_start, span_end)
        ent = self._find_matching_entity(doc, raw_text, span_start, span_end)

        atomic_reason = self._atomic_reason(phrase_text_norm, ent)
        if self._token_count(phrase_text_norm) <= 1 and atomic_reason is None:
            return PhraseAnalysis(
                phrase_text=raw_text,
                phrase_text_norm=phrase_text_norm,
                phrase_type=PHRASE_TYPE_SINGLE_TOKEN,
            )

        if span is not None:
            head_info = self._head_and_modifiers(span, doc)
        else:
            head_info = self._empty_head_info()

        if atomic_reason is not None:
            phrase_type = PHRASE_TYPE_ATOMIC
        elif span is not None and head_info["head_text"] is not None:
            phrase_type = PHRASE_TYPE_COMPOSITIONAL
        else:
            phrase_type = PHRASE_TYPE_UNKNOWN

        return PhraseAnalysis(
            phrase_text=raw_text,
            phrase_text_norm=phrase_text_norm,
            phrase_type=phrase_type,
            atomic_reason=atomic_reason,
            **head_info,
        )

    def _resolve_raw_text(self, span_text, chunk_text, span_start, span_end) -> str:
        if (
            chunk_text is not None
            and span_start is not None
            and span_end is not None
            and 0 <= span_start <= span_end <= len(chunk_text)
        ):
            return chunk_text[span_start:span_end].strip()
        return str(span_text).strip()

    def _ensure_doc(self, chunk_text, raw_text, doc):
        if doc is not None:
            return doc
        return self.nlp(chunk_text if chunk_text is not None else raw_text)

    def _resolve_spacy_span(self, doc, raw_text, span_start, span_end):
        if span_start is not None and span_end is not None:
            span = doc.char_span(span_start, span_end, alignment_mode="expand")
            if span is not None:
                return span

        target = normalize_text(raw_text)
        for noun_chunk in doc.noun_chunks:
            if normalize_text(noun_chunk.text) == target:
                return noun_chunk
        for ent in doc.ents:
            if normalize_text(ent.text) == target:
                return ent
        return None

    def _find_matching_entity(self, doc, raw_text, span_start, span_end):
        if span_start is not None and span_end is not None:
            for ent in doc.ents:
                if ent.start_char <= span_start and ent.end_char >= span_end:
                    if normalize_text(ent.text) == normalize_text(raw_text):
                        return ent

        target = normalize_text(raw_text)
        for ent in doc.ents:
            if normalize_text(ent.text) == target:
                return ent
        return None

    def _atomic_reason(self, phrase_text_norm, ent):
        if ent is not None and ent.label_.upper() not in self.ignored_atomic_ner_labels:
            return f"ner:{ent.label_}"
        if phrase_text_norm in self.title_aliases:
            return "title_alias"
        if phrase_text_norm in self.fixed_collocations:
            return "fixed_collocation"
        return None

    def _head_and_modifiers(self, span, doc):
        root = span.root
        modifier_tokens = [
            token
            for token in span
            if token.i != root.i and token.dep_ in self.MODIFIER_DEPS
        ]
        return {
            "head_text": root.text,
            "head_text_norm": normalize_text(root.lemma_ or root.text),
            "head_start": root.idx,
            "head_end": root.idx + len(root.text),
            "modifier_texts": [token.text for token in modifier_tokens],
            "modifier_texts_norm": [
                normalize_text(token.lemma_ or token.text) for token in modifier_tokens
            ],
            "modifier_spans": [
                (token.idx, token.idx + len(token.text)) for token in modifier_tokens
            ],
            "atomic_modifier_spans": self._find_atomic_subspans(span, doc, root),
            "dependency_labels": [
                (token.text, token.dep_, token.head.text) for token in span
            ],
        }

    def _find_atomic_subspans(self, span, doc, root):
        candidates = []
        seen = set()

        for ent in doc.ents:
            if not self._span_contains_char_span(span, ent.start_char, ent.end_char):
                continue
            if ent.start_char == span.start_char and ent.end_char == span.end_char:
                continue
            if ent.start_char <= root.idx < ent.end_char:
                continue
            if ent.label_.upper() in self.ignored_atomic_ner_labels:
                continue
            candidates.append((ent.start_char, ent.end_char, ent.text, f"ner:{ent.label_}"))

        span_tokens = list(span)
        for start_idx in range(len(span_tokens)):
            for end_idx in range(start_idx + 1, len(span_tokens) + 1):
                sub_tokens = span_tokens[start_idx:end_idx]
                if len(sub_tokens) < 2:
                    continue
                if any(token.i == root.i for token in sub_tokens):
                    continue
                start_char = sub_tokens[0].idx
                end_char = sub_tokens[-1].idx + len(sub_tokens[-1].text)
                text = doc.text[start_char:end_char]
                text_norm = normalize_text(text)
                reason = None
                if text_norm in self.title_aliases:
                    reason = "title_alias"
                elif text_norm in self.fixed_collocations:
                    reason = "fixed_collocation"
                if reason is not None:
                    candidates.append((start_char, end_char, text, reason))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[3]))
        filtered = []
        occupied = []
        for start_char, end_char, text, reason in candidates:
            key = (start_char, end_char, reason)
            if key in seen:
                continue
            if any(start_char >= lo and end_char <= hi for lo, hi in occupied):
                continue
            seen.add(key)
            occupied.append((start_char, end_char))
            filtered.append(
                {
                    "text": text,
                    "text_norm": normalize_text(text),
                    "start": start_char,
                    "end": end_char,
                    "reason": reason,
                }
            )
        return filtered

    @staticmethod
    def _span_contains_char_span(span, start_char, end_char):
        return span.start_char <= start_char and end_char <= span.end_char

    @staticmethod
    def _empty_head_info():
        return {
            "head_text": None,
            "head_text_norm": None,
            "head_start": None,
            "head_end": None,
            "modifier_texts": [],
            "modifier_texts_norm": [],
            "modifier_spans": [],
            "atomic_modifier_spans": [],
            "dependency_labels": [],
        }

    @staticmethod
    def _token_count(text: str) -> int:
        return len([word for word in str(text).split() if word])
