import fitz
import os
import torch
import torch.nn.functional as F
import re


# 方案A 开关：保留原始大小写，让 spaCy 在真实大小写文本上做 NER/POS/依存分析。
# 默认 True = 方案A（保留大小写，NER/POS/依存质量更好，PROPN 不做复数还原）。
# 设 LITESEM_PRESERVE_CASE=0 可退回方案B（旧行为：整体小写、仅保留缩写与多词专名的大小写）。
# token/phrase 的匹配键仍由下游 normalize_text 统一小写（缩写除外），所以大小写仅影响
# 语言学分析质量，不改变匹配口径。可调用 set_preserve_case() 在进程内切换。
# 索引端与查询端必须使用同一设置——切换后需用相同设置重建索引。
PRESERVE_CASE = os.environ.get("LITESEM_PRESERVE_CASE", "1").strip().lower() in {"1", "true", "yes"}


def set_preserve_case(value: bool) -> None:
    global PRESERVE_CASE
    PRESERVE_CASE = bool(value)


# Normalize raw text by standardizing whitespace, quotes, and casing while preserving selected proper-case spans.
def clean_text(text):

    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    if PRESERVE_CASE:
        # 方案A：不做整体小写，只规整空白并去掉引号，保留原始大小写交给 spaCy。
        text = re.sub(r"[\"']", "", text)
        return text.strip()

    placeholders = {}
    counter = 0

                           
    # Stash matched spans behind unique placeholders so the later lower-casing
    # skips them. Use re.sub (a single positional pass) instead of str.replace,
    # which is an unbounded substring swap and would corrupt overlapping
    # matches, e.g. "US USB" -> "US USb".
    def _stash(match):
        nonlocal counter
        placeholder = f"__PH_{counter}__"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    text = re.sub(r'\b[A-Z]{2,}\b', _stash, text)

                                
    text = re.sub(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', _stash, text)

               
    text = re.sub(r"[\"']", "", text)

            
    text = text.lower()

              
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder.lower(), original)

    return text.strip()


# Group sentences into chunks that stay under the configured token budget.
def chunk_by_sentences(text, nlp, max_tokens=200):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    token_count = 0

    for sent in doc.sents:
        sent_len = len(sent)

        if token_count + sent_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                token_count = 0

        current_chunk.append(sent.text)
        token_count += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Read a document and split it into token-limited text chunks.
def split_doc(doc_name, nlp, max_tokens=300):
    with fitz.open(doc_name) as pdf:
        text = "".join(page.get_text() for page in pdf)

    text = clean_text(text)
    chunks = chunk_by_sentences(text, nlp, max_tokens=max_tokens)

    return chunks


# Format a span-like item into readable debug text.
def _debug_item_text(item):
    return getattr(item, "text", str(item))


# Build a compact debug prefix for span extraction messages.
def _debug_prefix(item_type, stage=None):
    if stage:
        return f"[{stage}][{item_type}]"
    return f"[{item_type}]"


# Print a span extraction stage header when debug mode is enabled.
def _debug_stage(stage):
    print(f"\n=== {stage} ===")


# Print a debug message for a discarded span candidate.
def _debug_skip(item_type, item, reason, stage=None):
    print(f"{_debug_prefix(item_type, stage)} Skipping '{_debug_item_text(item)}': {reason}")


# Print a debug message for a retained span candidate.
def _debug_keep(item_type, item, reason="passed all filters", stage=None):
    return


# Print span candidates with their labels and character offsets.
def _debug_dump_spans(name, spans):
    print(f"[final_output] {name} ({len(spans)} items):")
    if not spans:
        print("  (none)")
        return

    for index, (text, start_char, end_char) in enumerate(spans, start=1):
        print(f"  {index}. '{text}' [{start_char}, {end_char})")


PROTECTED_ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "NORP",
}

DISCARDED_PHRASE_ENTITY_LABELS = {
    "DATE",
}

# Dependency labels that mark a clausal subject. clean_text lowercases the whole
# sentence before spaCy runs, which destroys the casing NER/POS rely on: a title-case
# subject like "Frozen" gets re-tagged ADJ/VERB and dropped by the POS gate below, even
# though it is the most discriminative term in the query. The dependency parse survives
# lowercasing far better, so we keep subject tokens even when their POS is not NOUN/PROPN.
SUBJECT_DEPS = {
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
}

ROMAN_NUMERAL_RE = re.compile(r"(?i)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")


# Collect valid named entities and spans that should influence phrase filtering.
def _collect_valid_entities(doc, discard_no_word=False, debug_mode=False, debug_stage=None):
    entity_spans = []
    token_intervals = []
    char_intervals = []

    for ent in doc.ents:
        if ent.label_.upper() in DISCARDED_PHRASE_ENTITY_LABELS:
            if debug_mode:
                _debug_skip(
                    "entity",
                    ent,
                    f"{ent.label_} entities are discarded as phrase candidates",
                    stage=debug_stage,
                )
            continue

        if all_words_are_digits(ent.text):
            if debug_mode:
                _debug_skip("entity", ent, "every word in the entity is numeric", stage=debug_stage)
            continue

        if _is_single_word_quantifier_span(ent):
            if debug_mode:
                _debug_skip(
                    "entity",
                    ent,
                    "single-word entity is a quantifier-like cardinal or ordinal token",
                    stage=debug_stage,
                )
            continue

        if discard_no_word and re.search(r"[a-zA-Z]", ent.text) is None:
            if debug_mode:
                _debug_skip(
                    "entity",
                    ent,
                    "discard_no_word=True and the entity contains no alphabetic characters",
                    stage=debug_stage,
                )
            continue

        if debug_mode:
            _debug_keep("entity", ent, stage=debug_stage)

        preserve_leading_article = ent.label_.upper() == "WORK_OF_ART"
        entity_spans.append(
            (
                normalize_text(
                    ent.text,
                    preserve_leading_article=preserve_leading_article,
                ),
                ent.start_char,
                ent.end_char,
            )
        )
        token_intervals.append((ent.start, ent.end))
        char_intervals.append((ent.start_char, ent.end_char))

    return entity_spans, token_intervals, char_intervals


def _is_protected_entity(ent):
    return ent.label_.upper() in PROTECTED_ENTITY_LABELS


def _collect_protected_entity_records(doc, discard_no_word=False):
    records = []
    for ent in doc.ents:
        if not _is_protected_entity(ent):
            continue
        if all_words_are_digits(ent.text):
            continue
        if discard_no_word and re.search(r"[a-zA-Z]", ent.text) is None:
            continue
        records.append(
            {
                "start": ent.start,
                "end": ent.end,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "text": ent.text,
                "label": ent.label_,
            }
        )
    return records


def _append_protected_span_record(records, span, label):
    span_key = (span.start, span.end, span.start_char, span.end_char)
    for record in records:
        record_key = (
            record["start"],
            record["end"],
            record["start_char"],
            record["end_char"],
        )
        if record_key == span_key:
            return

    records.append(
        {
            "start": span.start,
            "end": span.end,
            "start_char": span.start_char,
            "end_char": span.end_char,
            "text": span.text,
            "label": label,
        }
    )


def _trim_residual_span(span):
    start = span.start
    end = span.end
    while start < end and (span.doc[start].is_stop or span.doc[start].is_punct):
        start += 1
    while start < end and (span.doc[end - 1].is_stop or span.doc[end - 1].is_punct):
        end -= 1
    return span.doc[start:end]


def _trim_leading_poss(span):
    start = span.start
    end = span.end
    doc = span.doc
    while start < end and doc[start].dep_ == "poss":
        start += 1
    if start >= end:
        return span
    return doc[start:end]


def _normalize_phrase_from_span(span, *, preserve_leading_article=False):
    trimmed = _trim_leading_poss(span)
    phrase_text = trimmed.text.strip()
    return (
        normalize_text(phrase_text, preserve_leading_article=preserve_leading_article),
        trimmed.start_char,
        trimmed.end_char,
        trimmed.start,
        trimmed.end,
    )


def _protected_entities_inside_span(span, protected_entity_records):
    return [
        record for record in protected_entity_records
        if span.start <= record["start"] and record["end"] <= span.end
    ]


def _split_span_by_protected_entities(span, protected_entity_records):
    contained_entities = _protected_entities_inside_span(span, protected_entity_records)
    if not contained_entities:
        return []

    residual_spans = []
    cursor = span.start
    for record in sorted(contained_entities, key=lambda item: (item["start"], item["end"])):
        if cursor < record["start"]:
            residual = _trim_residual_span(span.doc[cursor:record["start"]])
            if len(residual):
                residual_spans.append(residual)
        cursor = max(cursor, record["end"])

    if cursor < span.end:
        residual = _trim_residual_span(span.doc[cursor:span.end])
        if len(residual):
            residual_spans.append(residual)

    return residual_spans


def _token_inside_protected_entity(token, protected_entity_records):
    for record in protected_entity_records:
        if record["start"] <= token.i < record["end"]:
            return True
    return False


def _find_protected_span_record(protected_entity_records, start_char, end_char):
    for record in protected_entity_records:
        if record["start_char"] == start_char and record["end_char"] == end_char:
            return record
    return None


def _find_entity_for_char_span(doc, start_char, end_char):
    for ent in doc.ents:
        if ent.start_char == start_char and ent.end_char == end_char:
            return ent
    return None


def _build_phrase_audit_record(
    doc,
    phrase_text,
    start_char,
    end_char,
    source,
    protected_entity_records,
):
    ent = _find_entity_for_char_span(doc, start_char, end_char)
    protected_record = _find_protected_span_record(
        protected_entity_records,
        start_char,
        end_char,
    )
    protected_reason = None
    if protected_record is not None:
        protected_reason = protected_record.get("label")

    raw_text = doc.text[start_char:end_char]
    return {
        "phrase": phrase_text,
        "raw_text": raw_text,
        "start_char": start_char,
        "end_char": end_char,
        "source": source,
        "protected": protected_record is not None,
        "protected_reason": protected_reason,
        "entity_label": ent.label_ if ent is not None else None,
    }


# Remove phrase candidates that are contained by stronger entity or phrase spans.
def _drop_contained_phrases(phrases, debug_mode=False, debug_stage=None):
    if len(phrases) <= 1:
        return phrases

    indexed_phrases = list(enumerate(phrases))
    indexed_phrases.sort(
        key=lambda item: (
            -(item[1][2] - item[1][1]),                     
            item[1][1],                                      
            item[0],                                          
        )
    )

    kept = []                                                    
    for orig_idx, phrase in indexed_phrases:
        _, start_char, end_char = phrase
        is_contained = any(
            kept_start <= start_char and end_char <= kept_end
            for _, (_, kept_start, kept_end) in kept
        )
        if is_contained:
            if debug_mode:
                _debug_skip(
                    "phrase",
                    phrase[0],
                    "its span is fully contained by a longer accepted phrase",
                    stage=debug_stage,
                )
            continue
        kept.append((orig_idx, phrase))

    kept_indices = {orig_idx for orig_idx, _ in kept}
    return [phrase for idx, phrase in enumerate(phrases) if idx in kept_indices]


_QUANTIFIER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth",
}

# Return whether a text span behaves like a numeric or quantifier expression.
def _is_quantifier_like(token):
    token_text = token.text.lower()
    token_lemma = token.lemma_.lower() if token.lemma_ else token_text
    num_types = {value.lower() for value in token.morph.get("NumType")}

    if re.fullmatch(r"\d+(st|nd|rd|th)", token_text):
        return True

    if token.pos_ == "NUM" or token.tag_ == "CD":
        return True

    if "card" in num_types or "ord" in num_types:
        return True

    if token_text in _QUANTIFIER_WORDS or token_lemma in _QUANTIFIER_WORDS:
        return True

    return False


# Detect single-word spans that should be treated as quantifier-like.
def _is_single_word_quantifier_span(span):
    meaningful_tokens = [token for token in span if not token.is_space and not token.is_punct]
    if len(meaningful_tokens) != 1:
        return False

    token = meaningful_tokens[0]
    return _is_quantifier_like(token) or re.fullmatch(r"\d+(\.\d+)?", token.text)


def _is_roman_numeral_text(text):
    return bool(text and ROMAN_NUMERAL_RE.fullmatch(text))


def _is_title_case_token_text(text):
    if not text or not text[0].isupper():
        return False
    return any(char.islower() for char in text[1:])


def _is_title_like_token(token):
    token_text = token.text.strip()
    if not token_text:
        return False
    return (
        token.pos_ == "PROPN"
        or token_text.isupper()
        or _is_roman_numeral_text(token_text)
        or _is_title_case_token_text(token_text)
    )


# Detect multi-token noun chunks that look like titles or proper-name spans.
def _is_title_like_noun_chunk(noun_chunk):
    content_tokens = [
        token for token in noun_chunk
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    if len(content_tokens) < 2:
        return False

    title_like_count = sum(1 for token in content_tokens if _is_title_like_token(token))
    if title_like_count * 2 <= len(content_tokens):
        return False

    # Avoid protecting ordinary sentence-initial phrases where only the first
    # content token has capitalization as its signal.
    first_content = content_tokens[0]
    if title_like_count == 1 and _is_title_case_token_text(first_content.text):
        return False

    return True


# Decide whether a spaCy noun chunk is useful enough to index.
def _is_valid_noun_chunk(noun_chunk, min_tokens=2, debug_mode=False, debug_stage=None):
    if all(token.is_stop or token.is_punct for token in noun_chunk):
        if debug_mode:
            _debug_skip(
                "noun_chunk",
                noun_chunk,
                "all tokens are stop words or punctuation",
                stage=debug_stage,
            )
        return False

    if not any(token.pos_ in ["NOUN", "PROPN", "ADJ"] for token in noun_chunk):
        if debug_mode:
            _debug_skip(
                "noun_chunk",
                noun_chunk,
                "it contains no token with POS in {NOUN, PROPN, ADJ}",
                stage=debug_stage,
            )
        return False

    if noun_chunk.root.pos_ not in ["NOUN", "PROPN"]:
        if debug_mode:
            _debug_skip(
                "noun_chunk",
                noun_chunk,
                f"root token '{noun_chunk.root.text}' has POS={noun_chunk.root.pos_}; expected NOUN or PROPN",
                stage=debug_stage,
            )
        return False

    valid_tokens = [
        token for token in noun_chunk
        if not token.is_stop and not token.is_punct
    ]

    phrase_text = noun_chunk.text.strip()
    if not re.match(r"^(?!.*\.\.)(?![.'])[A-Za-z0-9][A-Za-z0-9.\-\' ]*[A-Za-z0-9]$", phrase_text):
        if debug_mode:
            _debug_skip(
                "noun_chunk",
                noun_chunk,
                f"text '{phrase_text}' contains unsupported punctuation or boundary characters",
                stage=debug_stage,
            )
        return False

    if len(valid_tokens) < min_tokens:
        if debug_mode:
            _debug_skip(
                "noun_chunk",
                noun_chunk,
                f"only {len(valid_tokens)} non-stopword tokens remain; requires at least {min_tokens}",
                stage=debug_stage,
            )
        return False

    if debug_mode:
        _debug_keep("noun_chunk", noun_chunk, stage=debug_stage)
    return True


# Decide whether a token is useful enough to index as an important token.
def _is_valid_token(token, debug_mode=False, debug_stage=None):
    if token.is_space or token.is_punct or token.is_stop:
        if debug_mode:
            failed_checks = []
            if token.is_space:
                failed_checks.append("space")
            if token.is_punct:
                failed_checks.append("punctuation")
            if token.is_stop:
                failed_checks.append("stop word")
            _debug_skip(
                "token",
                token,
                f"filtered out because it is marked as {', '.join(failed_checks)}",
                stage=debug_stage,
            )
        return False

    if _is_quantifier_like(token):
        if debug_mode:
            _debug_skip(
                "token",
                token,
                "token is a quantifier-like cardinal or ordinal term",
                stage=debug_stage,
            )
        return False

    if re.fullmatch(r"\d+(\.\d+)?", token.text):
        if debug_mode:
            _debug_skip(
                "token",
                token,
                "token text is a pure integer or decimal number",
                stage=debug_stage,
            )
        return False

    is_model = any(c.isdigit() for c in token.text)
    # Clausal subjects are kept even with a non-NOUN/PROPN POS: lowercasing in clean_text
    # frequently mis-tags a title-case subject (e.g. "Frozen") as ADJ/VERB, but the parse
    # still labels it nsubj, and the subject is usually the query's key discriminator.
    is_subject = token.dep_ in SUBJECT_DEPS
    if token.pos_ not in ["NOUN", "PROPN"] and not is_model and not is_subject:
        if debug_mode:
            _debug_skip(
                "token",
                token,
                f"POS={token.pos_} is not in {{NOUN, PROPN}}, not a model/code token, and not a clausal subject",
                stage=debug_stage,
            )
        return False

    if all(not c.isalnum() for c in token.text):
        if debug_mode:
            _debug_skip(
                "token",
                token,
                "token contains no alphanumeric characters",
                stage=debug_stage,
            )
        return False

    if len(token.text) < 2:
        if debug_mode:
            _debug_skip(
                "token",
                token,
                "token length is shorter than 2 characters",
                stage=debug_stage,
            )
        return False

    if len(token.text) <= 2 and token.text.islower():
        if debug_mode:
            _debug_skip(
                "token",
                token,
                "very short lowercase token is treated as low-signal noise",
                stage=debug_stage,
            )
        return False

    if debug_mode and debug_stage != "token_filter":
        _debug_keep("token", token, stage=debug_stage)
    return True

# Extract important phrase and token spans from text using spaCy filtering rules.
def extract_important_spans(
    chunk,
    nlp,
    min_tokens=2,
    discard_no_word=False,
    debug_mode=False,
    return_phrase_audit=False,
    doc=None,
):
    if doc is None:
        doc = nlp(chunk)
    protected_entity_records = _collect_protected_entity_records(
        doc,
        discard_no_word=discard_no_word,
    )
    phrase_audit_records = []

    if debug_mode:
        _debug_stage("entity_filter")

    important_phrases, phrase_token_spans, _ = _collect_valid_entities(
        doc,
        discard_no_word=discard_no_word,
        debug_mode=debug_mode,
        debug_stage="entity_filter",
    )
    if return_phrase_audit:
        for phrase, start_char, end_char in important_phrases:
            phrase_audit_records.append(
                _build_phrase_audit_record(
                    doc,
                    phrase,
                    start_char,
                    end_char,
                    "entity",
                    protected_entity_records,
                )
            )

    if debug_mode:
        _debug_stage("noun_chunk_filter")

    for noun_chunk in doc.noun_chunks:
        start_char = noun_chunk.start_char
        end_char = noun_chunk.end_char
        start_token = noun_chunk.start
        end_token = noun_chunk.end

        if phrase_is_contained(phrase_token_spans, (start_token, end_token)):
            if debug_mode:
                _debug_skip(
                    "noun_chunk",
                    noun_chunk,
                    "its token span is already covered by an accepted entity",
                    stage="noun_chunk_filter",
                )
            continue

        residual_spans = _split_span_by_protected_entities(
            noun_chunk,
            protected_entity_records,
        )
        if residual_spans:
            for residual in residual_spans:
                if not _is_valid_noun_chunk(
                    residual,
                    min_tokens=min_tokens,
                    debug_mode=debug_mode,
                    debug_stage="noun_chunk_filter",
                ):
                    continue
                phrase_norm, start_char, end_char, start_token, end_token = _normalize_phrase_from_span(
                    residual,
                    preserve_leading_article=_is_title_like_noun_chunk(residual),
                )
                important_phrases.append((phrase_norm, start_char, end_char))
                phrase_token_spans.append((start_token, end_token))
                if _is_title_like_noun_chunk(residual):
                    _append_protected_span_record(
                        protected_entity_records,
                        residual,
                        "TITLE_LIKE_NOUN_CHUNK",
                    )
                if return_phrase_audit:
                    phrase_audit_records.append(
                        _build_phrase_audit_record(
                            doc,
                            phrase_norm,
                            start_char,
                            end_char,
                            "noun_chunk_residual",
                            protected_entity_records,
                        )
                    )
            continue

        if not _is_valid_noun_chunk(
            noun_chunk,
            min_tokens=min_tokens,
            debug_mode=debug_mode,
            debug_stage="noun_chunk_filter",
        ):
            continue

        phrase_norm, start_char, end_char, start_token, end_token = _normalize_phrase_from_span(
            noun_chunk,
            preserve_leading_article=_is_title_like_noun_chunk(noun_chunk),
        )
        important_phrases.append((phrase_norm, start_char, end_char))
        phrase_token_spans.append((start_token, end_token))
        if _is_title_like_noun_chunk(noun_chunk):
            _append_protected_span_record(
                protected_entity_records,
                noun_chunk,
                "TITLE_LIKE_NOUN_CHUNK",
            )
        if return_phrase_audit:
            phrase_audit_records.append(
                _build_phrase_audit_record(
                    doc,
                    phrase_norm,
                    start_char,
                    end_char,
                    "noun_chunk",
                    protected_entity_records,
                )
            )

    if debug_mode:
        _debug_stage("phrase_dedup")

    important_phrases = _drop_contained_phrases(
        important_phrases,
        debug_mode=debug_mode,
        debug_stage="phrase_dedup" if debug_mode else None,
    )
    if return_phrase_audit:
        kept_phrase_keys = {
            (phrase, start_char, end_char)
            for phrase, start_char, end_char in important_phrases
        }
        phrase_audit_records = [
            record for record in phrase_audit_records
            if (
                record["phrase"],
                record["start_char"],
                record["end_char"],
            ) in kept_phrase_keys
        ]

    important_tokens = []

    if debug_mode:
        _debug_stage("token_filter")

    for token in doc:
        if not _is_valid_token(token, debug_mode=debug_mode, debug_stage="token_filter"):
            continue

        if _token_inside_protected_entity(token, protected_entity_records):
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "token is inside a protected entity",
                    stage="token_filter",
                )
            continue

        if debug_mode:
            _debug_keep(
                "token",
                token,
                "passed token filters",
                stage="token_output",
            )

        start_char = token.idx
        end_char = token.idx + len(token.text)

        important_tokens.append((_normalize_token_text(token), start_char, end_char))

    if debug_mode:
        _debug_stage("final_output")
        _debug_dump_spans("important_phrases", important_phrases)
        _debug_dump_spans("important_tokens", important_tokens)

    if return_phrase_audit:
        return important_phrases, important_tokens, phrase_audit_records
    return important_phrases, important_tokens


# Normalize text for matching by lowercasing words and removing leading articles.
def normalize_text(text: str, *, preserve_leading_article: bool = False) -> str:
    # Normalize one word while preserving meaningful acronyms.
    def process_word(word: str) -> str:
        if re.search(r'[A-Z]{2,}', word):
            return word
        return word.lower()

    words = text.split()
    if not words:
        return ""

    if (
        not preserve_leading_article
        and words[0].lower() in {"the", "a", "an"}
    ):
        words = words[1:]

    return ' '.join(process_word(word) for word in words)


# Normalize a token span into a clean query/index surface form.
def _normalize_token_text(token) -> str:
    token_text = token.text
    token_numbers = set(token.morph.get("Number"))
    if token.pos_ == "NOUN" and "Plur" in token_numbers:
        lemma_text = str(getattr(token, "lemma_", "")).strip()
        if lemma_text:
            token_text = lemma_text
    return normalize_text(token_text)

# Extract named entities and phrase spans for indexing or querying.
def extract_important_phrases(chunk, nlp, min_tokens=2,debug_mode=False):
    doc = nlp(chunk)
    important_phrases, _, phrase_token_spans = _collect_valid_entities(doc, debug_mode=debug_mode)
    protected_entity_records = _collect_protected_entity_records(doc)
    num_ents = len(important_phrases)

    for noun_chunk in doc.noun_chunks:
        start_char = noun_chunk.start_char
        end_char = noun_chunk.end_char

        if phrase_is_contained(phrase_token_spans,(start_char,end_char)):
            if debug_mode:
                _debug_skip(
                    "noun_chunk",
                    noun_chunk,
                    "its character span is already covered by an accepted entity",
                )
            continue

        residual_spans = _split_span_by_protected_entities(
            noun_chunk,
            protected_entity_records,
        )
        if residual_spans:
            for residual in residual_spans:
                if not _is_valid_noun_chunk(
                    residual,
                    min_tokens=min_tokens,
                    debug_mode=debug_mode,
                ):
                    continue
                phrase_norm, start_char, end_char, _, _ = _normalize_phrase_from_span(
                    residual,
                    preserve_leading_article=_is_title_like_noun_chunk(residual),
                )
                important_phrases.append((phrase_norm, start_char, end_char))
            continue

        if not _is_valid_noun_chunk(
            noun_chunk,
            min_tokens=min_tokens,
            debug_mode=debug_mode,
        ):
            continue
        phrase_norm, start_char, end_char, _, _ = _normalize_phrase_from_span(
            noun_chunk,
            preserve_leading_article=_is_title_like_noun_chunk(noun_chunk),
        )
        important_phrases.append((phrase_norm, start_char, end_char))

    return important_phrases, num_ents

# Extract important standalone token spans for indexing or querying.
def extract_important_tokens(chunk,nlp,debug_mode=False):
    doc = nlp(chunk)
    protected_entity_records = _collect_protected_entity_records(doc)
    protected_token_spans = [
        (record["start"], record["end"])
        for record in protected_entity_records
    ]
    for noun_chunk in doc.noun_chunks:
        if phrase_is_contained(protected_token_spans, (noun_chunk.start, noun_chunk.end)):
            continue

        residual_spans = _split_span_by_protected_entities(
            noun_chunk,
            protected_entity_records,
        )
        candidate_spans = residual_spans if residual_spans else [noun_chunk]
        for span in candidate_spans:
            if not _is_valid_noun_chunk(span, min_tokens=2):
                continue
            if _is_title_like_noun_chunk(span):
                _append_protected_span_record(
                    protected_entity_records,
                    span,
                    "TITLE_LIKE_NOUN_CHUNK",
                )

    spans = []
    for token in doc:
        if not _is_valid_token(token, debug_mode=debug_mode):
            continue

        if _token_inside_protected_entity(token, protected_entity_records):
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "token is inside a protected entity",
                )
            continue

        start_char = token.idx
        end_char = token.idx + len(token.text)
        spans.append((_normalize_token_text(token),start_char, end_char))

    return spans

# Find tokenizer token indices that overlap a character span.
def get_token_indices_for_phrase(start_char, end_char, offsets):
    token_indices = []

    for idx, (start, end) in enumerate(offsets):
        if start == end:                             
            continue
        if not (end <= start_char or start >= end_char):
            token_indices.append(idx)

    return token_indices


# Return a safe maximum sequence length for tokenizer calls.
def _get_tokenizer_max_length(tokenizer, fallback=512):
    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and 0 < max_length < 100000:
        return max_length
    return fallback

# Encode one chunk and return token embeddings with offset mappings.
def encode_chunk(text, text_encoder, tokenizer, device):
    max_length = _get_tokenizer_max_length(tokenizer)

    inputs = tokenizer(

        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )

    offsets = inputs["offset_mapping"][0]

    with torch.no_grad():
        outputs = text_encoder(
            **{k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"},
            output_hidden_states=True
        )

        token_embeddings = outputs.hidden_states[-2][0]

    return token_embeddings.detach().cpu(), offsets.cpu()


# Encode a batch of chunks and return embeddings and offsets per chunk.
def encode_chunk_batch(text_list, text_encoder, tokenizer, device):
    max_length = _get_tokenizer_max_length(tokenizer)

    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    offsets = inputs["offset_mapping"]

    with torch.no_grad():
        outputs = text_encoder(
            **{k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"},
            output_hidden_states=True
        )

        token_embeddings = outputs.hidden_states[-2]
    token_embeddings = token_embeddings.detach().cpu()

    return token_embeddings, offsets


# Pool token embeddings for phrase and token spans using tokenizer offsets.
def get_token_embeds(token_embeddings, offsets, phrase_list, token_list):
    phrase_embs = []
    token_embs = []

                           
                                                                                                         
                                                                                  

    for phrase_item in phrase_list:
        phrase, start_char, end_char, *extra_metadata = phrase_item
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        phrase_record = (
            phrase,
            token_embeddings[token_idxs].mean(dim=0),
            start_char,
            end_char,
        )
        if extra_metadata:
            phrase_record = phrase_record + (extra_metadata[0],)
        phrase_embs.append(phrase_record)

    for token_item in token_list:
        token, start_char, end_char, *extra_metadata = token_item
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        token_record = (
            token,
            token_embeddings[token_idxs].mean(dim=0),
            start_char,
            end_char,
        )
        if extra_metadata:
            token_record = token_record + (extra_metadata[0],)
        token_embs.append(token_record)

    return phrase_embs, token_embs

# Encode arbitrary text and return token embeddings with offsets.
def encode_text(query_text, text_encoder, tokenizer, device):
    max_length = _get_tokenizer_max_length(tokenizer)
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = inputs["offset_mapping"][0]
    with torch.no_grad():
        outputs = text_encoder(
            **{k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"},
            output_hidden_states=True
        )
        token_embeddings = outputs.hidden_states[-2][0]
                           
                                                                                                         
                                                                                  
    return token_embeddings, offsets

# Return the pooled embedding for a query span based on character offsets.
def get_embed_by_offest(token_embeddings,offsets, token_metadata):
    phrase, start_char, end_char = token_metadata
    token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
    return token_embeddings[token_idxs].mean(dim=0)

# Encode a query and return one pooled query embedding.
def get_query_embed(query_text, token_list, text_encoder, tokenizer, device):
    token_embeddings, offsets = encode_text(query_text, text_encoder, tokenizer, device)
    token_embs = []
    for phrase, start_char, end_char in token_list:
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        token_embs.append((phrase, token_embeddings[token_idxs].mean(dim=0)))
    return token_embs


# Compute token salience scores by cosine similarity to the sentence embedding.
def token_salience_cosine(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    topk: int = 4,
):
                                                
    mask = attention_mask.unsqueeze(-1).float()                          
    summed = (last_hidden_state * mask).sum(dim=1)                    
    denom = mask.sum(dim=1).clamp_min(1.0)                            
    sent_vec = summed / denom                                         

                  
    sent_vec = F.normalize(sent_vec, p=2, dim=-1)                     
    tok_vec = F.normalize(last_hidden_state, p=2, dim=-1)                

                                                               
    salience = (tok_vec * sent_vec.unsqueeze(1)).sum(dim=-1)          

                                                       
    salience = salience.masked_fill(attention_mask == 0, -1e9)

                           
    topk_idx = torch.topk(salience, k=min(topk, salience.size(1)), dim=-1).indices             

    return topk_idx, salience, sent_vec



# Print the highest-salience tokenizer tokens for inspection.
def print_topk_tokens(tokenizer, input_ids: torch.Tensor, topk_idx: torch.Tensor):
    B, K = topk_idx.shape
    for b in range(B):
        ids = input_ids[b, topk_idx[b]].tolist()                   
        toks = tokenizer.convert_ids_to_tokens(ids)                          
        print(f"[batch {b}] top tokens:", toks)

# Count spaCy tokens in a text string.
def get_num_tokens(text, nlp):
    return len(nlp(text))


# Count whitespace-separated words in text.
def count_words(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    return len(words)

# Return whether every whitespace-separated word is numeric.
def all_words_are_digits(s: str) -> bool:

    words = s.split()
    if not words:            
        return False
    return all(w.isdigit() for w in words)

# Return whether one phrase appears as a whole phrase inside another.
def phrase_is_contained(intervals, new_interval):
    new_start, new_end = new_interval
    return any(s <= new_start and new_end <= e for s, e in intervals)

# Compute the BM25 term-frequency saturation factor for a document length.
def bm25_tf_saturation(tf: int, dl: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    if tf <= 0:
        return 0.0
                                                                 
    if avgdl <= 0:
        avgdl = max(1.0, float(dl))

    denom = tf + k1 * (1.0 - b + b * (float(dl) / avgdl))
    return (tf * (k1 + 1.0)) / denom
