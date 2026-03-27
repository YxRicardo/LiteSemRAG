import fitz
import torch
import torch.nn.functional as F
import re


def clean_text(text):
    """
    功能说明：
    1. 去除因换行产生的断词（如 "-\n"），并将多个换行和多余空白字符替换为单个空格。
    2. 识别连续两个及以上的大写缩写词（如 NASA）并保护。
    3. 识别连续两个及以上首字母大写的单词（如 John Smith）并保护。
    4. 删除所有英文引号（单引号 ' 和双引号 "）。
    5. 将全文转换为小写。
    6. 将之前替换的内容恢复。
    7. 返回去除首尾空格后的清洗文本。
    """
    # 1. 处理换行和空白
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    placeholders = {}
    counter = 0

    # 2. 保护全大写缩写（NASA, USA）
    for match in re.findall(r'\b[A-Z]{2,}\b', text):
        placeholder = f"__PH_{counter}__"
        placeholders[placeholder] = match
        text = text.replace(match, placeholder)
        counter += 1

    # 3. 保护连续首字母大写单词（John Smith）
    pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    for match in re.findall(pattern, text):
        placeholder = f"__PH_{counter}__"
        placeholders[placeholder] = match
        text = text.replace(match, placeholder)
        counter += 1

    # 4. 删除英文引号
    text = re.sub(r"[\"']", "", text)

    # 5. 转小写
    text = text.lower()

    # 6. 恢复占位符
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder.lower(), original)

    return text.strip()


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

def split_doc(doc_name, nlp, max_tokens=300):
    with fitz.open(doc_name) as pdf:
        text = "".join(page.get_text() for page in pdf)

    text = clean_text(text)
    chunks = chunk_by_sentences(text, nlp, max_tokens=max_tokens)

    return chunks


def _debug_item_text(item):
    return getattr(item, "text", str(item))


def _debug_prefix(item_type, stage=None):
    if stage:
        return f"[{stage}][{item_type}]"
    return f"[{item_type}]"


def _debug_stage(stage):
    print(f"\n=== {stage} ===")


def _debug_skip(item_type, item, reason, stage=None):
    print(f"{_debug_prefix(item_type, stage)} Skipping '{_debug_item_text(item)}': {reason}")


def _debug_keep(item_type, item, reason="passed all filters", stage=None):
    return


def _debug_dump_spans(name, spans):
    print(f"[final_output] {name} ({len(spans)} items):")
    if not spans:
        print("  (none)")
        return

    for index, (text, start_char, end_char) in enumerate(spans, start=1):
        print(f"  {index}. '{text}' [{start_char}, {end_char})")


def _collect_valid_entities(doc, discard_no_word=False, debug_mode=False, debug_stage=None):
    entity_spans = []
    token_intervals = []
    char_intervals = []

    for ent in doc.ents:
        if all_words_are_digits(ent.text):
            if debug_mode:
                _debug_skip("entity", ent, "every word in the entity is numeric", stage=debug_stage)
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

        entity_spans.append((normalize_text(ent.text), ent.start_char, ent.end_char))
        token_intervals.append((ent.start, ent.end))
        char_intervals.append((ent.start_char, ent.end_char))

    return entity_spans, token_intervals, char_intervals


def _drop_contained_phrases(phrases, debug_mode=False, debug_stage=None):
    """
    Remove phrases whose character span is fully contained by another phrase span.
    Keep outer/longer phrases, and keep original output order for survivors.
    """
    if len(phrases) <= 1:
        return phrases

    indexed_phrases = list(enumerate(phrases))
    indexed_phrases.sort(
        key=lambda item: (
            -(item[1][2] - item[1][1]),  # longer span first
            item[1][1],                  # then earlier start
            item[0],                     # then original order
        )
    )

    kept = []  # tuples: (orig_idx, (text, start_char, end_char))
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
    if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
        if debug_mode:
            _debug_skip(
                "token",
                token,
                f"POS={token.pos_} is not in {{NOUN, PROPN, ADJ}} and the token does not look like a model/code token",
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

def extract_important_spans(chunk, nlp, min_tokens=2, remove_duplicate=True,discard_no_word=False,debug_mode=False):
    doc = nlp(chunk)

    if debug_mode:
        _debug_stage("entity_filter")

    important_phrases, phrase_token_spans, _ = _collect_valid_entities(
        doc,
        discard_no_word=discard_no_word,
        debug_mode=debug_mode,
        debug_stage="entity_filter",
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

        if not _is_valid_noun_chunk(
            noun_chunk,
            min_tokens=min_tokens,
            debug_mode=debug_mode,
            debug_stage="noun_chunk_filter",
        ):
            continue

        important_phrases.append((normalize_text(noun_chunk.text.strip()), start_char, end_char))
        phrase_token_spans.append((start_token, end_token))

    if debug_mode:
        _debug_stage("phrase_dedup")

    important_phrases = _drop_contained_phrases(
        important_phrases,
        debug_mode=debug_mode,
        debug_stage="phrase_dedup" if debug_mode else None,
    )

    important_tokens = []

    if debug_mode:
        _debug_stage("token_filter")

    for token in doc:
        if not _is_valid_token(token, debug_mode=debug_mode, debug_stage="token_filter"):
            continue

        # 排除纯数字
        if re.fullmatch(r"\d+(\.\d+)?", token.text):
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "token text is a pure integer or decimal number",
                    stage="token_filter",
                )
            continue

        # 只要包含数字就算型号
        is_model = any(c.isdigit() for c in token.text)

        if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    f"POS={token.pos_} is not in {{NOUN, PROPN, ADJ}} and the token does not look like a model/code token",
                    stage="token_filter",
                )
            continue

        # 排除纯符号
        if all(not c.isalnum() for c in token.text):
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "token contains no alphanumeric characters",
                    stage="token_filter",
                )
            continue

        if len(token.text) < 2:
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "token length is shorter than 2 characters",
                    stage="token_filter",
                )
            continue

        if len(token.text) <= 2 and token.text.islower():
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "very short lowercase token is treated as low-signal noise",
                    stage="token_filter",
                )
            continue

        if remove_duplicate:
            inside_phrase = any(
                token.i >= start and token.i < end
                for start, end in phrase_token_spans
            )

            if inside_phrase:
                if debug_mode:
                    _debug_skip(
                        "token",
                        token,
                        "token is inside an accepted entity or noun chunk span",
                        stage="token_dedup",
                    )
                continue

        if debug_mode:
            _debug_keep(
                "token",
                token,
                "passed token filters and is not covered by a larger accepted span",
                stage="token_output",
            )

        start_char = token.idx
        end_char = token.idx + len(token.text)

        important_tokens.append((normalize_text(token.text), start_char, end_char))

    if debug_mode:
        _debug_stage("final_output")
        _debug_dump_spans("important_phrases", important_phrases)
        _debug_dump_spans("important_tokens", important_tokens)

    return important_phrases, important_tokens


def normalize_text(text: str) -> str:
    def process_word(word: str) -> str:
        if re.search(r'[A-Z]{2,}', word):
            return word
        return word.lower()

    words = text.split()
    if not words:
        return ""

    # 如果第一个单词是冠词（忽略大小写），删除
    if words[0].lower() in {"the", "a", "an"}:
        words = words[1:]

    return ' '.join(process_word(word) for word in words)

def extract_important_phrases(chunk, nlp, min_tokens=2,debug_mode=False):
    doc = nlp(chunk)
    important_phrases, _, phrase_token_spans = _collect_valid_entities(doc, debug_mode=debug_mode)
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

        if not _is_valid_noun_chunk(noun_chunk, min_tokens=min_tokens, debug_mode=debug_mode):
            continue
        important_phrases.append(
            (normalize_text(noun_chunk.text.strip()), noun_chunk.start_char, noun_chunk.end_char)
        )

    return important_phrases, num_ents

def extract_important_tokens(chunk,nlp,debug_mode=False):
    doc = nlp(chunk)
    spans = []
    for token in doc:

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
                )
            continue

        # 排除纯数字
        if re.fullmatch(r"\d+(\.\d+)?", token.text):
            if debug_mode:
                _debug_skip("token", token, "token text is a pure integer or decimal number")
            continue

        # 只要包含数字就算型号
        is_model = any(c.isdigit() for c in token.text)

        if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    f"POS={token.pos_} is not in {{NOUN, PROPN, ADJ}} and the token does not look like a model/code token",
                )
            continue

        # 排除纯符号
        if all(not c.isalnum() for c in token.text):
            if debug_mode:
                _debug_skip("token", token, "token contains no alphanumeric characters")
            continue

        if len(token.text) < 2:
            if debug_mode:
                _debug_skip("token", token, "token length is shorter than 2 characters")
            continue

        if len(token.text) <= 2 and token.text.islower():
            if debug_mode:
                _debug_skip(
                    "token",
                    token,
                    "very short lowercase token is treated as low-signal noise",
                )
            continue


        start_char = token.idx
        end_char = token.idx + len(token.text)
        spans.append((normalize_text(token.text),start_char, end_char))

    return spans

def get_token_indices_for_phrase(start_char, end_char, offsets):
    token_indices = []

    for idx, (start, end) in enumerate(offsets):
        if start == end:  # special tokens like [CLS]
            continue
        if not (end <= start_char or start >= end_char):
            token_indices.append(idx)

    return token_indices


def _get_tokenizer_max_length(tokenizer, fallback=512):
    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and 0 < max_length < 100000:
        return max_length
    return fallback

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


def get_token_embeds(token_embeddings, offsets, phrase_list, token_list):
    phrase_embs = []
    token_embs = []

    # with torch.no_grad():
    #     outputs = text_encoder(**{k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"})
    #     token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    for phrase, start_char, end_char in phrase_list:
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        phrase_embs.append((phrase, token_embeddings[token_idxs].mean(dim=0)))

    for token, start_char, end_char in token_list:
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        token_embs.append((token,token_embeddings[token_idxs].mean(dim=0)))

    return phrase_embs, token_embs

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
    # with torch.no_grad():
    #     outputs = text_encoder(**{k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"})
    #     token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
    return token_embeddings, offsets

def get_embed_by_offest(token_embeddings,offsets, token_metadata):
    phrase, start_char, end_char = token_metadata
    token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
    return token_embeddings[token_idxs].mean(dim=0)

def get_query_embed(query_text, token_list, text_encoder, tokenizer, device):
    token_embeddings, offsets = encode_text(query_text, text_encoder, tokenizer, device)
    token_embs = []
    for phrase, start_char, end_char in token_list:
        token_idxs = get_token_indices_for_phrase(start_char, end_char, offsets)
        if not token_idxs:
            continue
        token_embs.append((phrase, token_embeddings[token_idxs].mean(dim=0)))
    return token_embs


def token_salience_cosine(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    topk: int = 4,
):
    """
    last_hidden_state: (B, L, H) from DeBERTa (or any transformer)
    attention_mask:    (B, L) with 1 for real tokens, 0 for padding
    Returns:
      topk_idx: (B, topk) token indices with highest salience
      salience: (B, L) cosine(h_i, sent_vec) for each token
      sent_vec:(B, H) sentence vector (masked mean pooled, L2-normalized)
    """
    # masked mean pooling to get sentence vector
    mask = attention_mask.unsqueeze(-1).float()               # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)            # (B, H)
    denom = mask.sum(dim=1).clamp_min(1.0)                    # (B, 1)
    sent_vec = summed / denom                                 # (B, H)

    # L2 normalize
    sent_vec = F.normalize(sent_vec, p=2, dim=-1)             # (B, H)
    tok_vec = F.normalize(last_hidden_state, p=2, dim=-1)     # (B, L, H)

    # cosine similarity for each token: dot(normalized vectors)
    salience = (tok_vec * sent_vec.unsqueeze(1)).sum(dim=-1)  # (B, L)

    # mask out padding tokens so they won't be selected
    salience = salience.masked_fill(attention_mask == 0, -1e9)

    # top-k token positions
    topk_idx = torch.topk(salience, k=min(topk, salience.size(1)), dim=-1).indices  # (B, topk)

    return topk_idx, salience, sent_vec



def print_topk_tokens(tokenizer, input_ids: torch.Tensor, topk_idx: torch.Tensor):
    """
    tokenizer: HuggingFace tokenizer
    input_ids: (B, L)
    topk_idx:  (B, K) token positions
    """
    B, K = topk_idx.shape
    for b in range(B):
        ids = input_ids[b, topk_idx[b]].tolist()            # K ids
        toks = tokenizer.convert_ids_to_tokens(ids)         # K token strings
        print(f"[batch {b}] top tokens:", toks)

def get_num_tokens(text, nlp):
    return len(nlp(text))


def count_words(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    return len(words)

def all_words_are_digits(s: str) -> bool:

    words = s.split()
    if not words:  # 空字符串或全空白
        return False
    return all(w.isdigit() for w in words)

def phrase_is_contained(intervals, new_interval):
    new_start, new_end = new_interval
    return any(s <= new_start and new_end <= e for s, e in intervals)

def bm25_tf_saturation(tf: int, dl: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """
    BM25-style term-frequency saturation factor (without IDF).
    Returns the multiplicative factor applied to a term weight for a (term, chunk) pair.

    tf   : term frequency in the chunk
    dl   : chunk length (e.g., number of tokens)
    avgdl: average chunk length
    """
    if tf <= 0:
        return 0.0
    # Avoid division by zero if avgdl is 0 in degenerate datasets
    if avgdl <= 0:
        avgdl = max(1.0, float(dl))

    denom = tf + k1 * (1.0 - b + b * (float(dl) / avgdl))
    return (tf * (k1 + 1.0)) / denom
