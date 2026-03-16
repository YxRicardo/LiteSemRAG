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


def _collect_valid_entities(doc, discard_no_word=False, debug_mode=False):
    entity_spans = []
    token_intervals = []
    char_intervals = []

    for ent in doc.ents:
        if all_words_are_digits(ent.text):
            if debug_mode:
                print(f"ent {ent.text} doesn't pass point1")
            continue

        if discard_no_word and re.search(r"[a-zA-Z]", ent.text) is None:
            if debug_mode:
                print(f"ent {ent.text} doesn't pass point2")
            continue

        if debug_mode:
            print(f"ent {ent.text} pass all test")

        entity_spans.append((normalize_text(ent.text), ent.start_char, ent.end_char))
        token_intervals.append((ent.start, ent.end))
        char_intervals.append((ent.start_char, ent.end_char))

    return entity_spans, token_intervals, char_intervals


def _is_valid_noun_chunk(noun_chunk, min_tokens=2, debug_mode=False):
    if all(token.is_stop or token.is_punct for token in noun_chunk):
        if debug_mode:
            print(f"chunk {noun_chunk} doesn't pass point1")
        return False

    if not any(token.pos_ in ["NOUN", "PROPN", "ADJ"] for token in noun_chunk):
        if debug_mode:
            print(f"chunk {noun_chunk} doesn't pass point2")
        return False

    if noun_chunk.root.pos_ not in ["NOUN", "PROPN"]:
        if debug_mode:
            print(f"chunk {noun_chunk} doesn't pass point3")
        return False

    valid_tokens = [
        token for token in noun_chunk
        if not token.is_stop and not token.is_punct
    ]

    phrase_text = noun_chunk.text.strip()
    if not re.match(r"^(?!.*\.\.)(?![.'])[A-Za-z0-9][A-Za-z0-9.\-\' ]*[A-Za-z0-9]$", phrase_text):
        if debug_mode:
            print(f"chunk {noun_chunk} doesn't pass point4")
        return False

    if len(valid_tokens) < min_tokens:
        if debug_mode:
            print(f"chunk {noun_chunk} doesn't pass point5")
        return False

    if debug_mode:
        print(f"chunk {noun_chunk} pass all test")
    return True


def _is_valid_token(token, debug_mode=False):
    if token.is_space or token.is_punct or token.is_stop:
        if debug_mode:
            print(f"token {token} doesn't pass point1")
        return False

    if re.fullmatch(r"\d+(\.\d+)?", token.text):
        if debug_mode:
            print(f"token {token} doesn't pass point2")
        return False

    is_model = any(c.isdigit() for c in token.text)
    if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
        if debug_mode:
            print(f"token {token} doesn't pass point3")
        return False

    if all(not c.isalnum() for c in token.text):
        if debug_mode:
            print(f"token {token} doesn't pass point4")
        return False

    if len(token.text) < 2:
        if debug_mode:
            print(f"token {token} doesn't pass point5")
        return False

    if len(token.text) <= 2 and token.text.islower():
        if debug_mode:
            print(f"token {token} doesn't pass point6")
        return False

    if debug_mode:
        print(f"token {token} pass all test")
    return True

def extract_important_spans(chunk, nlp, min_tokens=2, remove_duplicate=True,discard_no_word=False,debug_mode=False):
    doc = nlp(chunk)

    important_phrases, phrase_token_spans, _ = _collect_valid_entities(
        doc,
        discard_no_word=discard_no_word,
        debug_mode=debug_mode,
    )

    for noun_chunk in doc.noun_chunks:
        start_char = noun_chunk.start_char
        end_char = noun_chunk.end_char
        start_token = noun_chunk.start
        end_token = noun_chunk.end

        if phrase_is_contained(phrase_token_spans, (start_token, end_token)):
            if debug_mode:
                print(f"chunk {noun_chunk} doesn't pass due to duplication")
            continue

        if not _is_valid_noun_chunk(noun_chunk, min_tokens=min_tokens, debug_mode=debug_mode):
            continue

        important_phrases.append((normalize_text(noun_chunk.text.strip()), start_char, end_char))
        phrase_token_spans.append((start_token, end_token))


    important_tokens = []

    for token in doc:
        if not _is_valid_token(token, debug_mode=debug_mode):
            continue

        # 排除纯数字
        if re.fullmatch(r"\d+(\.\d+)?", token.text):
            if debug_mode:
                print(f"token {token} doesn't pass point2")
            continue

        # 只要包含数字就算型号
        is_model = any(c.isdigit() for c in token.text)

        if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
            if debug_mode:
                print(f"token {token} doesn't pass point3")
            continue

        # 排除纯符号
        if all(not c.isalnum() for c in token.text):
            if debug_mode:
                print(f"token {token} doesn't pass point4")
            continue

        if len(token.text) < 2:
            if debug_mode:
                print(f"token {token} doesn't pass point5")
            continue

        if len(token.text) <= 2 and token.text.islower():
            if debug_mode:
                print(f"token {token} doesn't pass point6")
            continue

        if remove_duplicate:
            inside_phrase = any(
                token.i >= start and token.i < end
                for start, end in phrase_token_spans
            )

            if inside_phrase:
                if debug_mode:
                    print(f"token {token} doesn't pass due to duplication")
                continue

        if debug_mode:
            print(f"token {token} pass all test")

        start_char = token.idx
        end_char = token.idx + len(token.text)

        important_tokens.append((normalize_text(token.text), start_char, end_char))

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
                print(f"chunk {noun_chunk} doesn't pass due to duplication")
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
                print(f"token {token} doesn't pass point1")
            continue

        # 排除纯数字
        if re.fullmatch(r"\d+(\.\d+)?", token.text):
            if debug_mode:
                print(f"token {token} doesn't pass point2")
            continue

        # 只要包含数字就算型号
        is_model = any(c.isdigit() for c in token.text)

        if token.pos_ not in ["NOUN", "PROPN", "ADJ"] and not is_model:
            if debug_mode:
                print(f"token {token} doesn't pass point3")
            continue

        # 排除纯符号
        if all(not c.isalnum() for c in token.text):
            if debug_mode:
                print(f"token {token} doesn't pass point4")
            continue

        if len(token.text) < 2:
            if debug_mode:
                print(f"token {token} doesn't pass point5")
            continue

        if len(token.text) <= 2 and token.text.islower():
            if debug_mode:
                print(f"token {token} doesn't pass point6")
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

def encode_chunk(text, text_encoder, tokenizer, device):

    inputs = tokenizer(

        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
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

    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding=True
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
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
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
    # 去掉首尾空白并按任意空白分词（空格/Tab/换行都会处理）
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
