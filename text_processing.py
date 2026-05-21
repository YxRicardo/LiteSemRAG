import fitz
import torch
import torch.nn.functional as F
import re


# Normalize raw text by standardizing whitespace, quotes, and casing while preserving selected proper-case spans.
def clean_text(text):
                
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    placeholders = {}
    counter = 0

                           
    for match in re.findall(r'\b[A-Z]{2,}\b', text):
        placeholder = f"__PH_{counter}__"
        placeholders[placeholder] = match
        text = text.replace(match, placeholder)
        counter += 1

                                
    pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    for match in re.findall(pattern, text):
        placeholder = f"__PH_{counter}__"
        placeholders[placeholder] = match
        text = text.replace(match, placeholder)
        counter += 1

               
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


# Collect valid named entities and spans that should influence phrase filtering.
def _collect_valid_entities(doc, discard_no_word=False, debug_mode=False, debug_stage=None):
    entity_spans = []
    token_intervals = []
    char_intervals = []

    for ent in doc.ents:
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

        entity_spans.append((normalize_text(ent.text), ent.start_char, ent.end_char))
        token_intervals.append((ent.start, ent.end))
        char_intervals.append((ent.start_char, ent.end_char))

    return entity_spans, token_intervals, char_intervals


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

# Extract important phrase and token spans from text using spaCy filtering rules.
def extract_important_spans(chunk, nlp, min_tokens=2, discard_no_word=False,debug_mode=False):
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

    return important_phrases, important_tokens


# Normalize text for matching by lowercasing words and removing leading articles.
def normalize_text(text: str) -> str:
    # Normalize one word while preserving meaningful acronyms.
    def process_word(word: str) -> str:
        if re.search(r'[A-Z]{2,}', word):
            return word
        return word.lower()

    words = text.split()
    if not words:
        return ""

                          
    if words[0].lower() in {"the", "a", "an"}:
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

        if not _is_valid_noun_chunk(
            noun_chunk,
            min_tokens=min_tokens,
            debug_mode=debug_mode,
        ):
            continue
        important_phrases.append(
            (normalize_text(noun_chunk.text.strip()), noun_chunk.start_char, noun_chunk.end_char)
        )

    return important_phrases, num_ents

# Extract important standalone token spans for indexing or querying.
def extract_important_tokens(chunk,nlp,debug_mode=False):
    doc = nlp(chunk)
    spans = []
    for token in doc:
        if not _is_valid_token(token, debug_mode=debug_mode):
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
