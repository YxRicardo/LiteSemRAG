from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

from text_processing import clean_text, extract_important_spans


DEFAULT_HOTPOT_PATH = Path("jupyter_notebooks/hotpot_dev_distractor_v1.json")
DEFAULT_CACHE_DIR = Path("hotpot_QA_qwen_scan_cache")
DEFAULT_CACHE_TAG = "current_rules_v1"
DEFAULT_SPACY_MODEL = "en_core_web_lg"


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(obj, handle)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_custom_nlp(model_name: str):
    nlp = spacy.load(model_name)
    infixes = [pattern for pattern in nlp.Defaults.infixes if "-" not in pattern]
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,
    )
    return nlp


def build_hotpot_retrieval_dataset(file_path: Path, num_samples: int | None = None):
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if num_samples is not None:
        data = data[:num_samples]

    title_to_doc: dict[str, dict[str, Any]] = {}
    for item in data:
        for title, sentences in item["context"]:
            text = " ".join(sentences).strip()
            if title not in title_to_doc:
                title_to_doc[title] = {
                    "title": title,
                    "text": text,
                    "sentences": sentences,
                }
            elif len(text) > len(title_to_doc[title]["text"]):
                title_to_doc[title]["text"] = text
                title_to_doc[title]["sentences"] = sentences

    documents = []
    title_to_id = {}
    for idx, (title, doc) in enumerate(title_to_doc.items()):
        doc_entry = {
            "doc_id": idx,
            "title": title,
            "text": doc["text"],
            "sentences": doc["sentences"],
        }
        documents.append(doc_entry)
        title_to_id[title] = idx

    samples = []
    for item in data:
        supporting_titles = [title for title, _ in item["supporting_facts"]]
        gold_doc_ids = sorted(
            {title_to_id[title] for title in supporting_titles if title in title_to_id}
        )
        context_doc_ids = [
            title_to_id[title] for title, _ in item["context"] if title in title_to_id
        ]
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "gold_doc_ids": gold_doc_ids,
                "context_doc_ids": context_doc_ids,
                "supporting_facts": item["supporting_facts"],
                "_id": item.get("_id"),
            }
        )

    return documents, samples


def load_or_build_hotpot_documents(
    hotpot_path: Path,
    documents_cache_path: Path,
    samples_cache_path: Path,
    num_samples: int | None,
    force_rebuild_documents: bool,
):
    if (
        not force_rebuild_documents
        and documents_cache_path.exists()
        and samples_cache_path.exists()
    ):
        print(f"Loading cached documents: {documents_cache_path}")
        with documents_cache_path.open("rb") as handle:
            documents = pickle.load(handle)
        with samples_cache_path.open("rb") as handle:
            samples = pickle.load(handle)
        print(f"Loaded {len(documents)} documents and {len(samples)} samples")
        return documents, samples

    print(f"Building HotpotQA document cache from {hotpot_path}")
    documents, samples = build_hotpot_retrieval_dataset(hotpot_path, num_samples=num_samples)
    atomic_pickle_dump(documents, documents_cache_path)
    atomic_pickle_dump(samples, samples_cache_path)
    print(f"Saved documents: {documents_cache_path}")
    print(f"Saved samples: {samples_cache_path}")
    return documents, samples


def collect_span_records(span_list, kind: str, document_idx: int, title: str):
    records = []
    for span in span_list:
        term, start_char, end_char = span[:3]
        record = {
            "kind": kind,
            "term": term,
            "document_idx": int(document_idx),
            "title": title,
            "span": (int(start_char), int(end_char)),
        }
        if len(span) >= 4 and isinstance(span[3], dict):
            record["metadata"] = span[3]
        records.append(record)
    return records


def build_phrase_token_scan_store(
    documents: list[dict[str, Any]],
    nlp,
    *,
    cache_tag: str,
    cache_dir: Path,
    documents_cache_path: Path,
    samples_cache_path: Path,
    output_path: Path,
    spacy_model_name: str,
    discard_no_word: bool,
    nlp_batch_size: int,
):
    cleaned_documents = [clean_text(doc["text"]) for doc in documents]
    store = {
        "documents": documents,
        "cleaned_documents": cleaned_documents,
        "index": defaultdict(list),
        "config": {
            "cache_tag": cache_tag,
            "cache_dir": str(cache_dir),
            "documents_cache_path": str(documents_cache_path),
            "samples_cache_path": str(samples_cache_path),
            "scan_store_path": str(output_path),
            "spacy_model_name": spacy_model_name,
            "discard_no_word": discard_no_word,
            "min_tokens": 2,
            "scan_only": True,
            "span_extractor": "text_processing.extract_important_spans",
            "tokenizer": "LiteSemRAG hyphen-preserving spaCy tokenizer",
        },
    }

    phrase_occurrences = 0
    token_occurrences = 0
    total_documents = len(documents)
    print(f"Scanning {total_documents} documents with current extraction rules")

    docs = nlp.pipe(cleaned_documents, batch_size=nlp_batch_size)
    for document_idx, (cleaned_text, spacy_doc) in enumerate(zip(cleaned_documents, docs)):
        phrases, tokens = extract_important_spans(
            cleaned_text,
            nlp,
            min_tokens=2,
            discard_no_word=discard_no_word,
            doc=spacy_doc,
        )

        title = documents[document_idx]["title"]
        phrase_records = collect_span_records(phrases, "phrase", document_idx, title)
        token_records = collect_span_records(tokens, "token", document_idx, title)

        for record in phrase_records:
            store["index"][record["term"]].append(record)
        for record in token_records:
            store["index"][record["term"]].append(record)

        phrase_occurrences += len(phrase_records)
        token_occurrences += len(token_records)

        processed = document_idx + 1
        if processed % 500 == 0 or processed == total_documents:
            print(f"Processed {processed}/{total_documents} documents")

    store["stats"] = {
        "num_documents": len(documents),
        "num_unique_terms": len(store["index"]),
        "num_phrase_occurrences": phrase_occurrences,
        "num_token_occurrences": token_occurrences,
        "num_total_occurrences": phrase_occurrences + token_occurrences,
    }
    return store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the HotpotQA phrase/token span scan store with current extraction rules."
    )
    parser.add_argument("--hotpot-path", type=Path, default=DEFAULT_HOTPOT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-tag", default=DEFAULT_CACHE_TAG)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--output-sample-tag",
        default=None,
        help=(
            "Override the filename sample tag. Use 'all' to overwrite the original "
            "qwen_scan_v1_all cache even when --num-samples is set."
        ),
    )
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    parser.add_argument("--discard-no-word", action="store_true")
    parser.add_argument("--nlp-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-documents", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_tag = (
        str(args.output_sample_tag)
        if args.output_sample_tag is not None
        else ("all" if args.num_samples is None else str(args.num_samples))
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    documents_cache_path = args.cache_dir / f"hotpot_documents_{args.cache_tag}_{sample_tag}.pkl"
    samples_cache_path = args.cache_dir / f"hotpot_samples_{args.cache_tag}_{sample_tag}.pkl"
    output_path = args.cache_dir / f"hotpot_phrase_token_scan_store_{args.cache_tag}_{sample_tag}.pkl"

    if output_path.exists() and not args.force:
        print(f"Scan store already exists: {output_path}")
        print("Use --force to rebuild it.")
        return

    documents, _ = load_or_build_hotpot_documents(
        args.hotpot_path,
        documents_cache_path,
        samples_cache_path,
        args.num_samples,
        args.force_documents,
    )

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = load_custom_nlp(args.spacy_model)

    store = build_phrase_token_scan_store(
        documents,
        nlp,
        cache_tag=args.cache_tag,
        cache_dir=args.cache_dir,
        documents_cache_path=documents_cache_path,
        samples_cache_path=samples_cache_path,
        output_path=output_path,
        spacy_model_name=args.spacy_model,
        discard_no_word=args.discard_no_word,
        nlp_batch_size=args.nlp_batch_size,
    )
    atomic_pickle_dump(store, output_path)
    print(f"Saved scan store: {output_path}")
    print(store["stats"])


if __name__ == "__main__":
    main()
