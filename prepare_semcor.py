from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


RESOURCE_NAMES = ("semcor", "wordnet", "omw-1.4")
WORDNET_POS_MAP = {
    "1": "n",
    "2": "v",
    "3": "a",
    "4": "r",
    "5": "s",
}
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "''", "'s", "n't"}
NO_SPACE_AFTER = {"(", "[", "{", "``", "$"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SemCor via NLTK and export it into convenient JSONL files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/semcor"),
        help="Base directory used for raw NLTK data and processed exports.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite processed output files even if they already exist.",
    )
    return parser.parse_args()


def ensure_parent_dirs(base_dir: Path) -> tuple[Path, Path]:
    raw_dir = base_dir / "raw" / "nltk_data"
    processed_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def ensure_nltk_resources(download_dir: Path) -> tuple[Any, Any]:
    import nltk

    download_dir = download_dir.resolve()
    download_dir_str = str(download_dir)
    if download_dir_str not in nltk.data.path:
        nltk.data.path.insert(0, download_dir_str)

    for resource_name in RESOURCE_NAMES:
        try:
            nltk.data.find(f"corpora/{resource_name}")
        except LookupError:
            ok = nltk.download(resource_name, download_dir=download_dir_str, quiet=False)
            if not ok:
                raise RuntimeError(f"Failed to download NLTK resource: {resource_name}")

    from nltk.corpus import semcor, wordnet as wn

    semcor.ensure_loaded()
    wn.ensure_loaded()
    return semcor, wn


def iter_xml_words(element: ElementTree.Element) -> Iterable[ElementTree.Element]:
    for child in element:
        if child.tag in {"wf", "punc"}:
            yield child
        else:
            yield from iter_xml_words(child)


def wordnet_pos_from_lexsn(lexsn: str | None) -> str | None:
    if not lexsn:
        return None
    return WORDNET_POS_MAP.get(lexsn.split(":", 1)[0])


def split_surface_tokens(text: str, is_punctuation: bool) -> list[str]:
    if not text:
        return []
    if is_punctuation:
        return [text]
    return [part for part in text.split("_") if part]


def detokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""

    pieces: list[str] = []
    for token in tokens:
        if not pieces:
            pieces.append(token)
            continue

        if token in NO_SPACE_BEFORE or pieces[-1] in NO_SPACE_AFTER:
            pieces[-1] += token
        else:
            pieces.append(token)

    return " ".join(pieces)


def build_annotation_record(
    xmlword: ElementTree.Element,
    annotation_index: int,
    token_start: int,
    wn: Any,
) -> tuple[dict[str, Any], list[str]]:
    surface = xmlword.text or ""
    is_punctuation = xmlword.tag == "punc"
    tokens = split_surface_tokens(surface, is_punctuation=is_punctuation)
    token_end = token_start + len(tokens)

    lemma = xmlword.get("lemma")
    lexsn = xmlword.get("lexsn")
    wnsn = xmlword.get("wnsn")
    pos = xmlword.get("pos")
    rdf = xmlword.get("rdf")
    pn = xmlword.get("pn")
    wordnet_pos = wordnet_pos_from_lexsn(lexsn)
    sense_key = f"{lemma}%{lexsn}" if lemma and lexsn else None
    is_oov_entity = pn is not None

    synset_name = None
    synset_definition = None
    lexname = None
    sense_lookup_error = None

    if sense_key is not None:
        try:
            lemma_obj = wn.lemma_from_key(sense_key)
            synset = lemma_obj.synset()
            synset_name = synset.name()
            synset_definition = synset.definition()
            lexname = synset.lexname()
        except Exception as exc:
            sense_lookup_error = str(exc)

    record = {
        "annotation_index": annotation_index,
        "text": surface,
        "tokens": tokens,
        "token_start": token_start,
        "token_end": token_end,
        "is_punctuation": is_punctuation,
        "is_multiword": len(tokens) > 1,
        "lemma": lemma,
        "rdf": rdf,
        "pos": pos,
        "wnsn": wnsn,
        "lexsn": lexsn,
        "wordnet_pos": wordnet_pos,
        "sense_key": sense_key,
        "synset_name": synset_name,
        "synset_definition": synset_definition,
        "lexname": lexname,
        "pn": pn,
        "is_oov_entity": is_oov_entity,
        "has_semantic_annotation": bool(wnsn or is_oov_entity),
        "sense_lookup_error": sense_lookup_error,
    }
    return record, tokens


def is_noun_annotation(record: dict[str, Any]) -> bool:
    if not record["has_semantic_annotation"]:
        return False
    if record["wordnet_pos"] == "n":
        return True
    return bool(record["is_oov_entity"] and (record["pos"] or "").startswith("NN"))


def process_semcor_file(
    file_pointer: Any,
    fileid: str,
    wn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    with file_pointer.open() as handle:
        root = ElementTree.parse(handle).getroot()

    sentences: list[dict[str, Any]] = []
    flat_annotations: list[dict[str, Any]] = []
    stats = Counter()

    for sent_index, xml_sentence in enumerate(root.findall(".//s")):
        sentence_num = xml_sentence.attrib.get("snum", str(sent_index))
        sentence_id = f"{fileid}:{sentence_num}"
        sentence_tokens: list[str] = []
        annotations: list[dict[str, Any]] = []
        token_cursor = 0

        for annotation_index, xmlword in enumerate(iter_xml_words(xml_sentence)):
            record, new_tokens = build_annotation_record(
                xmlword=xmlword,
                annotation_index=annotation_index,
                token_start=token_cursor,
                wn=wn,
            )
            token_cursor += len(new_tokens)
            sentence_tokens.extend(new_tokens)
            annotations.append(record)

            flat_record = {
                "sentence_id": sentence_id,
                "fileid": fileid,
                "sentence_num": sentence_num,
                **record,
            }
            flat_annotations.append(flat_record)

            stats["annotation_count"] += 1
            if record["has_semantic_annotation"]:
                stats["semantic_annotation_count"] += 1
            if record["is_oov_entity"]:
                stats["oov_entity_count"] += 1
            if record["is_multiword"]:
                stats["multiword_annotation_count"] += 1
            if is_noun_annotation(record):
                stats["noun_annotation_count"] += 1

        sentence_text = detokenize(sentence_tokens)
        noun_annotations = [
            annotation for annotation in annotations if is_noun_annotation(annotation)
        ]
        sentence_record = {
            "sentence_id": sentence_id,
            "fileid": fileid,
            "sentence_num": sentence_num,
            "token_count": len(sentence_tokens),
            "annotation_count": len(annotations),
            "semantic_annotation_count": sum(
                1 for annotation in annotations if annotation["has_semantic_annotation"]
            ),
            "noun_annotation_count": len(noun_annotations),
            "text": sentence_text,
            "tokens": sentence_tokens,
            "annotations": annotations,
            "noun_annotations": noun_annotations,
        }
        sentences.append(sentence_record)

        stats["sentence_count"] += 1
        stats["token_count"] += len(sentence_tokens)

    return sentences, flat_annotations, stats


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_semcor_stats(base_dir: str | Path = "data/semcor") -> dict[str, Any]:
    stats_path = Path(base_dir) / "processed" / "semcor_stats.json"
    with stats_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        lines = handle if limit is None else islice(handle, limit)
        for line in lines:
            if line.strip():
                yield json.loads(line)


def iter_sentence_records(
    base_dir: str | Path = "data/semcor",
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    sentence_path = Path(base_dir) / "processed" / "semcor_sentences.jsonl"
    return iter_jsonl(sentence_path, limit=limit)


def iter_annotation_records(
    base_dir: str | Path = "data/semcor",
    limit: int | None = None,
    semantic_only: bool = False,
) -> Iterable[dict[str, Any]]:
    annotation_path = Path(base_dir) / "processed" / "semcor_annotations.jsonl"
    records = iter_jsonl(annotation_path, limit=limit)
    if not semantic_only:
        return records
    return (record for record in records if record["has_semantic_annotation"])


def iter_noun_sentence_records(
    base_dir: str | Path = "data/semcor",
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    noun_sentence_path = Path(base_dir) / "processed" / "semcor_noun_sentences.jsonl"
    return iter_jsonl(noun_sentence_path, limit=limit)


def iter_noun_annotation_records(
    base_dir: str | Path = "data/semcor",
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    noun_annotation_path = Path(base_dir) / "processed" / "semcor_noun_annotations.jsonl"
    return iter_jsonl(noun_annotation_path, limit=limit)


def get_sentence_record_by_index(
    index: int,
    base_dir: str | Path = "data/semcor",
) -> dict[str, Any]:
    if index < 0:
        raise IndexError("index must be non-negative")

    try:
        return next(islice(iter_sentence_records(base_dir=base_dir), index, index + 1))
    except StopIteration as exc:
        raise IndexError(f"sentence index out of range: {index}") from exc


def collect_semcor_exports(base_dir: Path, force: bool = False) -> dict[str, Any]:
    nltk_dir, processed_dir = ensure_parent_dirs(base_dir)
    sentence_path = processed_dir / "semcor_sentences.jsonl"
    annotation_path = processed_dir / "semcor_annotations.jsonl"
    noun_sentence_path = processed_dir / "semcor_noun_sentences.jsonl"
    noun_annotation_path = processed_dir / "semcor_noun_annotations.jsonl"
    stats_path = processed_dir / "semcor_stats.json"

    if (
        not force
        and sentence_path.exists()
        and annotation_path.exists()
        and noun_sentence_path.exists()
        and noun_annotation_path.exists()
        and stats_path.exists()
    ):
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    semcor, wn = ensure_nltk_resources(nltk_dir)

    all_sentences: list[dict[str, Any]] = []
    all_annotations: list[dict[str, Any]] = []
    all_noun_sentences: list[dict[str, Any]] = []
    all_noun_annotations: list[dict[str, Any]] = []
    total_stats = Counter()

    for fileid, abspath in zip(semcor.fileids(), semcor.abspaths()):
        file_sentences, file_annotations, file_stats = process_semcor_file(
            file_pointer=abspath,
            fileid=fileid,
            wn=wn,
        )
        all_sentences.extend(file_sentences)
        all_annotations.extend(file_annotations)
        all_noun_sentences.extend(
            {
                "sentence_id": sentence["sentence_id"],
                "fileid": sentence["fileid"],
                "sentence_num": sentence["sentence_num"],
                "token_count": sentence["token_count"],
                "noun_annotation_count": sentence["noun_annotation_count"],
                "text": sentence["text"],
                "tokens": sentence["tokens"],
                "noun_annotations": sentence["noun_annotations"],
            }
            for sentence in file_sentences
            if sentence["noun_annotation_count"] > 0
        )
        all_noun_annotations.extend(
            annotation for annotation in file_annotations if is_noun_annotation(annotation)
        )
        total_stats.update(file_stats)
        total_stats["file_count"] += 1

    write_jsonl(all_sentences, sentence_path)
    write_jsonl(all_annotations, annotation_path)
    write_jsonl(all_noun_sentences, noun_sentence_path)
    write_jsonl(all_noun_annotations, noun_annotation_path)

    stats_payload = {
        "dataset_name": "SemCor",
        "base_dir": str(base_dir.resolve()),
        "nltk_data_dir": str(nltk_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "resources": list(RESOURCE_NAMES),
        "file_count": total_stats["file_count"],
        "sentence_count": total_stats["sentence_count"],
        "token_count": total_stats["token_count"],
        "annotation_count": total_stats["annotation_count"],
        "semantic_annotation_count": total_stats["semantic_annotation_count"],
        "noun_annotation_count": total_stats["noun_annotation_count"],
        "oov_entity_count": total_stats["oov_entity_count"],
        "multiword_annotation_count": total_stats["multiword_annotation_count"],
        "sentence_jsonl": str(sentence_path.resolve()),
        "annotation_jsonl": str(annotation_path.resolve()),
        "noun_sentence_jsonl": str(noun_sentence_path.resolve()),
        "noun_annotation_jsonl": str(noun_annotation_path.resolve()),
    }

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, ensure_ascii=False, indent=2)

    return stats_payload


def main() -> None:
    args = parse_args()
    stats = collect_semcor_exports(base_dir=args.base_dir, force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
