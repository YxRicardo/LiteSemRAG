from __future__ import annotations

import gzip
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def normalize_wikipedia_title(title: str) -> str:
    text = str(title).strip().replace("_", " ")
    return " ".join(text.split()).casefold()


def get_bucket_key(normalized_title: str) -> str:
    chars = [char for char in normalized_title if char.isalnum()]
    if not chars:
        return "__"
    if len(chars) == 1:
        return f"{chars[0]}_"
    return f"{chars[0]}{chars[1]}"


@dataclass(frozen=True)
class RedirectPair:
    redirect: str
    canonical: str


class WikipediaRedirectIndex:
    """
    Dependency-free redirect index stored as sharded gzip pickles.

    Storage layout:
    - metadata.json
    - redirect_buckets/<bucket>.pkl.gz
    - canonical_buckets/<bucket>.pkl.gz
    """

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.bucket_dir = self.index_dir / "redirect_buckets"
        self.canonical_bucket_dir = self.index_dir / "canonical_buckets"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_bucket_dir.mkdir(parents=True, exist_ok=True)
        self._bucket_cache: dict[str, dict[str, tuple[str, str]]] = {}
        self._canonical_bucket_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._metadata_cache: dict[str, str] | None = None

    def close(self) -> None:
        self._bucket_cache.clear()
        self._canonical_bucket_cache.clear()
        self._metadata_cache = None

    def __enter__(self) -> "WikipediaRedirectIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    def clear(self) -> None:
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_bucket_dir.mkdir(parents=True, exist_ok=True)
        self.close()

    def set_metadata(self, key: str, value: str) -> None:
        metadata = self._load_metadata()
        metadata[key] = value
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_metadata(self, key: str) -> str | None:
        return self._load_metadata().get(key)

    def bulk_add_pairs(
        self,
        pairs: Iterable[RedirectPair | Sequence[str]],
        flush_every: int = 100000,
    ) -> int:
        build_dir = self.index_dir / "_build"
        redirect_spool_dir = build_dir / "redirect_spool"
        canonical_spool_dir = build_dir / "canonical_spool"
        build_dir.mkdir(parents=True, exist_ok=True)
        redirect_spool_dir.mkdir(parents=True, exist_ok=True)
        canonical_spool_dir.mkdir(parents=True, exist_ok=True)

        redirect_buffers: dict[str, list[tuple[str, str, str]]] = {}
        canonical_buffers: dict[str, list[tuple[str, str, str]]] = {}
        inserted = 0

        try:
            for pair in pairs:
                redirect, canonical = self._coerce_pair(pair)
                if not redirect or not canonical:
                    continue

                normalized_redirect = normalize_wikipedia_title(redirect)
                normalized_canonical = normalize_wikipedia_title(canonical)
                if normalized_redirect == normalized_canonical:
                    continue

                redirect_bucket = get_bucket_key(normalized_redirect)
                canonical_bucket = get_bucket_key(normalized_canonical)

                redirect_buffers.setdefault(redirect_bucket, []).append(
                    (normalized_redirect, redirect, canonical)
                )
                canonical_buffers.setdefault(canonical_bucket, []).append(
                    (normalized_canonical, canonical, redirect)
                )
                inserted += 1

                if inserted % flush_every == 0:
                    self._flush_spool_buffers(redirect_spool_dir, redirect_buffers)
                    self._flush_spool_buffers(canonical_spool_dir, canonical_buffers)
                    print(f"[index-build] spooled={inserted:,}", flush=True)

            self._flush_spool_buffers(redirect_spool_dir, redirect_buffers)
            self._flush_spool_buffers(canonical_spool_dir, canonical_buffers)

            print("[index-build] finalizing redirect buckets...", flush=True)
            self._finalize_redirect_buckets(redirect_spool_dir)

            print("[index-build] finalizing canonical buckets...", flush=True)
            canonical_pages = self._finalize_canonical_buckets(canonical_spool_dir)

            self.set_metadata("canonical_pages", str(canonical_pages))
            self.set_metadata("redirects", str(inserted))
            print(
                f"[index-build] completed: canonical_pages={canonical_pages:,}, redirects={inserted:,}",
                flush=True,
            )
        finally:
            if build_dir.exists():
                shutil.rmtree(build_dir)

        self._bucket_cache.clear()
        self._canonical_bucket_cache.clear()
        return inserted

    @staticmethod
    def _coerce_pair(pair: RedirectPair | Sequence[str]) -> tuple[str, str]:
        if isinstance(pair, RedirectPair):
            return str(pair.redirect), str(pair.canonical)
        return str(pair[0]), str(pair[1])

    def resolve_redirect(self, title: str) -> str | None:
        normalized = normalize_wikipedia_title(title)
        bucket = self._load_bucket(get_bucket_key(normalized))
        redirect_entry = bucket.get(normalized)
        if redirect_entry is not None:
            return redirect_entry[1]

        canonical_entry = self._load_canonical_bucket(get_bucket_key(normalized)).get(normalized)
        if canonical_entry is None:
            return None
        return str(canonical_entry["title"])

    def get_synonyms(self, title: str, include_canonical: bool = True) -> list[str]:
        normalized = normalize_wikipedia_title(title)
        canonical_bucket = self._load_canonical_bucket(get_bucket_key(normalized))
        canonical_entry = canonical_bucket.get(normalized)

        if canonical_entry is None:
            redirect_bucket = self._load_bucket(get_bucket_key(normalized))
            redirect_entry = redirect_bucket.get(normalized)
            if redirect_entry is None:
                return []
            canonical_normalized = normalize_wikipedia_title(redirect_entry[1])
            canonical_entry = self._load_canonical_bucket(get_bucket_key(canonical_normalized)).get(
                canonical_normalized
            )
            if canonical_entry is None:
                return []

        canonical_title = str(canonical_entry["title"])
        redirects = list(canonical_entry["redirects"])
        if include_canonical:
            return [canonical_title, *redirects]
        return redirects

    def iter_pairs(self) -> Iterator[RedirectPair]:
        for bucket_file in sorted(self.bucket_dir.glob("*.pkl.gz")):
            bucket = self._read_pickle(bucket_file)
            for redirect, canonical in bucket.values():
                yield RedirectPair(redirect=redirect, canonical=canonical)

    def stats(self) -> dict[str, int]:
        metadata = self._load_metadata()
        canonical_pages = metadata.get("canonical_pages")
        redirects = metadata.get("redirects")
        if canonical_pages is not None and redirects is not None:
            return {
                "canonical_pages": int(canonical_pages),
                "redirects": int(redirects),
            }

        canonical_pages_count = 0
        redirect_count = 0
        for bucket_file in self.canonical_bucket_dir.glob("*.pkl.gz"):
            bucket = self._read_pickle(bucket_file)
            canonical_pages_count += len(bucket)
            redirect_count += sum(len(value["redirects"]) for value in bucket.values())
        return {
            "canonical_pages": canonical_pages_count,
            "redirects": redirect_count,
        }

    def _load_metadata(self) -> dict[str, str]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.metadata_path.exists():
            self._metadata_cache = {}
            return self._metadata_cache
        self._metadata_cache = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._metadata_cache

    def _load_bucket(self, bucket_key: str) -> dict[str, tuple[str, str]]:
        if bucket_key in self._bucket_cache:
            return self._bucket_cache[bucket_key]
        path = self.bucket_dir / f"{bucket_key}.pkl.gz"
        if not path.exists():
            self._bucket_cache[bucket_key] = {}
            return self._bucket_cache[bucket_key]
        self._bucket_cache[bucket_key] = self._read_pickle(path)
        return self._bucket_cache[bucket_key]

    def _load_canonical_bucket(self, bucket_key: str) -> dict[str, dict[str, object]]:
        if bucket_key in self._canonical_bucket_cache:
            return self._canonical_bucket_cache[bucket_key]
        path = self.canonical_bucket_dir / f"{bucket_key}.pkl.gz"
        if not path.exists():
            self._canonical_bucket_cache[bucket_key] = {}
            return self._canonical_bucket_cache[bucket_key]
        self._canonical_bucket_cache[bucket_key] = self._read_pickle(path)
        return self._canonical_bucket_cache[bucket_key]

    def _flush_spool_buffers(
        self,
        spool_dir: Path,
        buffers: dict[str, list[tuple[str, str, str]]],
    ) -> None:
        for bucket_key, rows in buffers.items():
            if not rows:
                continue
            spool_path = spool_dir / f"{bucket_key}.jsonl"
            spool_path.parent.mkdir(parents=True, exist_ok=True)
            with spool_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
        buffers.clear()

    def _finalize_redirect_buckets(self, spool_dir: Path) -> None:
        spool_files = sorted(spool_dir.glob("*.jsonl"))
        total_files = len(spool_files)
        for index, spool_file in enumerate(spool_files, start=1):
            bucket: dict[str, tuple[str, str]] = {}
            with spool_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    normalized_redirect, redirect, canonical = json.loads(line)
                    bucket[normalized_redirect] = (redirect, canonical)
            self._write_pickle(self.bucket_dir / f"{spool_file.stem}.pkl.gz", bucket)
            if index % 50 == 0 or index == total_files:
                print(
                    f"[index-build] redirect buckets finalized={index:,}/{total_files:,}",
                    flush=True,
                )

    def _finalize_canonical_buckets(self, spool_dir: Path) -> int:
        spool_files = sorted(spool_dir.glob("*.jsonl"))
        total_files = len(spool_files)
        canonical_pages = 0
        for index, spool_file in enumerate(spool_files, start=1):
            bucket: dict[str, dict[str, object]] = {}
            with spool_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    normalized_canonical, canonical, redirect = json.loads(line)
                    entry = bucket.setdefault(
                        normalized_canonical,
                        {"title": canonical, "redirects": set()},
                    )
                    entry["title"] = canonical
                    entry["redirects"].add(redirect)

            normalized_bucket = {
                key: {
                    "title": value["title"],
                    "redirects": sorted(value["redirects"], key=str.casefold),
                }
                for key, value in bucket.items()
            }
            canonical_pages += len(normalized_bucket)
            self._write_pickle(
                self.canonical_bucket_dir / f"{spool_file.stem}.pkl.gz",
                normalized_bucket,
            )
            if index % 50 == 0 or index == total_files:
                print(
                    f"[index-build] canonical buckets finalized={index:,}/{total_files:,}",
                    flush=True,
                )
        return canonical_pages

    @staticmethod
    def _write_pickle(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _read_pickle(path: Path) -> object:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
