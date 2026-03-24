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
    Dependency-free redirect index stored on disk as sharded gzip pickles.

    Storage layout:
    - metadata.json
    - canonical_map.pkl.gz
    - redirect_buckets/<bucket>.pkl.gz

    Query behavior:
    - redirect -> canonical: loads only the relevant shard
    - canonical -> redirects: loads one canonical map
    """

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.bucket_dir = self.index_dir / "redirect_buckets"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        self._bucket_cache: dict[str, dict[str, tuple[str, str]]] = {}
        self._canonical_map: dict[str, dict[str, object]] | None = None
        self._metadata_cache: dict[str, str] | None = None

    def close(self) -> None:
        self._bucket_cache.clear()
        self._canonical_map = None
        self._metadata_cache = None

    def __enter__(self) -> "WikipediaRedirectIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.json"

    @property
    def canonical_map_path(self) -> Path:
        return self.index_dir / "canonical_map.pkl.gz"

    def clear(self) -> None:
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
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
    ) -> int:
        canonical_map: dict[str, dict[str, object]] = {}
        bucket_maps: dict[str, dict[str, tuple[str, str]]] = {}
        inserted = 0

        for pair in pairs:
            redirect, canonical = self._coerce_pair(pair)
            if not redirect or not canonical:
                continue

            normalized_redirect = normalize_wikipedia_title(redirect)
            normalized_canonical = normalize_wikipedia_title(canonical)
            if normalized_redirect == normalized_canonical:
                continue

            bucket = bucket_maps.setdefault(get_bucket_key(normalized_redirect), {})
            bucket[normalized_redirect] = (redirect, canonical)

            canonical_entry = canonical_map.setdefault(
                normalized_canonical,
                {"title": canonical, "redirects": []},
            )
            canonical_entry["title"] = canonical
            canonical_entry["redirects"].append(redirect)
            inserted += 1

        for bucket_key, bucket_data in bucket_maps.items():
            self._write_pickle(self.bucket_dir / f"{bucket_key}.pkl.gz", bucket_data)

        normalized_canonical_map = {
            key: {
                "title": value["title"],
                "redirects": sorted(set(value["redirects"]), key=str.casefold),
            }
            for key, value in canonical_map.items()
        }
        self._write_pickle(self.canonical_map_path, normalized_canonical_map)
        self._canonical_map = normalized_canonical_map
        self._bucket_cache = bucket_maps
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

        canonical_entry = self._load_canonical_map().get(normalized)
        if canonical_entry is None:
            return None
        return str(canonical_entry["title"])

    def get_synonyms(self, title: str, include_canonical: bool = True) -> list[str]:
        normalized = normalize_wikipedia_title(title)
        canonical_map = self._load_canonical_map()
        canonical_entry = canonical_map.get(normalized)

        if canonical_entry is None:
            bucket = self._load_bucket(get_bucket_key(normalized))
            redirect_entry = bucket.get(normalized)
            if redirect_entry is None:
                return []
            canonical_entry = canonical_map.get(normalize_wikipedia_title(redirect_entry[1]))
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
        canonical_map = self._load_canonical_map()
        redirect_count = sum(len(value["redirects"]) for value in canonical_map.values())
        return {
            "canonical_pages": len(canonical_map),
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

    def _load_canonical_map(self) -> dict[str, dict[str, object]]:
        if self._canonical_map is not None:
            return self._canonical_map
        if not self.canonical_map_path.exists():
            self._canonical_map = {}
            return self._canonical_map
        self._canonical_map = self._read_pickle(self.canonical_map_path)
        return self._canonical_map

    def _load_bucket(self, bucket_key: str) -> dict[str, tuple[str, str]]:
        if bucket_key in self._bucket_cache:
            return self._bucket_cache[bucket_key]
        path = self.bucket_dir / f"{bucket_key}.pkl.gz"
        if not path.exists():
            self._bucket_cache[bucket_key] = {}
            return self._bucket_cache[bucket_key]
        self._bucket_cache[bucket_key] = self._read_pickle(path)
        return self._bucket_cache[bucket_key]

    @staticmethod
    def _write_pickle(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _read_pickle(path: Path) -> object:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
