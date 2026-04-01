from __future__ import annotations

import gzip
import json
import pickle
import re
import shutil
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence


_PAREN_PATTERN = re.compile(r"\([^()]*\)|\uFF08[^\uFF08\uFF09]*\uFF09")
_SPACE_PATTERN = re.compile(r"\s+")
_LOW_INFORMATION_TOKENS = frozenset({"a", "an", "and", "by", "for", "in", "of", "on", "the", "to"})


def _strip_parenthetical_content(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _PAREN_PATTERN.sub(" ", current)
    return current


def _remove_special_symbols(text: str) -> str:
    cleaned_chars: list[str] = []
    for char in text:
        if char.isascii() and (char.isalnum() or char in {" ", "'", "-", "&", "/", "."}):
            cleaned_chars.append(char)
            continue

        category = unicodedata.category(char)
        if category.startswith("Z"):
            cleaned_chars.append(" ")
            continue

        cleaned_chars.append(" ")
    return "".join(cleaned_chars)


def normalize_wikipedia_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", str(title).strip().replace("_", " "))
    text = text.casefold()
    text = _strip_parenthetical_content(text)
    text = _remove_special_symbols(text)
    text = text.replace(".", " ")
    text = _SPACE_PATTERN.sub(" ", text).strip()
    return text


def should_keep_index_title(normalized_title: str) -> bool:
    if not normalized_title:
        return False
    if not re.search(r"[a-z0-9]", normalized_title):
        return False
    if not re.search(r"[a-z]", normalized_title):
        return False
    return True


def contains_non_ascii(text: str) -> bool:
    return not text.isascii()


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


@dataclass(frozen=True)
class WikipediaGraphNode:
    node_id: str
    title: str
    normalized_title: str
    redirect_target_id: str | None
    redirect_target_title: str | None
    incoming_redirects: tuple[str, ...]

    @property
    def is_redirect(self) -> bool:
        return self.redirect_target_id is not None

    @property
    def incoming_count(self) -> int:
        return len(self.incoming_redirects)


def print_inline_progress(message: str) -> None:
    print(f"\r{message}", end="", flush=True)


def finish_inline_progress(message: str) -> None:
    print(f"\r{message}", flush=True)


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
                if contains_non_ascii(redirect):
                    continue
                if contains_non_ascii(canonical):
                    continue

                normalized_redirect = normalize_wikipedia_title(redirect)
                normalized_canonical = normalize_wikipedia_title(canonical)
                if not should_keep_index_title(normalized_redirect):
                    continue
                if not should_keep_index_title(normalized_canonical):
                    continue
                if normalized_redirect == normalized_canonical:
                    continue

                redirect_bucket = get_bucket_key(normalized_redirect)
                canonical_bucket = get_bucket_key(normalized_canonical)

                redirect_buffers.setdefault(redirect_bucket, []).append(
                    (normalized_redirect, normalized_redirect, normalized_canonical)
                )
                canonical_buffers.setdefault(canonical_bucket, []).append(
                    (normalized_canonical, normalized_canonical, normalized_redirect)
                )
                inserted += 1

                if inserted % flush_every == 0:
                    self._flush_spool_buffers(redirect_spool_dir, redirect_buffers)
                    self._flush_spool_buffers(canonical_spool_dir, canonical_buffers)
                    print_inline_progress(f"[index-build] spooled={inserted:,}")

            self._flush_spool_buffers(redirect_spool_dir, redirect_buffers)
            self._flush_spool_buffers(canonical_spool_dir, canonical_buffers)

            finish_inline_progress(f"[index-build] spooled={inserted:,}")
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

    def get_synonyms(
        self,
        title: str,
        include_canonical: bool = True,
        tier: Literal["all", "filtered", "high_confidence"] = "filtered",
    ) -> list[str]:
        groups = self.get_synonym_groups(title, include_canonical=include_canonical)
        if tier == "all":
            return groups["all_synonyms"]
        if tier == "high_confidence":
            return groups["high_confidence_synonyms"]
        return groups["filtered_synonyms"]

    def get_all_synonyms(self, title: str, include_canonical: bool = True) -> list[str]:
        return self.get_synonyms(title, include_canonical=include_canonical, tier="all")

    def get_filtered_synonyms(self, title: str, include_canonical: bool = True) -> list[str]:
        return self.get_synonyms(title, include_canonical=include_canonical, tier="filtered")

    def get_high_confidence_synonyms(self, title: str, include_canonical: bool = True) -> list[str]:
        return self.get_synonyms(title, include_canonical=include_canonical, tier="high_confidence")

    def get_synonym_groups(self, title: str, include_canonical: bool = True) -> dict[str, list[str]]:
        canonical_title, redirects = self._lookup_canonical_and_redirects(title)
        if canonical_title is None:
            return {
                "all_synonyms": [],
                "filtered_synonyms": [],
                "high_confidence_synonyms": [],
            }

        filtered_redirects: list[str] = []
        high_confidence_redirects: list[str] = []
        for redirect in redirects:
            if self._is_typo_like_alias(redirect, canonical_title):
                continue
            score = self._score_alias_quality(redirect, canonical_title)
            if score >= 35:
                filtered_redirects.append(redirect)
            if score >= 55:
                high_confidence_redirects.append(redirect)

        return {
            "all_synonyms": self._assemble_synonym_list(canonical_title, redirects, include_canonical),
            "filtered_synonyms": self._assemble_synonym_list(
                canonical_title,
                filtered_redirects,
                include_canonical,
            ),
            "high_confidence_synonyms": self._assemble_synonym_list(
                canonical_title,
                high_confidence_redirects,
                include_canonical,
            ),
        }

    def get_node(self, title: str) -> WikipediaGraphNode | None:
        normalized = normalize_wikipedia_title(title)
        resolved = self._resolve_node_by_normalized_title(normalized)
        if resolved is None:
            return None
        return resolved

    def get_neighbors(
        self,
        title: str,
        directed: bool = False,
        include_incoming: bool = False,
    ) -> list[str]:
        node = self.get_node(title)
        if node is None:
            return []

        if not directed:
            neighbors: list[str] = []
            if node.redirect_target_title is not None:
                neighbors.append(node.redirect_target_title)
            neighbors.extend(node.incoming_redirects)
            return list(dict.fromkeys(neighbors))

        if include_incoming:
            neighbors = []
            if node.redirect_target_title is not None:
                neighbors.append(node.redirect_target_title)
            neighbors.extend(node.incoming_redirects)
            return list(dict.fromkeys(neighbors))

        if node.redirect_target_title is None:
            return []
        return [node.redirect_target_title]

    def hop_distance(self, source_title: str, target_title: str, directed: bool = False) -> int | None:
        source = self.get_node(source_title)
        target = self.get_node(target_title)
        if source is None or target is None:
            return None

        if source.node_id == target.node_id:
            return 0

        if directed:
            path = self._directed_path(source.normalized_title, target.normalized_title)
        else:
            path = self._undirected_path(source.normalized_title, target.normalized_title)
        if path is None:
            return None
        return len(path) - 1

    def shortest_path(
        self,
        source_title: str,
        target_title: str,
        directed: bool = False,
    ) -> list[str] | None:
        source = self.get_node(source_title)
        target = self.get_node(target_title)
        if source is None or target is None:
            return None

        if directed:
            path = self._directed_path(source.normalized_title, target.normalized_title)
        else:
            path = self._undirected_path(source.normalized_title, target.normalized_title)
        if path is None:
            return None
        return [self._display_title_for_normalized(normalized) for normalized in path]

    def are_connected_within_hops(
        self,
        source_title: str,
        target_title: str,
        max_hops: int,
        directed: bool = False,
    ) -> bool:
        if max_hops < 0:
            return False
        source = self.get_node(source_title)
        target = self.get_node(target_title)
        if source is None or target is None:
            return False
        if directed:
            path = self._directed_path(
                source.normalized_title,
                target.normalized_title,
                max_hops=max_hops,
            )
        else:
            path = self._undirected_path(
                source.normalized_title,
                target.normalized_title,
                max_hops=max_hops,
            )
        return path is not None and len(path) - 1 <= max_hops

    def graph_stats(self) -> dict[str, int]:
        stats = self.stats()
        redirect_count = stats["redirects"]
        target_count = stats["canonical_pages"]
        return {
            "redirect_nodes": redirect_count,
            "target_nodes": target_count,
            "max_total_nodes": target_count + redirect_count,
            "directed_edges": redirect_count,
            "undirected_edges": redirect_count,
        }

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

    def _resolve_node_by_normalized_title(self, normalized: str) -> WikipediaGraphNode | None:
        redirect_bucket = self._load_bucket(get_bucket_key(normalized))
        redirect_entry = redirect_bucket.get(normalized)
        canonical_entry = self._load_canonical_bucket(get_bucket_key(normalized)).get(normalized)
        if redirect_entry is None and canonical_entry is None:
            return None

        title = self._display_title_from_entries(normalized, redirect_entry, canonical_entry)
        redirect_target_title: str | None = None
        redirect_target_id: str | None = None
        if redirect_entry is not None:
            redirect_target_title = str(redirect_entry[1])
            redirect_target_id = self._make_node_id(normalize_wikipedia_title(redirect_target_title))

        incoming_redirects: tuple[str, ...]
        if canonical_entry is None:
            incoming_redirects = ()
        else:
            incoming_redirects = tuple(canonical_entry["redirects"])

        return WikipediaGraphNode(
            node_id=self._make_node_id(normalized),
            title=title,
            normalized_title=normalized,
            redirect_target_id=redirect_target_id,
            redirect_target_title=redirect_target_title,
            incoming_redirects=incoming_redirects,
        )

    @staticmethod
    def _make_node_id(normalized_title: str) -> str:
        return f"t:{normalized_title}"

    def _display_title_from_entries(
        self,
        normalized: str,
        redirect_entry: tuple[str, str] | None,
        canonical_entry: dict[str, object] | None,
    ) -> str:
        if redirect_entry is not None:
            return str(redirect_entry[0])
        if canonical_entry is not None:
            return str(canonical_entry["title"])
        return normalized

    def _lookup_canonical_and_redirects(self, title: str) -> tuple[str | None, list[str]]:
        normalized = normalize_wikipedia_title(title)
        redirect_bucket = self._load_bucket(get_bucket_key(normalized))
        redirect_entry = redirect_bucket.get(normalized)
        canonical_bucket = self._load_canonical_bucket(get_bucket_key(normalized))
        local_canonical_entry = canonical_bucket.get(normalized)

        target_canonical_entry: dict[str, object] | None = None
        if redirect_entry is not None:
            target_normalized = normalize_wikipedia_title(redirect_entry[1])
            target_canonical_entry = self._load_canonical_bucket(get_bucket_key(target_normalized)).get(
                target_normalized
            )

        if local_canonical_entry is None and target_canonical_entry is None:
            return None, []

        canonical_entry = self._choose_synonym_anchor_entry(
            local_canonical_entry=local_canonical_entry,
            target_canonical_entry=target_canonical_entry,
        )
        if canonical_entry is None:
            return None, []

        canonical_title = str(canonical_entry["title"])
        redirects = list(canonical_entry["redirects"])
        return canonical_title, redirects

    @staticmethod
    def _choose_synonym_anchor_entry(
        local_canonical_entry: dict[str, object] | None,
        target_canonical_entry: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if local_canonical_entry is None:
            return target_canonical_entry
        if target_canonical_entry is None:
            return local_canonical_entry

        local_redirects = len(local_canonical_entry["redirects"])
        target_redirects = len(target_canonical_entry["redirects"])

        if target_redirects > local_redirects:
            return target_canonical_entry
        return local_canonical_entry

    @staticmethod
    def _assemble_synonym_list(
        canonical_title: str,
        redirects: Sequence[str],
        include_canonical: bool,
    ) -> list[str]:
        if include_canonical:
            return [canonical_title, *redirects]
        return list(redirects)

    def _score_alias_quality(self, alias: str, canonical_title: str) -> int:
        alias_normalized = normalize_wikipedia_title(alias)
        canonical_normalized = normalize_wikipedia_title(canonical_title)

        if not alias_normalized:
            return -100
        if alias_normalized == canonical_normalized:
            return 100

        alias_tokens = alias_normalized.split()
        canonical_tokens = canonical_normalized.split()
        alias_token_set = set(alias_tokens)
        canonical_token_set = set(canonical_tokens)

        score = 0

        if self._is_acronym_alias(alias_normalized, canonical_tokens):
            score += 55

        token_overlap = len(alias_token_set & canonical_token_set)
        if token_overlap > 0:
            score += min(30, token_overlap * 12)

        if alias_token_set and alias_token_set <= canonical_token_set:
            score += 12
        if canonical_token_set and canonical_token_set <= alias_token_set:
            score += 8

        similarity = SequenceMatcher(None, alias_normalized, canonical_normalized).ratio()
        if similarity >= 0.92:
            score += 30
        elif similarity >= 0.82:
            score += 20
        elif similarity >= 0.70:
            score += 10
        elif similarity < 0.45:
            score -= 18

        raw_casefold = unicodedata.normalize("NFKC", str(alias).strip()).casefold()
        punctuation_count = sum(
            1 for char in raw_casefold if not char.isalnum() and not char.isspace()
        )
        apostrophe_count = raw_casefold.count("'")
        digit_count = sum(1 for char in raw_casefold if char.isdigit())

        if apostrophe_count > 0:
            score -= 25
        if punctuation_count >= 3:
            score -= 10
        if digit_count > 0 and not any(char.isdigit() for char in canonical_title):
            score -= 15

        if len(alias_tokens) > max(len(canonical_tokens) + 2, 5):
            score -= 20
        if len(alias_normalized) > max(len(canonical_normalized) + 12, 36):
            score -= 12

        extra_token_count = len(alias_token_set - canonical_token_set)
        if extra_token_count >= 3:
            score -= 16
        elif extra_token_count == 2:
            score -= 8

        extra_low_information = sum(
            1 for token in alias_tokens if token not in canonical_token_set and token in _LOW_INFORMATION_TOKENS
        )
        score -= extra_low_information * 6

        extra_short_tokens = sum(
            1 for token in alias_tokens if token not in canonical_token_set and len(token) <= 2
        )
        score -= extra_short_tokens * 4

        if alias_tokens and canonical_tokens and alias_tokens[0] in _LOW_INFORMATION_TOKENS:
            score -= 6

        single_char_tokens = sum(1 for token in alias_tokens if len(token) == 1)
        if single_char_tokens >= 3 and not self._is_acronym_alias(alias_normalized, canonical_tokens):
            score -= 10

        return score

    def _is_typo_like_alias(self, alias: str, canonical_title: str) -> bool:
        alias_normalized = normalize_wikipedia_title(alias)
        canonical_normalized = normalize_wikipedia_title(canonical_title)
        if not alias_normalized or alias_normalized == canonical_normalized:
            return False
        if self._is_acronym_alias(alias_normalized, canonical_normalized.split()):
            return False

        alias_tokens = alias_normalized.split()
        canonical_tokens = canonical_normalized.split()

        if not alias_tokens or not canonical_tokens:
            return False
        if abs(len(alias_tokens) - len(canonical_tokens)) > 1:
            return False

        aligned_pairs = list(zip(alias_tokens, canonical_tokens))
        exact_matches = sum(1 for alias_token, canonical_token in aligned_pairs if alias_token == canonical_token)
        mismatched_pairs = [
            (alias_token, canonical_token)
            for alias_token, canonical_token in aligned_pairs
            if alias_token != canonical_token
        ]

        token_overlap = len(set(alias_tokens) & set(canonical_tokens))
        shared_ratio = token_overlap / max(len(set(canonical_tokens)), 1)
        string_similarity = SequenceMatcher(None, alias_normalized, canonical_normalized).ratio()

        if exact_matches == 0 and string_similarity < 0.82:
            return False
        if shared_ratio < 0.5 and string_similarity < 0.88:
            return False

        if len(alias_tokens) == len(canonical_tokens):
            typoish_changes = 0
            for alias_token, canonical_token in mismatched_pairs:
                if self._is_small_token_typo(alias_token, canonical_token):
                    typoish_changes += 1
                else:
                    return False
            return typoish_changes > 0

        if len(alias_tokens) + 1 == len(canonical_tokens):
            return self._is_single_missing_token_variant(alias_tokens, canonical_tokens)

        if len(canonical_tokens) + 1 == len(alias_tokens):
            return self._is_single_missing_token_variant(canonical_tokens, alias_tokens)

        return False

    @staticmethod
    def _is_small_token_typo(alias_token: str, canonical_token: str) -> bool:
        if alias_token == canonical_token:
            return False
        if abs(len(alias_token) - len(canonical_token)) > 2:
            return False

        similarity = SequenceMatcher(None, alias_token, canonical_token).ratio()
        if similarity < 0.72:
            return False

        if WikipediaRedirectIndex._is_adjacent_transposition(alias_token, canonical_token):
            return True

        if alias_token in canonical_token or canonical_token in alias_token:
            return True

        edits = WikipediaRedirectIndex._bounded_edit_distance(alias_token, canonical_token, max_distance=2)
        if edits is None:
            return False
        return edits <= 2

    @staticmethod
    def _is_single_missing_token_variant(shorter_tokens: Sequence[str], longer_tokens: Sequence[str]) -> bool:
        for skip_index in range(len(longer_tokens)):
            candidate = list(longer_tokens[:skip_index]) + list(longer_tokens[skip_index + 1 :])
            if list(shorter_tokens) != candidate:
                continue
            skipped_token = longer_tokens[skip_index]
            return len(skipped_token) <= 2 or skipped_token in _LOW_INFORMATION_TOKENS
        return False

    @staticmethod
    def _is_adjacent_transposition(left: str, right: str) -> bool:
        if len(left) != len(right):
            return False
        mismatches = [index for index, (l_char, r_char) in enumerate(zip(left, right)) if l_char != r_char]
        if len(mismatches) != 2:
            return False
        first, second = mismatches
        if second != first + 1:
            return False
        return left[first] == right[second] and left[second] == right[first]

    @staticmethod
    def _bounded_edit_distance(left: str, right: str, max_distance: int) -> int | None:
        if abs(len(left) - len(right)) > max_distance:
            return None

        previous_row = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current_row = [left_index]
            row_min = current_row[0]
            for right_index, right_char in enumerate(right, start=1):
                insert_cost = current_row[right_index - 1] + 1
                delete_cost = previous_row[right_index] + 1
                replace_cost = previous_row[right_index - 1] + (left_char != right_char)
                value = min(insert_cost, delete_cost, replace_cost)
                current_row.append(value)
                if value < row_min:
                    row_min = value
            if row_min > max_distance:
                return None
            previous_row = current_row

        distance = previous_row[-1]
        if distance > max_distance:
            return None
        return distance

    @staticmethod
    def _is_acronym_alias(alias_normalized: str, canonical_tokens: Sequence[str]) -> bool:
        acronym = "".join(token[0] for token in canonical_tokens if token)
        compact_alias = alias_normalized.replace(" ", "")
        if not acronym:
            return False
        if compact_alias == acronym:
            return True
        if compact_alias == acronym[:2]:
            return True
        return False

    def _display_title_for_normalized(self, normalized: str) -> str:
        node = self._resolve_node_by_normalized_title(normalized)
        if node is None:
            return normalized
        return node.title

    def _get_outgoing_normalized(self, normalized: str) -> str | None:
        redirect_entry = self._load_bucket(get_bucket_key(normalized)).get(normalized)
        if redirect_entry is None:
            return None
        return normalize_wikipedia_title(redirect_entry[1])

    def _get_incoming_normalized(self, normalized: str) -> tuple[str, ...]:
        canonical_entry = self._load_canonical_bucket(get_bucket_key(normalized)).get(normalized)
        if canonical_entry is None:
            return ()
        return tuple(str(redirect) for redirect in canonical_entry["redirects"])

    def _iter_neighbor_normalized(self, normalized: str, directed: bool) -> Iterator[str]:
        outgoing = self._get_outgoing_normalized(normalized)
        if outgoing is not None:
            yield outgoing
        if not directed:
            yield from self._get_incoming_normalized(normalized)

    def _directed_path(
        self,
        source_normalized: str,
        target_normalized: str,
        max_hops: int | None = None,
    ) -> list[str] | None:
        path = [source_normalized]
        if source_normalized == target_normalized:
            return path

        visited = {source_normalized}
        current = source_normalized
        steps = 0
        while True:
            if max_hops is not None and steps >= max_hops:
                return None
            next_node = self._get_outgoing_normalized(current)
            if next_node is None or next_node in visited:
                return None
            path.append(next_node)
            steps += 1
            if next_node == target_normalized:
                return path
            visited.add(next_node)
            current = next_node

    def _undirected_path(
        self,
        source_normalized: str,
        target_normalized: str,
        max_hops: int | None = None,
    ) -> list[str] | None:
        if source_normalized == target_normalized:
            return [source_normalized]

        frontier_from_source = {source_normalized}
        frontier_from_target = {target_normalized}
        parents_from_source: dict[str, str | None] = {source_normalized: None}
        parents_from_target: dict[str, str | None] = {target_normalized: None}
        depth_from_source = 0
        depth_from_target = 0

        while frontier_from_source and frontier_from_target:
            if max_hops is not None and depth_from_source + depth_from_target >= max_hops:
                direct_overlap = frontier_from_source & frontier_from_target
                if direct_overlap:
                    meeting = next(iter(direct_overlap))
                    return self._reconstruct_bidirectional_path(
                        meeting,
                        parents_from_source,
                        parents_from_target,
                    )
                return None

            expand_source = len(frontier_from_source) <= len(frontier_from_target)
            if expand_source:
                if max_hops is not None and depth_from_source >= max_hops:
                    return None
                frontier_from_source, meeting = self._expand_frontier(
                    frontier_from_source,
                    parents_from_source,
                    parents_from_target,
                )
                depth_from_source += 1
            else:
                if max_hops is not None and depth_from_target >= max_hops:
                    return None
                frontier_from_target, meeting = self._expand_frontier(
                    frontier_from_target,
                    parents_from_target,
                    parents_from_source,
                )
                depth_from_target += 1

            if meeting is not None:
                return self._reconstruct_bidirectional_path(
                    meeting,
                    parents_from_source,
                    parents_from_target,
                )

        return None

    def _expand_frontier(
        self,
        frontier: set[str],
        parents_here: dict[str, str | None],
        parents_other: dict[str, str | None],
    ) -> tuple[set[str], str | None]:
        next_frontier: set[str] = set()
        for normalized in frontier:
            for neighbor in self._iter_neighbor_normalized(normalized, directed=False):
                if neighbor in parents_here:
                    continue
                parents_here[neighbor] = normalized
                if neighbor in parents_other:
                    return next_frontier, neighbor
                next_frontier.add(neighbor)
        return next_frontier, None

    def _reconstruct_bidirectional_path(
        self,
        meeting: str,
        parents_from_source: dict[str, str | None],
        parents_from_target: dict[str, str | None],
    ) -> list[str]:
        left_path: list[str] = []
        current: str | None = meeting
        while current is not None:
            left_path.append(current)
            current = parents_from_source[current]
        left_path.reverse()

        right_path: list[str] = []
        current = parents_from_target[meeting]
        while current is not None:
            right_path.append(current)
            current = parents_from_target[current]

        return left_path + right_path

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
                print_inline_progress(
                    f"[index-build] redirect buckets finalized={index:,}/{total_files:,}"
                )
        if total_files > 0:
            finish_inline_progress(
                f"[index-build] redirect buckets finalized={total_files:,}/{total_files:,}"
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
                print_inline_progress(
                    f"[index-build] canonical buckets finalized={index:,}/{total_files:,}"
                )
        if total_files > 0:
            finish_inline_progress(
                f"[index-build] canonical buckets finalized={total_files:,}/{total_files:,}"
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
