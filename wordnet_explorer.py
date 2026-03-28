from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from typing import Any, Optional


class WordNetExplorer:
    """Reusable WordNet helper extracted from `explore_wordnet.ipynb`."""

    def __init__(self, ensure_resources: bool = True, auto_install: bool = False) -> None:
        self.nltk, self.wn = self._load_nltk(auto_install=auto_install)
        self.pos_map = {
            None: None,
            "noun": self.wn.NOUN,
            "verb": self.wn.VERB,
            "adjective": self.wn.ADJ,
            "adverb": self.wn.ADV,
        }
        self.pos_label = {
            "n": "noun",
            "v": "verb",
            "a": "adjective",
            "s": "adjective satellite",
            "r": "adverb",
        }

        if ensure_resources:
            self.ensure_nltk_resource("corpora/wordnet", "wordnet")
            self.ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

    @staticmethod
    def _load_nltk(auto_install: bool = False):
        try:
            nltk = importlib.import_module("nltk")
        except ImportError as exc:
            if not auto_install:
                raise ImportError(
                    "nltk is required. Install it with `pip install nltk`, "
                    "or initialize WordNetExplorer(auto_install=True)."
                ) from exc

            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            nltk = importlib.import_module("nltk")

        wn = importlib.import_module("nltk.corpus").wordnet
        return nltk, wn

    def ensure_nltk_resource(self, resource_path: str, download_name: str) -> None:
        try:
            self.nltk.data.find(resource_path)
        except LookupError:
            self.nltk.download(download_name)

    @staticmethod
    def _normalize_lookup(text: str) -> str:
        return text.replace("_", " ").strip().lower()

    @staticmethod
    def _clean_list(values: list[str], empty_text: str = "(none)") -> str:
        if not values:
            return empty_text
        return ", ".join(values)

    def _normalize_pos(self, pos: Optional[str]) -> Optional[str]:
        normalized_pos = pos.lower().strip() if isinstance(pos, str) else None
        if normalized_pos not in self.pos_map:
            valid = ["noun", "verb", "adjective", "adverb", "None"]
            raise ValueError(f"Invalid pos='{pos}'. Valid options: {', '.join(valid)}")
        return normalized_pos

    @staticmethod
    def _validate_word(word: str) -> str:
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Please provide a non-empty word string.")
        return word.strip()

    @classmethod
    def _wordnet_lookup_candidates(cls, word: str) -> list[str]:
        validated_word = cls._validate_word(word)
        space_normalized = " ".join(validated_word.split())
        lowered_word = space_normalized.lower()

        candidates: list[str] = []

        def add(candidate: str) -> None:
            candidate = candidate.strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for base in (space_normalized, lowered_word):
            add(base)
            add(base.replace("_", " "))
            add(base.replace("-", " "))
            add(base.replace(" ", "_"))
            add(base.replace("-", "_"))
            add(base.replace("-", " ").replace(" ", "_"))

        return candidates

    def _synset_items(self, synsets) -> list[str]:
        items = []
        for synset in synsets:
            try:
                items.append(f"{synset.name()} - {synset.definition()}")
            except Exception:
                items.append(str(synset))
        return items

    @staticmethod
    def _print_section(title: str, values: list[str]) -> None:
        print(f"  {title}:")
        if not values:
            print("    (none)")
            return
        for item in values:
            print(f"    - {item}")

    @staticmethod
    def _synset_head_word(synset) -> str:
        return synset.name().split(".", 1)[0].replace("_", " ")

    def _filter_synsets(
        self,
        synsets,
        word: str,
        noun_only_words: bool,
        exact_match_only: bool,
    ) -> list[tuple[Any, list[str]]]:
        if not noun_only_words and not exact_match_only:
            return [
                (synset, [lemma.name().replace("_", " ") for lemma in synset.lemmas()])
                for synset in synsets
            ]

        normalized_word = self._normalize_lookup(word)
        filtered_synsets = []

        for synset in synsets:
            if noun_only_words and synset.pos() != self.wn.NOUN:
                continue

            if exact_match_only and self._normalize_lookup(self._synset_head_word(synset)) != normalized_word:
                continue

            matched_lemmas = []
            for lemma in synset.lemmas():
                lemma_name = lemma.name()
                if exact_match_only and self._normalize_lookup(lemma_name) != normalized_word:
                    continue
                matched_lemmas.append(lemma_name.replace("_", " "))

            if matched_lemmas:
                filtered_synsets.append((synset, matched_lemmas))

        return filtered_synsets

    def _get_raw_synsets(self, word: str, pos: Optional[str] = None):
        normalized_pos = self._normalize_pos(pos)
        wn_pos = self.pos_map[normalized_pos]
        synsets = []
        seen_synset_names = set()

        for candidate in self._wordnet_lookup_candidates(word):
            current_synsets = self.wn.synsets(candidate, pos=wn_pos) if wn_pos else self.wn.synsets(candidate)
            for synset in current_synsets:
                synset_name = synset.name()
                if synset_name in seen_synset_names:
                    continue
                seen_synset_names.add(synset_name)
                synsets.append(synset)

        return synsets

    def get_synset_details(
        self,
        word: str,
        pos: Optional[str] = None,
        noun_only_words: bool = False,
        exact_match_only: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_word = self._validate_word(word)
        raw_synsets = self._get_raw_synsets(normalized_word, pos=pos)
        synsets = self._filter_synsets(
            raw_synsets,
            normalized_word,
            noun_only_words=noun_only_words,
            exact_match_only=exact_match_only,
        )

        results = []
        for synset, lemmas in synsets:
            similar_tos = synset.similar_tos() if hasattr(synset, "similar_tos") else []
            results.append(
                {
                    "synset_name": synset.name(),
                    "pos": self.pos_label.get(synset.pos(), synset.pos()),
                    "definition": synset.definition() or "(none)",
                    "examples": synset.examples() or [],
                    "lemmas": lemmas,
                    "hypernyms": self._synset_items(synset.hypernyms()),
                    "hyponyms": self._synset_items(synset.hyponyms()),
                    "member_holonyms": self._synset_items(synset.member_holonyms()),
                    "part_holonyms": self._synset_items(synset.part_holonyms()),
                    "substance_holonyms": self._synset_items(synset.substance_holonyms()),
                    "member_meronyms": self._synset_items(synset.member_meronyms()),
                    "part_meronyms": self._synset_items(synset.part_meronyms()),
                    "substance_meronyms": self._synset_items(synset.substance_meronyms()),
                    "attributes": self._synset_items(synset.attributes()),
                    "entailments": self._synset_items(synset.entailments()),
                    "similar_tos": self._synset_items(similar_tos),
                }
            )

        return results

    def collect_wordnet_definitions(
        self,
        word: str,
        pos: Optional[str] = None,
        noun_only_words: bool = True,
        exact_match_only: bool = True,
    ) -> list[str]:
        return [
            item["definition"]
            for item in self.get_synset_details(
                word,
                pos=pos,
                noun_only_words=noun_only_words,
                exact_match_only=exact_match_only,
            )
        ]

    def inspect_wordnet(
        self,
        word: str,
        pos: Optional[str] = None,
        noun_only_words: bool = False,
        exact_match_only: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_word = self._validate_word(word)
        normalized_pos = self._normalize_pos(pos)
        results = self.get_synset_details(
            normalized_word,
            pos=normalized_pos,
            noun_only_words=noun_only_words,
            exact_match_only=exact_match_only,
        )

        print("=" * 90)
        print(f"WordNet inspection for word: '{normalized_word}'")
        print(f"POS filter: {normalized_pos if normalized_pos else 'None (all parts of speech)'}")
        print(f"Noun-only synsets: {noun_only_words}")
        print(f"Exact lemma match: {exact_match_only}")
        print(f"Synsets found: {len(results)}")
        print("=" * 90)

        if not results:
            print("No synsets found for this word with the selected filters.")
            return results

        for index, item in enumerate(results, start=1):
            print(f"\n[{index}] {item['synset_name']}")
            print("-" * 90)
            print(f"  Synset name : {item['synset_name']}")
            print(f"  POS         : {item['pos']}")
            print(f"  Definition  : {item['definition']}")

            self._print_section("Example sentences", item["examples"])
            self._print_section("Lemma names (synonyms)", item["lemmas"])
            self._print_section("Hypernyms", item["hypernyms"])
            self._print_section("Hyponyms", item["hyponyms"])
            self._print_section("Member holonyms", item["member_holonyms"])
            self._print_section("Part holonyms", item["part_holonyms"])
            self._print_section("Substance holonyms", item["substance_holonyms"])
            self._print_section("Member meronyms", item["member_meronyms"])
            self._print_section("Part meronyms", item["part_meronyms"])
            self._print_section("Substance meronyms", item["substance_meronyms"])
            self._print_section("Attributes", item["attributes"])
            self._print_section("Entailments", item["entailments"])
            self._print_section("Similar_tos", item["similar_tos"])
            print("-" * 90)

        print("\nInspection complete.")
        return results

    def print_hypernym_paths_for_word(
        self,
        word: str,
        pos: Optional[str] = None,
        synset_index: int = 0,
    ) -> list[list[str]]:
        synsets = self._get_raw_synsets(word, pos=pos)

        if not synsets:
            print(f"No synsets found for '{word}'.")
            return []

        if synset_index < 0 or synset_index >= len(synsets):
            raise IndexError(f"synset_index out of range. Choose 0 to {len(synsets) - 1}.")

        target = synsets[synset_index]
        paths = target.hypernym_paths() or []

        print("=" * 90)
        print(f"Hypernym paths for: {target.name()} - {target.definition()}")
        print(f"Total paths: {len(paths)}")
        print("=" * 90)

        if not paths:
            print("No hypernym paths available for this synset.")
            return []

        formatted_paths: list[list[str]] = []
        for path_index, path in enumerate(paths, start=1):
            print(f"\nPath {path_index}:")
            current_path = []
            for depth, node in enumerate(path):
                line = f"{node.name()} ({self.pos_label.get(node.pos(), node.pos())})"
                current_path.append(line)
                indent = "  " * depth
                print(f"{indent}- {line}")
            formatted_paths.append(current_path)

        return formatted_paths

    def compare_words_synsets(
        self,
        word1: str,
        word2: str,
        pos: Optional[str] = None,
        max_synsets: int = 8,
    ) -> list[tuple[str, str]]:
        synsets1 = self._get_raw_synsets(word1, pos=pos)
        synsets2 = self._get_raw_synsets(word2, pos=pos)

        print("=" * 120)
        print(f"Synset comparison: '{word1}' vs '{word2}' | POS: {self._normalize_pos(pos) if pos else 'all'}")
        print(f"Counts: {len(synsets1)} vs {len(synsets2)}")
        print("=" * 120)

        if not synsets1 and not synsets2:
            print("No synsets found for either word.")
            return []

        width = 58
        rows = max(min(len(synsets1), max_synsets), min(len(synsets2), max_synsets), 1)

        def synset_line(synset) -> str:
            if synset is None:
                return "(none)"
            return f"{synset.name()} - {synset.definition()}"

        print(f"{f'{word1} synsets':<{width}} | {f'{word2} synsets':<{width}}")
        print("-" * width + "-+-" + "-" * width)

        comparison_rows = []
        for index in range(rows):
            left_synset = synsets1[index] if index < len(synsets1) and index < max_synsets else None
            right_synset = synsets2[index] if index < len(synsets2) and index < max_synsets else None

            left_text = textwrap.shorten(synset_line(left_synset), width=width, placeholder="...")
            right_text = textwrap.shorten(synset_line(right_synset), width=width, placeholder="...")
            comparison_rows.append((left_text, right_text))
            print(f"{left_text:<{width}} | {right_text:<{width}}")

        if len(synsets1) > max_synsets or len(synsets2) > max_synsets:
            print("\n(Comparison truncated by max_synsets.)")

        return comparison_rows


if __name__ == "__main__":
    explorer = WordNetExplorer()
    explorer.inspect_wordnet("director")
