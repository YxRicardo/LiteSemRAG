from __future__ import annotations

import argparse
import gzip
import pickle
import random
from pathlib import Path

from wikipedia_redirects import WikipediaRedirectIndex, get_bucket_key, normalize_wikipedia_title


def load_pickle_gz(path: Path) -> object:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def print_basic_stats(index_dir: Path) -> None:
    with WikipediaRedirectIndex(index_dir) as index:
        print("Index stats:")
        print(index.stats())
        metadata_path = index.metadata_path
        if metadata_path.exists():
            print(f"Metadata file: {metadata_path}")
            print(metadata_path.read_text(encoding="utf-8"))

    redirect_bucket_dir = index_dir / "redirect_buckets"
    canonical_bucket_dir = index_dir / "canonical_buckets"
    redirect_bucket_count = len(list(redirect_bucket_dir.glob("*.pkl.gz")))
    canonical_bucket_count = len(list(canonical_bucket_dir.glob("*.pkl.gz")))
    print(f"Redirect bucket files: {redirect_bucket_count}")
    print(f"Canonical bucket files: {canonical_bucket_count}")


def inspect_title(index_dir: Path, title: str, synonym_limit: int) -> None:
    normalized = normalize_wikipedia_title(title)
    redirect_bucket = get_bucket_key(normalized)

    with WikipediaRedirectIndex(index_dir) as index:
        canonical = index.resolve_redirect(title)
        synonyms = index.get_synonyms(title)

    print(f"Query: {title}")
    print(f"Normalized: {normalized}")
    print(f"Bucket key: {redirect_bucket}")
    print(f"Canonical: {canonical}")
    print(f"Synonyms ({len(synonyms)}): {synonyms[:synonym_limit]}")


def sample_pairs(index_dir: Path, sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    redirect_bucket_dir = index_dir / "redirect_buckets"
    bucket_files = sorted(redirect_bucket_dir.glob("*.pkl.gz"))
    if not bucket_files:
        print("No redirect buckets found.")
        return

    chosen_files = rng.sample(bucket_files, k=min(sample_size, len(bucket_files)))
    print("Sample redirect bucket contents:")
    for bucket_file in chosen_files:
        bucket = load_pickle_gz(bucket_file)
        print(f"[{bucket_file.name}] size={len(bucket)}")
        for idx, (normalized_redirect, pair) in enumerate(bucket.items()):
            print(f"  {normalized_redirect} -> {pair[1]} (raw={pair[0]})")
            if idx >= 2:
                break


def inspect_bucket(index_dir: Path, bucket_key: str, limit: int) -> None:
    redirect_path = index_dir / "redirect_buckets" / f"{bucket_key}.pkl.gz"
    canonical_path = index_dir / "canonical_buckets" / f"{bucket_key}.pkl.gz"

    if redirect_path.exists():
        redirect_bucket = load_pickle_gz(redirect_path)
        print(f"Redirect bucket {bucket_key}: {len(redirect_bucket)} entries")
        for idx, (normalized_redirect, pair) in enumerate(redirect_bucket.items()):
            print(f"  {normalized_redirect} -> {pair[1]} (raw={pair[0]})")
            if idx + 1 >= limit:
                break
    else:
        print(f"Redirect bucket {bucket_key} not found.")

    if canonical_path.exists():
        canonical_bucket = load_pickle_gz(canonical_path)
        print(f"Canonical bucket {bucket_key}: {len(canonical_bucket)} entries")
        for idx, (normalized_canonical, value) in enumerate(canonical_bucket.items()):
            redirects = value["redirects"][: min(limit, len(value["redirects"]))]
            print(f"  {normalized_canonical} -> title={value['title']}, redirects={redirects}")
            if idx + 1 >= limit:
                break
    else:
        print(f"Canonical bucket {bucket_key} not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a built Wikipedia redirect index.")
    parser.add_argument(
        "--index-dir",
        default="data/wikipedia_redirects/redirect_index",
        help="Path to the redirect index directory.",
    )
    parser.add_argument("--title", help="Resolve and inspect one title.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Print sample entries from randomly chosen redirect buckets.",
    )
    parser.add_argument(
        "--bucket",
        help="Inspect one bucket key directly, for example: us",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max items to print for title synonyms or bucket entries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir)

    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")

    print_basic_stats(index_dir)

    if args.title:
        print()
        inspect_title(index_dir, args.title, synonym_limit=args.limit)

    if args.sample > 0:
        print()
        sample_pairs(index_dir, sample_size=args.sample, seed=args.seed)

    if args.bucket:
        print()
        inspect_bucket(index_dir, bucket_key=args.bucket, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
