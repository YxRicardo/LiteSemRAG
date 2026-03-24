from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

from wikipedia_redirects import RedirectPair, WikipediaRedirectIndex


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return destination


def iter_sql_insert_rows(path: Path, table_name: str) -> Iterator[list[object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    prefix = f"INSERT INTO `{table_name}` VALUES "

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(prefix):
                continue
            values_part = line[len(prefix):].rstrip().rstrip(";")
            yield from parse_sql_values(values_part)


def parse_sql_values(values_part: str) -> Iterator[list[object]]:
    i = 0
    n = len(values_part)

    while i < n:
        if values_part[i] != "(":
            i += 1
            continue
        i += 1
        row: list[object] = []
        token_chars: list[str] = []
        in_string = False
        escape_next = False

        while i < n:
            char = values_part[i]

            if in_string:
                if escape_next:
                    token_chars.append(char)
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == "'":
                    in_string = False
                else:
                    token_chars.append(char)
                i += 1
                continue

            if char == "'":
                in_string = True
                i += 1
                continue

            if char == ",":
                row.append(convert_sql_value("".join(token_chars)))
                token_chars.clear()
                i += 1
                continue

            if char == ")":
                row.append(convert_sql_value("".join(token_chars)))
                token_chars.clear()
                i += 1
                break

            token_chars.append(char)
            i += 1

        yield row


def convert_sql_value(raw: str) -> object:
    value = raw.strip()
    if value == "NULL":
        return None
    if value == "":
        return ""
    if value.lstrip("-").isdigit():
        return int(value)
    return value.replace("_", " ")


def iter_redirect_pairs(
    page_dump_path: Path,
    redirect_dump_path: Path,
    article_namespace: int = 0,
) -> Iterator[RedirectPair]:
    redirect_targets: dict[int, str] = {}

    for row in iter_sql_insert_rows(redirect_dump_path, "redirect"):
        if len(row) < 3:
            continue
        page_id = int(row[0])
        namespace = int(row[1])
        target_title = str(row[2])
        if namespace != article_namespace:
            continue
        redirect_targets[page_id] = target_title

    for row in iter_sql_insert_rows(page_dump_path, "page"):
        if len(row) < 5:
            continue
        page_id = int(row[0])
        namespace = int(row[1])
        page_title = str(row[2])
        is_redirect = int(row[4])

        if namespace != article_namespace or is_redirect != 1:
            continue

        target_title = redirect_targets.get(page_id)
        if target_title is None:
            continue

        yield RedirectPair(redirect=page_title, canonical=target_title)


def build_index(
    page_dump_path: Path,
    redirect_dump_path: Path,
    index_dir: Path,
    wiki: str,
    dump_tag: str,
) -> dict[str, int]:
    with WikipediaRedirectIndex(index_dir) as index:
        index.clear()
        inserted = index.bulk_add_pairs(iter_redirect_pairs(page_dump_path, redirect_dump_path))
        index.set_metadata("wiki", wiki)
        index.set_metadata("dump_tag", dump_tag)
        index.set_metadata("page_dump", str(page_dump_path))
        index.set_metadata("redirect_dump", str(redirect_dump_path))
        stats = index.stats()
        stats["inserted"] = inserted
        return stats


def make_dump_url(wiki: str, dump_tag: str, dump_name: str) -> str:
    return f"https://dumps.wikimedia.org/{wiki}/{dump_tag}/{wiki}-{dump_tag}-{dump_name}.sql.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Wikipedia page/redirect dumps and build a redirect lookup index."
    )
    parser.add_argument("--wiki", default="enwiki", help="Wiki project name, for example: enwiki")
    parser.add_argument("--dump-tag", default="latest", help="Dump tag, for example: latest or 20260320")
    parser.add_argument(
        "--output-dir",
        default="data/wikipedia_redirects",
        help="Directory for downloaded dumps and the generated redirect index.",
    )
    parser.add_argument(
        "--index-name",
        default="redirect_index",
        help="Index directory name created under output-dir.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing dump files instead of downloading them.",
    )
    parser.add_argument(
        "--page-dump",
        help="Optional explicit path to page.sql.gz. Overrides output-dir convention.",
    )
    parser.add_argument(
        "--redirect-dump",
        help="Optional explicit path to redirect.sql.gz. Overrides output-dir convention.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    dump_tag = args.dump_tag
    wiki = args.wiki

    page_dump_path = Path(args.page_dump) if args.page_dump else raw_dir / f"{wiki}-{dump_tag}-page.sql.gz"
    redirect_dump_path = (
        Path(args.redirect_dump)
        if args.redirect_dump
        else raw_dir / f"{wiki}-{dump_tag}-redirect.sql.gz"
    )
    index_dir = output_dir / args.index_name

    if not args.skip_download:
        print(f"Downloading dump files for {wiki} ({dump_tag})...")
        download_file(make_dump_url(wiki, dump_tag, "page"), page_dump_path)
        download_file(make_dump_url(wiki, dump_tag, "redirect"), redirect_dump_path)

    if not page_dump_path.exists():
        raise FileNotFoundError(f"Missing page dump: {page_dump_path}")
    if not redirect_dump_path.exists():
        raise FileNotFoundError(f"Missing redirect dump: {redirect_dump_path}")

    print("Building redirect index...")
    stats = build_index(page_dump_path, redirect_dump_path, index_dir, wiki=wiki, dump_tag=dump_tag)
    print(f"Redirect index directory: {index_dir}")
    print(f"Inserted redirect pairs: {stats['inserted']}")
    print(f"Canonical pages: {stats['canonical_pages']}")
    print(f"Redirect entries: {stats['redirects']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
