from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Iterator

from wikipedia_redirects import RedirectPair, WikipediaRedirectIndex


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def is_valid_gzip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False


class ProgressLogger:
    def __init__(self, label: str, report_every: int = 200000) -> None:
        self.label = label
        self.report_every = report_every
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def maybe_report(self, count: int, extra: str = "") -> None:
        if count == 0 or count % self.report_every != 0:
            return
        self.report(count, extra=extra)

    def report(self, count: int, extra: str = "") -> None:
        now = time.time()
        elapsed = max(now - self.start_time, 1e-6)
        speed = count / elapsed
        suffix = f", {extra}" if extra else ""
        print(
            f"[{self.label}] processed={count:,}, rate={speed:,.0f}/s{suffix}",
            flush=True,
        )
        self.last_report_time = now

    def done(self, count: int, extra: str = "") -> None:
        self.report(count, extra=extra)


def print_inline_progress(message: str) -> None:
    print(f"\r{message}", end="", flush=True)


def finish_inline_progress(message: str | None = None) -> None:
    if message is None:
        print(flush=True)
    else:
        print(f"\r{message}", flush=True)


def download_file(url: str, destination: Path, timeout: int = 30) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if is_valid_gzip(destination):
            print(f"Using existing file: {destination}")
            return destination
        print(f"Existing file is incomplete or corrupted, re-downloading: {destination}")
        destination.unlink()

    print(f"Downloading: {url}")
    start_time = time.time()
    last_report_time = start_time
    downloaded = 0
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with suppress(FileNotFoundError):
        temp_path.unlink()

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, temp_path.open("wb") as output:
            total_size_header = response.headers.get("Content-Length")
            total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report_time >= 2:
                    elapsed = max(now - start_time, 1e-6)
                    speed = downloaded / elapsed
                    if total_size:
                        progress = downloaded / total_size * 100
                        print_inline_progress(
                            f"  {format_bytes(downloaded)} / {format_bytes(total_size)} "
                            f"({progress:.1f}%), {format_bytes(int(speed))}/s"
                        )
                    else:
                        print_inline_progress(
                            f"  {format_bytes(downloaded)} downloaded, {format_bytes(int(speed))}/s"
                        )
                    last_report_time = now
    except Exception:
        finish_inline_progress()
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise

    os.replace(temp_path, destination)

    if not is_valid_gzip(destination):
        with suppress(FileNotFoundError):
            destination.unlink()
        raise IOError(f"Downloaded file is corrupted: {destination}")

    elapsed = max(time.time() - start_time, 1e-6)
    finish_inline_progress(
        f"Finished: {destination} ({format_bytes(downloaded)} in {elapsed:.1f}s)"
    )
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
                    if i + 1 < n and values_part[i + 1] == "'":
                        token_chars.append("'")
                        i += 2
                        continue
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
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return value.replace("_", " ")


def iter_redirect_pairs(
    page_dump_path: Path,
    redirect_dump_path: Path,
    article_namespace: int = 0,
) -> Iterator[RedirectPair]:
    redirect_targets: dict[int, str] = {}
    redirect_row_count = 0
    redirect_kept_count = 0
    redirect_progress = ProgressLogger("redirect.sql", report_every=500000)

    for row in iter_sql_insert_rows(redirect_dump_path, "redirect"):
        redirect_row_count += 1
        if len(row) < 3:
            redirect_progress.maybe_report(
                redirect_row_count,
                extra=f"kept={redirect_kept_count:,}",
            )
            continue
        page_id = int(row[0])
        namespace = int(row[1])
        target_title = str(row[2])
        if namespace != article_namespace:
            redirect_progress.maybe_report(
                redirect_row_count,
                extra=f"kept={redirect_kept_count:,}",
            )
            continue
        redirect_targets[page_id] = target_title
        redirect_kept_count += 1
        redirect_progress.maybe_report(
            redirect_row_count,
            extra=f"kept={redirect_kept_count:,}",
        )

    redirect_progress.done(
        redirect_row_count,
        extra=f"kept={redirect_kept_count:,}",
    )

    page_row_count = 0
    page_redirect_count = 0
    emitted_pair_count = 0
    page_progress = ProgressLogger("page.sql", report_every=500000)

    for row in iter_sql_insert_rows(page_dump_path, "page"):
        page_row_count += 1
        if len(row) < 5:
            page_progress.maybe_report(
                page_row_count,
                extra=f"redirect_pages={page_redirect_count:,}, emitted={emitted_pair_count:,}",
            )
            continue
        page_id = int(row[0])
        namespace = int(row[1])
        page_title = str(row[2])
        is_redirect = int(row[4])

        if namespace != article_namespace or is_redirect != 1:
            page_progress.maybe_report(
                page_row_count,
                extra=f"redirect_pages={page_redirect_count:,}, emitted={emitted_pair_count:,}",
            )
            continue

        page_redirect_count += 1
        target_title = redirect_targets.get(page_id)
        if target_title is None:
            page_progress.maybe_report(
                page_row_count,
                extra=f"redirect_pages={page_redirect_count:,}, emitted={emitted_pair_count:,}",
            )
            continue

        emitted_pair_count += 1
        page_progress.maybe_report(
            page_row_count,
            extra=f"redirect_pages={page_redirect_count:,}, emitted={emitted_pair_count:,}",
        )
        yield RedirectPair(redirect=page_title, canonical=target_title)

    page_progress.done(
        page_row_count,
        extra=f"redirect_pages={page_redirect_count:,}, emitted={emitted_pair_count:,}",
    )


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
    if not is_valid_gzip(page_dump_path):
        raise IOError(
            f"Corrupted gzip file: {page_dump_path}. Delete it and rerun without --skip-download."
        )
    if not is_valid_gzip(redirect_dump_path):
        raise IOError(
            f"Corrupted gzip file: {redirect_dump_path}. Delete it and rerun without --skip-download."
        )

    print("Building redirect index...")
    stats = build_index(page_dump_path, redirect_dump_path, index_dir, wiki=wiki, dump_tag=dump_tag)
    print(f"Redirect index directory: {index_dir}")
    print(f"Inserted redirect pairs: {stats['inserted']}")
    print(f"Canonical pages: {stats['canonical_pages']}")
    print(f"Redirect entries: {stats['redirects']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# python prepare_wikipedia_redirects.py --wiki enwiki --dump-tag latest
