"""CLI: rebuild data/processed/documents.jsonl from a folder of course files."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from pathlib import Path

from ..config import DOCUMENTS_PATH
from . import load_corpus, save_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest course materials into documents.jsonl")
    parser.add_argument("--source", type=Path, required=True, help="folder with the course files")
    parser.add_argument("--no-ocr", action="store_true", help="skip OCR for image-only PDF pages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs = load_corpus(args.source.expanduser().resolve(), ocr=not args.no_ocr)
    save_documents(docs)

    by_file: dict[str, list] = defaultdict(list)
    for d in docs:
        by_file[d.source_file].append(d)

    print(f"\nSource: {args.source}")
    print(f"{'file':55} {'type':9} {'lec':>3} {'units':>5} {'chars':>7} {'empty':>5}")
    for name, ds in sorted(by_file.items(), key=lambda kv: (kv[1][0].lecture or 99, kv[0])):
        d0 = ds[0]
        empty = sum(len(d.text) < 40 for d in ds)
        print(
            f"{name[:55]:55} {d0.source_type:9} {d0.lecture or '-':>3} {len(ds):5} "
            f"{sum(len(d.text) for d in ds):7} {empty:5}"
        )
    print(f"\n{len(docs)} documents from {len(by_file)} files -> {DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
