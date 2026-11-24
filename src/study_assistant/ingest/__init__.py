"""Step 1 – Ingestion: turn the raw course folder into a list of `Document`s.

Usage:
    uv run python -m study_assistant.ingest            # build data/processed/documents.jsonl
    uv run python -m study_assistant.ingest --no-ocr   # skip OCR of image-only PDF pages
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import COURSE_DIR, DOCUMENTS_PATH
from .catalog import classify, discover_files
from .notebook_loader import load_notebook
from .pdf_loader import load_pdf
from .pptx_loader import load_pptx
from .schema import Document

__all__ = ["Document", "load_corpus", "load_file", "save_documents", "load_documents"]

log = logging.getLogger(__name__)


def load_file(path: Path, *, ocr: bool = True) -> list[Document]:
    info = classify(path)
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        return load_pptx(path, info)
    if suffix == ".ipynb":
        return load_notebook(path, info)
    if suffix == ".pdf":
        return load_pdf(path, info, ocr=ocr)
    raise ValueError(f"Unsupported file: {path}")


def load_corpus(course_dir: Path = COURSE_DIR, *, ocr: bool = True) -> list[Document]:
    """Load every supported course file. A file that fails to parse is logged and skipped."""
    if not course_dir.exists():
        raise FileNotFoundError(
            f"COURSE_DIR not found: {course_dir}\n"
            "Set COURSE_DIR in .env to your copy of the class files."
        )
    docs: list[Document] = []
    for path in discover_files(course_dir):
        try:
            loaded = load_file(path, ocr=ocr)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the whole corpus
            log.error("Failed to load %s: %s", path.name, exc)
            continue
        for d in loaded:
            if not d.text.strip():  # e.g. a PDF page that only had a header/footer
                continue
            d.extra["low_content"] = _is_low_content(d)
            docs.append(d)
    return docs


LOW_CONTENT_CHARS = 40


def _is_low_content(doc: Document) -> bool:
    """True for units with (almost) nothing beyond their title, e.g. section-divider slides.

    They stay in documents.jsonl for transparency, but the indexer (Step 2) skips them:
    a slide that only says "Variational Autoencoder" would otherwise out-rank the
    slides that actually explain VAEs for the query "what is a VAE?".
    """
    body = doc.text.replace(doc.section, "", 1)
    body = body.replace("```python", "").replace("```", "").strip("#\n ")
    return len(body) < LOW_CONTENT_CHARS


def save_documents(docs: list[Document], path: Path = DOCUMENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def load_documents(path: Path = DOCUMENTS_PATH) -> list[Document]:
    with path.open(encoding="utf-8") as fh:
        return [Document.from_dict(json.loads(line)) for line in fh if line.strip()]
