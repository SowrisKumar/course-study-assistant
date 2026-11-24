"""PDFs (activities, setup guide, exported slide decks) -> one Document per page.

Text extraction uses PyMuPDF, which keeps word spacing in code blocks
(pypdf turned `import torch.nn as nn` into `importtorch.nnasnn`).

Image-only pages (e.g. `module10_rl_slides.pdf`, an HTML deck exported as
pictures) have no text layer, so we fall back to OCR with RapidOCR
(pip-installable, no system dependencies). OCR results are cached under
data/cache/ocr/ keyed by the file's hash, so it only runs once per file.

Clean-up: lines that repeat on most pages (running headers/footers such as
"Applied Generative AI" or "MODULE 10"), page counters ("1/2", "3/28"),
browser-print URLs and timestamps are removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from ..config import OCR_CACHE_DIR
from .catalog import FileInfo
from .schema import Document
from .text_utils import clean_text

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 30  # pages with less extractable text than this are treated as images
OCR_DPI = 150

_NOISE_LINE = re.compile(
    r"^(\d+\s*/\s*\d+"  # page counters "1/2"
    r"|page\s+\d+(\s+of\s+\d+)?"  # "Page 3 of 4"
    r"|https?://\S+.*"  # URL footer lines (may be followed by a page counter)
    r"|(\d{1,3}\.){3}\d{1,3}(:\d+)?/.*"  # 127.0.0.1:8000/mysite/Module 5/vae_code/ 1/2
    r"|\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M.*"  # browser print timestamp header
    r")$",
    re.IGNORECASE,
)


# --- OCR ----------------------------------------------------------------------

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR  # imported lazily: only needed for image-only PDFs

        _ocr_engine = RapidOCR()
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)  # after init: it configures its own logger
    return _ocr_engine


OcrLine = list  # [x0, y0, x1, y1, text] in pixels at OCR_DPI


def _ocr_page(page: pymupdf.Page) -> list[OcrLine]:
    """Run OCR on one page; return text lines with their bounding boxes."""
    import numpy as np

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result = _get_ocr_engine()(img)
    if result.txts is None:
        return []
    lines = []
    for box, txt in zip(result.boxes, result.txts):
        xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
        lines.append([min(xs), min(ys), max(xs), max(ys), txt])
    return lines


def ocr_lines_to_text(lines: list[OcrLine], page_width: float) -> str:
    """Rebuild reading order from OCR boxes.

    OCR returns lines roughly top-to-bottom, which interleaves side-by-side boxes
    (two callouts at the same height get read across both columns). We therefore
      1. group lines into *blocks*: same left edge (within 2% of page width) and a
         small vertical gap to the previous line of the block;
      2. order blocks in *rows* (blocks starting at about the same height), and
         left-to-right within a row.
    """
    if not lines:
        return ""
    x_tol = 0.02 * page_width
    blocks: list[dict] = []
    for x0, y0, x1, y1, txt in sorted(lines, key=lambda ln: (ln[1], ln[0])):
        h = max(y1 - y0, 1.0)
        best, best_gap = None, None
        for b in blocks:
            gap = y0 - b["y1"]
            if abs(x0 - b["x0"]) <= x_tol and -0.3 * h <= gap <= 1.2 * h:
                if best_gap is None or gap < best_gap:
                    best, best_gap = b, gap
        if best is None:
            blocks.append({"x0": x0, "y0": y0, "y1": y1, "h": h, "lines": [txt]})
        else:
            best["lines"].append(txt)
            best["y1"] = max(best["y1"], y1)

    blocks.sort(key=lambda b: b["y0"])
    ordered: list[dict] = []
    row: list[dict] = []
    for b in blocks:
        if row and b["y0"] - row[0]["y0"] > 0.6 * row[0]["h"]:
            ordered.extend(sorted(row, key=lambda r: r["x0"]))
            row = []
        row.append(b)
    ordered.extend(sorted(row, key=lambda r: r["x0"]))
    return "\n".join("\n".join(b["lines"]) for b in ordered)


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


_OCR_CACHE_VERSION = "v2"  # v2 stores boxes, so layout tweaks never require re-running OCR


def _ocr_pages_cached(path: Path, doc: pymupdf.Document, page_numbers: list[int]) -> dict[int, str]:
    """OCR the given (0-based) pages, using/updating the on-disk cache. Returns page -> text."""
    cache_file = OCR_CACHE_DIR / f"{path.stem}.{_file_hash(path)}.{_OCR_CACHE_VERSION}.json"
    cache: dict[str, list[OcrLine]] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    missing = [n for n in page_numbers if str(n) not in cache]
    if missing:
        log.info("OCR: %s (%d pages, first run only)", path.name, len(missing))
        for n in missing:
            cache[str(n)] = _ocr_page(doc[n])
        OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    width_px = doc[0].rect.width * OCR_DPI / 72  # PDF points -> pixels at OCR_DPI
    return {n: ocr_lines_to_text(cache[str(n)], width_px) for n in page_numbers}


# --- cleaning -----------------------------------------------------------------


_EDGE_LINES = 3  # headers/footers live in the first/last few lines of a page


def _edge_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]


def _remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop running headers/footers: short lines at the top/bottom of most pages.

    Only the first/last few lines of each page are considered, so body text that
    legitimately repeats (e.g. `import torch` in two code listings) is kept.
    """
    if len(pages) < 2:
        return pages
    counts = Counter()
    for text in pages:
        counts.update({ln for ln in _edge_lines(text) if len(ln) <= 60})
    threshold = max(2, int(0.5 * len(pages) + 0.5))
    repeated = {ln for ln, c in counts.items() if c >= threshold}
    if not repeated:
        return pages

    cleaned = []
    for text in pages:
        lines = text.splitlines()
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(nonblank[:_EDGE_LINES] + nonblank[-_EDGE_LINES:])
        cleaned.append(
            "\n".join(ln for i, ln in enumerate(lines) if not (i in edge_idx and ln.strip() in repeated))
        )
    return cleaned


# Course-website PDFs have a site header + sidebar that end up in the text:
#   "Applied Generative AI"                                       (site header line)
#   "Applied Generative AI Table of contents Docker Installation" -> "Docker Installation"
_SITE_HEADER = "Applied Generative AI"
_SIDEBAR_PREFIX = re.compile(r"^.{0,40}?Table of contents\s*", re.IGNORECASE)


def _strip_noise(text: str) -> str:
    lines = (_SIDEBAR_PREFIX.sub("", ln.strip()) for ln in text.splitlines())
    return "\n".join(ln for ln in lines if not _NOISE_LINE.match(ln) and ln != _SITE_HEADER)


_CODE_LIKE = re.compile(r"^(def |class |import |from |return\b|#|@)|[=(){}\[\]]")


def _page_heading(text: str) -> str | None:
    """A heading for the page, or None if the page starts mid-content (a continuation page).

    Skips ALL-CAPS kicker labels like 'MOTIVATION' when a real title follows (OCR deck).
    """
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
    candidates = [ln for ln in lines[:3] if not ln.isupper()] or lines[:1]
    if not candidates:
        return None
    first = candidates[0]
    if _CODE_LIKE.search(first) or len(first) > 100:
        return None
    return first


# --- loader -------------------------------------------------------------------


def load_pdf(path: Path, info: FileInfo, *, ocr: bool = True) -> list[Document]:
    doc = pymupdf.open(str(path))
    # sort=True reads text blocks top-to-bottom, left-to-right, so a heading stays next to
    # the code listing under it (the default order put all code blocks at the end).
    raw = [clean_text(page.get_text(sort=True)) for page in doc]

    image_pages = [i for i, t in enumerate(raw) if len(t.strip()) < MIN_TEXT_CHARS]
    ocr_pages: set[int] = set()
    if image_pages and ocr:
        try:
            for n, text in _ocr_pages_cached(path, doc, image_pages).items():
                raw[n] = text
                ocr_pages.add(n)
        except ImportError:
            log.warning("rapidocr not installed; skipping OCR for %s", path.name)

    pages = [_strip_noise(t) for t in _remove_repeated_lines(raw)]
    unit = "slide" if info.source_type == "slides" else "page"
    title = info.title or path.stem

    docs: list[Document] = []
    prev_section = None
    for i, text in enumerate(pages):
        text = clean_text(text)
        heading = _page_heading(text)
        if heading is None:  # page continues the previous one (e.g. a code listing)
            heading = f"{prev_section} (continued)" if prev_section else f"{unit.title()} {i + 1}"
        else:
            prev_section = heading
        docs.append(
            Document(
                doc_id=f"{path.name}::{unit}-{i + 1}",
                text=text,
                source_file=path.name,
                source_type=info.source_type,
                title=title,
                section=heading,
                locator=f"{unit} {i + 1}",
                lecture=info.lecture,
                track=info.track,
                extra={"ocr": i in ocr_pages, "n_pages": len(pages)},
            )
        )
    return docs
