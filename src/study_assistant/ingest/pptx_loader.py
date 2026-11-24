"""Lecture slides (.pptx) -> one Document per slide.

What we keep:   slide title, every text box (including ones inside grouped shapes),
                tables (rows joined with " | "), and speaker notes.
What we drop:   slide-number / date / footer placeholders (e.g. the '‹#›' field).
Known gap:      equations and diagrams that are *images* carry no text, so we
                record `n_images` to flag slides whose meaning lives in pictures.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from .catalog import FileInfo
from .schema import Document
from .text_utils import clean_text, first_line

_SKIP_PLACEHOLDERS = {PP_PLACEHOLDER.SLIDE_NUMBER, PP_PLACEHOLDER.DATE, PP_PLACEHOLDER.FOOTER}
_SLIDE_NUMBER_FIELD = "‹#›"


def _iter_shapes(shapes):
    """Yield every shape, descending into groups."""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _is_skipped_placeholder(shape) -> bool:
    if not shape.is_placeholder:
        return False
    try:
        return shape.placeholder_format.type in _SKIP_PLACEHOLDERS
    except ValueError:  # some shapes report is_placeholder but have no format
        return False


def _table_text(table) -> str:
    rows = []
    for row in table.rows:
        cells = [clean_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _slide_content(slide) -> tuple[str, list[str], int]:
    """Return (title, body_blocks, n_images) for one slide."""
    title_shape = slide.shapes.title
    title_full = clean_text(title_shape.text_frame.text) if title_shape is not None else ""

    blocks: list[str] = []
    # Some decks put body text inside the title placeholder; keep everything after the first line.
    title = first_line(title_full)
    title_rest = title_full[len(title):].strip() if title_full.startswith(title) else ""
    if title_rest:
        blocks.append(title_rest)

    n_images = 0
    for shape in _iter_shapes(slide.shapes):
        if title_shape is not None and shape.shape_id == title_shape.shape_id:
            continue
        if _is_skipped_placeholder(shape):
            continue
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            n_images += 1
            continue
        if getattr(shape, "has_table", False) and shape.has_table:
            text = _table_text(shape.table)
        elif shape.has_text_frame:
            text = clean_text(shape.text_frame.text)
        else:
            continue
        text = text.replace(_SLIDE_NUMBER_FIELD, "").strip()
        if text:
            blocks.append(text)

    if not title and blocks:  # no title placeholder: use first line of the first text block
        title = first_line(blocks[0])
    return title, blocks, n_images


def load_pptx(path: Path, info: FileInfo) -> list[Document]:
    prs = Presentation(str(path))
    deck_title = info.title or path.stem
    docs: list[Document] = []

    for number, slide in enumerate(prs.slides, start=1):
        title, blocks, n_images = _slide_content(slide)

        notes = ""
        if slide.has_notes_slide:
            notes = clean_text(slide.notes_slide.notes_text_frame.text)

        parts = [title] if title else []
        parts += blocks
        if notes:
            parts.append(f"Speaker notes: {notes}")
        text = "\n\n".join(dict.fromkeys(p for p in parts if p))  # dedupe identical blocks, keep order

        docs.append(
            Document(
                doc_id=f"{path.name}::slide-{number}",
                text=text,
                source_file=path.name,
                source_type=info.source_type,
                title=deck_title,
                section=title or f"Slide {number}",
                locator=f"slide {number}",
                lecture=info.lecture,
                track=info.track,
                extra={"n_images": n_images, "has_notes": bool(notes), "n_slides": len(prs.slides)},
            )
        )
    return docs
