"""Jupyter notebooks (.ipynb) -> one Document per section.

A new *section* starts at
  - a markdown cell that begins with a heading (`#`, `##`, ...), or
  - a markdown cell that follows a code cell (a new "explanation -> code" step).
So each section is a natural teaching unit: some explanation plus the code that
implements it. Splitting only on headings made some notebooks collapse into
2-3 sections of 20k+ characters (e.g. `llm_as_judge_solution.ipynb`).
The section label is the heading breadcrumb (e.g. "Training › Loss function"),
plus the first line of the explanation when the step has no heading of its own.

What we keep:   markdown text and code (code is course content too: "how did we
                implement the VAE loss?").
What we drop:   cell outputs (mostly training logs / tables of numbers), pip-install
                lines, and the boilerplate "Static/Jupyter Version Notice" cells
                added by the marimo export.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import FileInfo, clean_stem
from .schema import Document
from .text_utils import clean_text

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NOTICE = re.compile(r"\*\*(Static|Jupyter) Version Notice\*\*")
_INSTALL_LINE = re.compile(r"^\s*#?\s*[!%]\s*(pip|uv|conda|apt)\b|^\s*%%capture")
_MD_MARKUP = re.compile(r"[*_`]+")


@dataclass
class _Section:
    heading_path: list[str]
    first_cell: int  # 1-based
    last_cell: int
    parts: list[str] = field(default_factory=list)
    has_code: bool = False


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _heading_of(markdown: str) -> tuple[int, str] | None:
    """(level, text) if the markdown cell *starts* with a heading."""
    for line in markdown.splitlines():
        if not line.strip():
            continue
        m = _HEADING.match(line.strip())
        if m:
            return len(m.group(1)), _MD_MARKUP.sub("", m.group(2)).strip()
        return None
    return None


def _short_label(markdown: str, max_len: int = 60) -> str:
    """First line of a markdown cell, markup stripped and truncated, for use as a section label."""
    for line in markdown.splitlines():
        line = _MD_MARKUP.sub("", line).strip(" #>-:")
        if line:
            return line if len(line) <= max_len else line[: max_len - 1].rstrip() + "…"
    return "…"


def _clean_code(code: str) -> str:
    lines = [ln for ln in code.splitlines() if not _INSTALL_LINE.match(ln)]
    return "\n".join(lines).strip()


def _notebook_title(cells: list[dict], fallback: str) -> str:
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        for line in _cell_source(cell).splitlines():
            m = _HEADING.match(line.strip())
            if m and len(m.group(1)) == 1:
                return _MD_MARKUP.sub("", m.group(2)).strip()
    return fallback


def load_notebook(path: Path, info: FileInfo) -> list[Document]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") in ("markdown", "code")]
    nb_title = info.title or _notebook_title(cells, fallback=clean_stem(path.name))

    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text) for breadcrumb
    prev_was_code = False

    def current() -> _Section:
        return sections[-1]

    for idx, cell in enumerate(cells, start=1):
        src = _cell_source(cell)
        if not src.strip():
            continue

        if cell["cell_type"] == "markdown":
            if _NOTICE.search(src):
                continue
            heading = _heading_of(src)
            if heading:
                level, text = heading
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level] + [(level, text)]
            if heading is not None or prev_was_code or not sections:
                # Breadcrumb without the notebook's own H1 title (already in `title`).
                path_names = [t for lv, t in heading_stack if not (lv == 1 and t == nb_title)]
                if heading is None:  # a step without its own heading: label it by its first line
                    path_names = path_names + [_short_label(src)]
                sections.append(_Section(path_names, idx, idx))
            current().parts.append(clean_text(src))
            current().last_cell = idx
            prev_was_code = False
        else:  # code
            code = _clean_code(src)
            if not code:
                continue
            if not sections:
                sections.append(_Section([], idx, idx))
            current().parts.append(f"```python\n{code}\n```")
            current().has_code = True
            current().last_cell = idx
            prev_was_code = True

    docs: list[Document] = []
    for n, sec in enumerate(sections, start=1):
        text = "\n\n".join(p for p in sec.parts if p).strip()
        if not text:
            continue
        section_name = " › ".join(sec.heading_path) or "Introduction"
        locator = (
            f"cell {sec.first_cell}"
            if sec.first_cell == sec.last_cell
            else f"cells {sec.first_cell}-{sec.last_cell}"
        )
        docs.append(
            Document(
                doc_id=f"{path.name}::section-{n}",
                text=text,
                source_file=path.name,
                source_type="notebook",
                title=nb_title,
                section=section_name,
                locator=f"{locator} ({section_name})",
                lecture=info.lecture,
                track=info.track,
                extra={"has_code": sec.has_code},
            )
        )
    return docs
