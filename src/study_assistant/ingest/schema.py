"""The record every loader produces.

A `Document` is one *natural unit* of course material:
  - one slide of a lecture deck
  - one section (heading + the cells under it) of a notebook
  - one page of a PDF

Chunking for retrieval happens later (Step 2) and may split long units, but it
always keeps this metadata so answers can cite e.g.
"Module 7 – Normalizing Flow Models, slide 5".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

SourceType = Literal["slides", "notebook", "activity", "guide"]


@dataclass
class Document:
    doc_id: str  # stable unique id, e.g. "Module 7 - Normalizing Flow Models.pptx::slide-5"
    text: str  # cleaned text content (what gets embedded / shown to the LLM)
    source_file: str  # original file name
    source_type: SourceType
    title: str  # human-friendly title of the whole file, e.g. "Module 7 – Normalizing Flow Models"
    section: str  # slide title / notebook heading / page heading
    locator: str  # "slide 5" | "cells 3-7" | "page 2"
    lecture: int | None  # lecture module this material belongs to (None = agents/LLM add-on labs)
    track: str  # "lecture" (core modules) or "agents" (agents/LLM add-on labs)
    extra: dict = field(default_factory=dict)  # loader-specific flags (ocr, n_images, has_code, ...)

    @property
    def citation(self) -> str:
        """Short human-readable citation shown next to answers."""
        return f"{self.title}, {self.locator}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        d = {k: v for k, v in d.items() if k != "citation"}
        return cls(**d)
