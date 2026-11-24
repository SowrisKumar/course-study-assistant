"""Checks on the shipped knowledge base (data/processed/documents.jsonl)."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from study_assistant.ingest import load_documents


@pytest.fixture(scope="module")
def docs():
    return load_documents()


def test_knowledge_base_is_not_empty(docs):
    assert len(docs) > 400
    assert len({d.source_file for d in docs}) > 40


def test_doc_ids_unique(docs):
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


def test_one_document_per_slide(docs):
    per_deck = Counter(d.source_file for d in docs if d.source_file.endswith(".pptx"))
    for d in docs:
        if d.source_file.endswith(".pptx"):
            assert per_deck[d.source_file] == d.extra["n_slides"], d.source_file


def test_lecture_track_has_lecture_number(docs):
    for d in docs:
        if d.track == "lecture":
            assert d.lecture is not None, d.doc_id


@pytest.mark.parametrize(
    "pattern",
    [r"‹#›", r"(?m)^\s*\d+\s*/\s*\d+\s*$", r"(?m)^\s*[!%]\s*pip\b", r"Version Notice", r"(?m)^MODULE 10$"],
)
def test_no_extraction_noise(docs, pattern):
    hits = [d.doc_id for d in docs if re.search(pattern, d.text)]
    assert not hits, hits[:5]


def test_ocr_deck_has_text(docs):
    rl = [d for d in docs if d.source_file == "module10_rl_slides.pdf"]
    assert rl and all(d.extra["ocr"] for d in rl)
    assert sum(len(d.text) for d in rl) > 10_000
