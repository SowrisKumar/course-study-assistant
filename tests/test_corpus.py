"""Checks against the real course materials. Skipped automatically if COURSE_DIR is missing."""

from __future__ import annotations

import re
from collections import Counter

import pytest
from pptx import Presentation

from study_assistant.config import COURSE_DIR
from study_assistant.ingest import load_corpus
from study_assistant.ingest.catalog import discover_files

pytestmark = pytest.mark.skipif(not COURSE_DIR.exists(), reason=f"COURSE_DIR not found: {COURSE_DIR}")


@pytest.fixture(scope="module")
def docs():
    return load_corpus(COURSE_DIR)


def test_every_course_file_produces_documents(docs):
    files = {p.name for p in discover_files(COURSE_DIR)}
    loaded = {d.source_file for d in docs}
    assert files == loaded, f"files without documents: {files - loaded}"


def test_one_document_per_slide(docs):
    per_file = Counter(d.source_file for d in docs)
    for path in discover_files(COURSE_DIR):
        if path.suffix == ".pptx":
            assert per_file[path.name] == len(Presentation(str(path)).slides), path.name


def test_doc_ids_unique(docs):
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


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
    if not rl:
        pytest.skip("extended RL deck not present")
    assert all(d.extra["ocr"] for d in rl)
    assert sum(len(d.text) for d in rl) > 10_000
