"""Unit tests for the loaders, using small synthetic files (no course materials needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from pptx import Presentation
from pptx.util import Inches

from study_assistant.ingest import load_file
from study_assistant.ingest.catalog import classify, clean_stem
from study_assistant.ingest.pdf_loader import ocr_lines_to_text
from study_assistant.ingest.text_utils import clean_text

# --- text utils / catalog -------------------------------------------------------


def test_clean_text_normalises_whitespace():
    raw = "Title\x0bsecond line  with spaces \n\n\n\n  next para  "
    assert clean_text(raw) == "Title\nsecond line with spaces\n\nnext para"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Module 4 - Modern Deep Learning Architectures Part 1.pptx", "Module 4 – Modern Deep Learning Architectures Part 1"),
        ("finetuning_text_model (1) (2).ipynb", "finetuning text model"),
        ("Module 1 Custom Setup Guide-3.pdf", "Module 1 Custom Setup Guide"),
        ("Module_6_Practical_1_GAN_(1).ipynb", "Module 6 Practical 1 GAN"),
    ],
)
def test_clean_stem(name, expected):
    assert clean_stem(name) == expected


def test_classify_maps_notebooks_to_their_lecture_not_their_filename_number():
    info = classify(Path("Module_4_Practical_VAE.ipynb"))
    assert (info.source_type, info.lecture, info.track) == ("notebook", 5, "lecture")


def test_classify_unknown_notebook_is_addon_lab():
    info = classify(Path("some_new_lab.ipynb"))
    assert info.lecture is None and info.track == "add-on lab"


def test_classify_pdfs():
    assert classify(Path("Module 6 Activity - Adding GAN to Helper Library.pdf")).source_type == "activity"
    assert classify(Path("module10_rl_slides.pdf")).source_type == "slides"


# --- notebooks ------------------------------------------------------------------


def _md(src):
    return {"cell_type": "markdown", "source": src, "metadata": {}}


def _code(src):
    return {"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None}


def test_notebook_sections(tmp_path):
    nb = {
        "cells": [
            _md("⚠️ **Static Version Notice**\n\nexported from marimo"),  # dropped
            _md("# My Lab Title\n\nIntro text."),
            _code("!pip install torch\nimport torch"),  # pip line dropped
            _md("Now we define the model."),  # markdown after code -> new section
            _code("model = torch.nn.Linear(2, 2)"),
            _md("## Training\n\nWe train it."),
            _md("Some more training notes."),  # markdown after markdown -> same section
            _code("loss.backward()"),
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "my_lab.ipynb"
    path.write_text(json.dumps(nb))
    docs = load_file(path)

    assert [d.title for d in docs] == ["My Lab Title"] * 3
    assert [d.section for d in docs] == ["Introduction", "Now we define the model.", "Training"]
    assert "Version Notice" not in " ".join(d.text for d in docs)
    assert "pip install" not in docs[0].text and "import torch" in docs[0].text
    assert docs[2].locator == "cells 6-8 (Training)"
    assert "Some more training notes." in docs[2].text and "loss.backward()" in docs[2].text


# --- slides -----------------------------------------------------------------------


def test_pptx_slide_text_table_and_notes(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # "Title Only"
    slide.shapes.title.text = "Attention"
    box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    box.text_frame.text = "Queries, keys and values."
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "Q", "K"
    table.cell(1, 0).text, table.cell(1, 1).text = "1", "2"
    slide.notes_slide.notes_text_frame.text = "Mention scaling by sqrt(d)."
    prs.slides.add_slide(prs.slide_layouts[5]).shapes.title.text = "Divider"

    path = tmp_path / "Module 9 - Test Deck.pptx"
    prs.save(path)
    docs = load_file(path)

    assert len(docs) == 2
    d = docs[0]
    assert d.section == "Attention" and d.locator == "slide 1" and d.lecture == 9
    assert "Queries, keys and values." in d.text
    assert "Q | K" in d.text and "1 | 2" in d.text
    assert "Speaker notes: Mention scaling by sqrt(d)." in d.text
    assert d.citation == "Module 9 – Test Deck, slide 1"


# --- PDFs ---------------------------------------------------------------------------


def test_pdf_removes_headers_footers_and_marks_continuations(tmp_path):
    pdf = pymupdf.open()
    bodies = [
        "GAN Class Activity\nIn model.py define the generator.",
        "def train_gan(model):\n    return model",
        "Usage Example\nCall train_gan on your data.",
    ]
    for i, body in enumerate(bodies, start=1):
        page = pdf.new_page()
        page.insert_text((72, 40), "Applied Generative AI")  # running header
        page.insert_text((72, 100), body)
        page.insert_text((72, 780), "127.0.0.1:8000/mysite/Module 6/gan_code/")  # URL footer
        page.insert_text((500, 780), f"{i}/3")  # page counter
    path = tmp_path / "Module 6 Activity - Test.pdf"
    pdf.save(path)

    docs = load_file(path, ocr=False)
    assert len(docs) == 3
    for d in docs:
        assert "Applied Generative AI" not in d.text
        assert "127.0.0.1" not in d.text
        assert "/3" not in d.text
    assert docs[0].section == "GAN Class Activity"
    assert docs[1].section == "GAN Class Activity (continued)"  # page starts with code
    assert docs[2].section == "Usage Example"
    assert docs[0].source_type == "activity" and docs[0].locator == "page 1"


def test_ocr_layout_reads_side_by_side_boxes_separately():
    W = 1000
    lines = [
        [50, 10, 600, 30, "Slide title"],
        [50, 100, 400, 120, "Left box heading"],
        [550, 100, 900, 120, "Right box heading"],
        [50, 125, 400, 145, "left line two"],
        [550, 125, 900, 145, "right line two"],
    ]
    text = ocr_lines_to_text(lines, page_width=W)
    assert text.splitlines() == [
        "Slide title",
        "Left box heading",
        "left line two",
        "Right box heading",
        "right line two",
    ]
