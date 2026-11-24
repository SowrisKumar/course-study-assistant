from __future__ import annotations

from study_assistant.chunking import MAX_TOKENS, chunk_document, chunk_documents, count_tokens, split_text
from study_assistant.ingest import load_documents
from study_assistant.ingest.schema import Document


def _doc(text: str, **kw) -> Document:
    base = dict(doc_id="lab.ipynb::section-1", text=text, source_file="lab.ipynb", source_type="notebook",
                title="My Lab", section="Training", locator="cells 1-3 (Training)", lecture=5, track="lecture")
    return Document(**{**base, **kw})


def test_short_document_is_one_chunk_with_header():
    chunks = chunk_document(_doc("A short explanation."))
    assert len(chunks) == 1
    assert chunks[0].embed_text == "My Lab › Training\n\nA short explanation."
    assert chunks[0].text == "A short explanation."
    assert chunks[0].chunk_id == "lab.ipynb::section-1#1" and chunks[0].citation == "My Lab, cells 1-3 (Training)"


def test_long_document_is_split_within_budget_and_loses_nothing():
    paragraphs = [f"Paragraph {i} explains step {i} of training the model in some detail. " * 4 for i in range(20)]
    code = "```python\n" + "\n".join(f"x_{i} = layer_{i}(x_{i - 1})" for i in range(60)) + "\n```"
    text = "\n\n".join(paragraphs[:10] + [code] + paragraphs[10:])
    chunks = chunk_document(_doc(text))

    assert len(chunks) > 1
    assert all(count_tokens(c.embed_text) <= MAX_TOKENS for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for p in paragraphs:
        assert p.strip() in joined
    for i in range(60):
        assert f"x_{i} = layer_{i}(x_{i - 1})" in joined
    assert all(c.text.count("```") % 2 == 0 for c in chunks), "a chunk has an unclosed code fence"
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))


def test_code_block_that_fits_is_never_split():
    code = "```python\ndef f(x):\n    return x + 1\n```"
    pieces = split_text("Intro paragraph.\n\n" + code + "\n\nOutro paragraph.", budget=20)
    assert any(code in p for p in pieces)


def test_low_content_documents_are_skipped():
    d1, d2 = _doc("Real content here."), _doc("Divider", doc_id="x::slide-1")
    d2.extra["low_content"] = True
    assert [c.doc_id for c in chunk_documents([d1, d2])] == [d1.doc_id]


def test_knowledge_base_chunks_fit_the_model():
    chunks = chunk_documents(load_documents())
    assert len(chunks) > 1000
    assert max(count_tokens(c.embed_text) for c in chunks) <= MAX_TOKENS
    assert len({c.chunk_id for c in chunks}) == len(chunks)
