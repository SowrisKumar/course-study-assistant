"""Step 2a – Chunking: split documents into pieces the embedding model can read in full.

Why: `all-MiniLM-L6-v2` reads at most 256 tokens; anything after that is silently
ignored. Slides fit (median ~105 tokens), but 185 of 249 notebook sections do not
(some are 4,800 tokens).

How:
  - Documents flagged `low_content` (title / divider slides) are skipped.
  - A document that fits becomes one chunk.
  - Longer ones are split on paragraph and code-block boundaries, packed greedily up
    to the token budget, so we never cut mid-sentence or mid-function. A paragraph
    that is itself too long is split by lines, then by words.
  - Consecutive chunks share one short paragraph (overlap), so an idea that spans a
    boundary appears whole in at least one chunk.
  - Every chunk gets a header "title › section" in the text that is *embedded*.
    A piece of training code from the middle of a notebook means much more to the
    retriever when it is labelled "Variational Autoencoders › Training".

Token counts use the embedding model's own tokenizer, so "fits" is exact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .ingest.schema import Document

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned Hugging Face revision, so a fresh clone gets exactly the model all results were produced with.
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_TOKENS = 254  # 256 minus the [CLS] and [SEP] tokens the model adds
OVERLAP_MAX_TOKENS = 60  # carry the previous chunk's last paragraph if it is at most this long


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id>#<n>"
    doc_id: str
    text: str  # chunk body (shown to the LLM)
    embed_text: str  # header + body (what retrievers index)
    citation: str
    title: str
    section: str
    locator: str
    source_file: str
    source_type: str
    lecture: int | None
    track: str
    part: int  # 1-based index of this chunk within its document
    n_parts: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    tok.model_max_length = 10**9  # we only count tokens; silences the "too long" warning
    return tok


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def header_for(doc: Document) -> str:
    section = doc.section if doc.section and doc.section != doc.title else ""
    return f"{doc.title} › {section}" if section else doc.title


# --- splitting --------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, but keep each fenced code block as a single unit."""
    parts: list[str] = []
    pos = 0
    for m in _CODE_FENCE.finditer(text):
        parts += [p for p in re.split(r"\n\s*\n", text[pos : m.start()]) if p.strip()]
        parts.append(m.group(0))
        pos = m.end()
    parts += [p for p in re.split(r"\n\s*\n", text[pos:]) if p.strip()]
    return [p.strip() for p in parts]


def _split_oversized(par: str, budget: int) -> list[str]:
    """Split one paragraph that exceeds the budget: by lines first, then by words."""
    is_code = par.startswith("```")
    inner = par.strip("`").removeprefix("python").strip("\n") if is_code else par
    units = inner.split("\n")
    if len(units) == 1:  # a single huge line: fall back to words
        units = inner.split(" ")
    sep = "\n" if len(inner.split("\n")) > 1 else " "
    wrap_cost = count_tokens("```python\n\n```") if is_code else 0

    pieces, current = [], []
    for unit in units:
        candidate = sep.join(current + [unit])
        if current and count_tokens(candidate) + wrap_cost > budget:
            pieces.append(sep.join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(sep.join(current))

    out = []
    for p in pieces:
        if count_tokens(p) + wrap_cost > budget:  # one line still too long: hard word split
            words, cur = p.split(" "), []
            for w in words:
                if cur and count_tokens(" ".join(cur + [w])) + wrap_cost > budget:
                    out.append(" ".join(cur))
                    cur = [w]
                else:
                    cur.append(w)
            if cur:
                out.append(" ".join(cur))
        else:
            out.append(p)
    return [f"```python\n{p}\n```" if is_code else p for p in out]


def split_text(text: str, budget: int) -> list[str]:
    """Greedy paragraph packing into pieces of at most `budget` tokens, with one-paragraph overlap."""
    paras: list[str] = []
    for p in _paragraphs(text):
        paras += _split_oversized(p, budget) if count_tokens(p) > budget else [p]

    chunks: list[list[str]] = []
    current: list[str] = []
    for p in paras:
        if current and count_tokens("\n\n".join(current + [p])) > budget:
            chunks.append(current)
            last = current[-1]
            overlap_ok = (
                count_tokens(last) <= OVERLAP_MAX_TOKENS
                and count_tokens(last + "\n\n" + p) <= budget
            )
            current = [last, p] if overlap_ok else [p]
        else:
            current.append(p)
    if current:
        chunks.append(current)
    return ["\n\n".join(c) for c in chunks]


# --- public API ---------------------------------------------------------------------


def chunk_document(doc: Document, max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    header = header_for(doc)
    budget = max_tokens - count_tokens(header + "\n\n")
    bodies = [doc.text] if count_tokens(doc.text) <= budget else split_text(doc.text, budget)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            text=body,
            embed_text=f"{header}\n\n{body}",
            citation=doc.citation,
            title=doc.title,
            section=doc.section,
            locator=doc.locator,
            source_file=doc.source_file,
            source_type=doc.source_type,
            lecture=doc.lecture,
            track=doc.track,
            part=i,
            n_parts=len(bodies),
            extra=dict(doc.extra),
        )
        for i, body in enumerate(bodies, start=1)
    ]


def chunk_documents(docs: list[Document], max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        if d.extra.get("low_content"):
            continue
        chunks.extend(chunk_document(d, max_tokens))
    return chunks
