"""One shared decision: did the model decline to answer? Used by the assistant and the evaluation.

A reply is a refusal if it
  - is empty, or a bare "unanswerable" / "n/a" (flan-t5's convention from its SQuAD-v2 style training data),
  - STARTS with the not-covered sentence or "I don't know", or
  - CONTAINS the not-covered sentence and cites nothing. Llama sometimes writes a general-knowledge hedge
    ("...not explicitly mentioned. However, typically...") and only then the sentence.
A cited answer that merely ends with a disclaimer ("...[1]. The rest isn't covered in the course
materials.") is NOT a refusal: the student still gets the supported part.
"""

from __future__ import annotations

import re

from .prompts import NOT_COVERED

_SENTENCE = re.sub(r"[^a-z ]", "", NOT_COVERED.lower()).strip()  # "this isnt covered in the course materials"
_COVERED = re.compile(r"(isn'?t|is not|not) covered in the course")
_BARE = re.compile(r"^(unanswerable|no answer|n/?a)\.?$")
_CITATION = re.compile(r"\[\d+\]")


def _norm(text: str) -> str:
    t = text.replace("’", "'").strip().strip('"').strip()
    return " ".join(t.lower().split())


def is_refusal(text: str) -> bool:
    t = _norm(text)
    if not t or _BARE.match(t):
        return True
    head = re.sub(r"[^a-z ]", "", t[:80].replace("'", ""))
    if head.startswith(_SENTENCE[:30]) or head.startswith("i dont know"):
        return True
    return bool(_COVERED.search(t)) and not _CITATION.search(text)
