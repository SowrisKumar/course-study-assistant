"""Small text-cleaning helpers shared by all loaders."""

from __future__ import annotations

import re

_MULTI_BLANK = re.compile(r"\n\s*\n\s*\n+")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_INLINE_WS = re.compile(r"[ \t  ]+")


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying line structure.

    - PowerPoint soft line breaks (vertical tab) become newlines
    - runs of spaces/tabs/non-breaking spaces collapse to one space
    - 3+ blank lines collapse to one blank line
    """
    text = text.replace("\x0b", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def first_line(text: str, max_len: int = 120) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:max_len]
    return ""
