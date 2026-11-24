"""Central configuration: paths and settings, overridable through a `.env` file.

Every other module imports paths from here, so there is exactly one place to
change where course materials live or where generated files go.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains pyproject.toml (two levels above this file's package).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_str: str) -> Path:
    """Resolve a path from .env; relative paths are taken relative to the project root."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# Where the raw course materials live (slides, notebooks, PDFs). Not committed to git.
COURSE_DIR: Path = _resolve(os.getenv("COURSE_DIR", "../Class Files"))

# Generated artifacts (git-ignored).
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
OCR_CACHE_DIR: Path = CACHE_DIR / "ocr"

DOCUMENTS_PATH: Path = PROCESSED_DIR / "documents.jsonl"
