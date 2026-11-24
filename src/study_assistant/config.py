"""Central configuration: paths and settings (API keys are read from an optional `.env`)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains pyproject.toml.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = PROJECT_ROOT / "data"

# The knowledge base: extracted, cleaned text of the course materials (committed to the repo).
DOCUMENTS_PATH: Path = DATA_DIR / "processed" / "documents.jsonl"

# Caches (not committed).
CACHE_DIR: Path = DATA_DIR / "cache"
OCR_CACHE_DIR: Path = CACHE_DIR / "ocr"
