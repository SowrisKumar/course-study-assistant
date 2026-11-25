"""Hand-checked verdicts (eval/hand_labels.json), tied to the exact answers they judged.

Each label stores a hash of the answer it was made for. A label is applied only if the answer in the
results still has that hash, so re-running an evaluation (which can produce different answers) never
silently inherits old verdicts: changed answers fall back to the automatic metric and are reported.

Labels can be marked `borderline` (we gave the answer the benefit of the doubt). The *strict reading*
counts borderline "correct" answers as not correct.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import PROJECT_ROOT

HAND_LABELS_PATH = PROJECT_ROOT / "eval" / "hand_labels.json"


def answer_hash(answer: str) -> str:
    return hashlib.sha1(answer.strip().encode("utf-8")).hexdigest()[:12]


def load_hand_labels(path: Path = HAND_LABELS_PATH) -> dict[str, dict[str, dict]]:
    """{run: {question id: {"verdict", "why", "answer_sha1", "borderline"}}}"""
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def lookup(labels: dict, run: str, qid: str, answer: str) -> dict | None:
    """The label for this exact answer, or None (no label, or the answer has changed since labelling)."""
    label = labels.get(run, {}).get(qid)
    if label is None or label["answer_sha1"] != answer_hash(answer):
        return None
    return label


def stale(labels: dict, results) -> list[tuple[str, str]]:
    """(run, question id) pairs whose label no longer matches the saved answer."""
    out = []
    for r in results.itertuples():
        label = labels.get(r.run, {}).get(r.id)
        if label is not None and label["answer_sha1"] != answer_hash(r.answer):
            out.append((r.run, r.id))
    return out


def correct(auto: bool, label: dict | None, *, strict: bool = False) -> bool:
    """Hand-checked correctness: the label's verdict if there is one, otherwise the automatic verdict."""
    if label is None:
        return auto
    return label["verdict"] == "correct" and not (strict and label.get("borderline"))
