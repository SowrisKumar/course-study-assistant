"""Hand labels apply only to the exact answers they were made for."""

import json

from study_assistant.config import PROJECT_ROOT
from study_assistant.hand_labels import answer_hash, correct, load_hand_labels, lookup

MAIN = "rag__ollama__llama3.1-8b__hybrid"


def test_every_label_matches_its_saved_answer():
    labels = load_hand_labels()
    for run, by_q in labels.items():
        answers = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(PROJECT_ROOT / "eval" / "results" / f"{run}.jsonl")}
        for qid, label in by_q.items():
            assert label["verdict"] in {"correct", "partial", "wrong"}
            assert lookup(labels, run, qid, answers[qid]) is label, (run, qid)


def test_changed_answer_does_not_inherit_the_label():
    labels = {MAIN: {"q29": {"verdict": "correct", "why": "", "answer_sha1": answer_hash("old answer"), "borderline": True}}}
    assert lookup(labels, MAIN, "q29", "old answer  ") is not None  # whitespace-insensitive
    assert lookup(labels, MAIN, "q29", "a different answer") is None


def test_strict_reading_counts_borderline_as_not_correct():
    borderline = {"verdict": "correct", "borderline": True}
    assert correct(False, borderline) and not correct(False, borderline, strict=True)
    assert correct(True, None) and not correct(False, None)  # no label: automatic verdict
    assert not correct(True, {"verdict": "partial"})
