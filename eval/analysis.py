"""Hand-corrected scores, 95% confidence intervals and paired significance tests for the saved runs.

    uv run python eval/analysis.py

Reads eval/results/*.jsonl (written by scripts/run_evaluation.sh) and eval/hand_labels.json.
With 38 answerable and 12 not-covered questions every rate is uncertain by several points, so we
report Wilson 95% intervals, and compare systems with an exact McNemar test (both systems answered
the same questions, so only the questions where they disagree carry information).
"""

from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
QUESTIONS = {json.loads(line)["id"]: json.loads(line) for line in open(HERE / "questions.jsonl") if line.strip()}
HAND = {k: v for k, v in json.loads((HERE / "hand_labels.json").read_text()).items() if not k.startswith("_")}

LABELS = {
    "rag__ollama__llama3.1-8b__hybrid": "Llama + course, hybrid (main)",
    "rag__ollama__llama3.1-8b__dense": "Llama + course, dense",
    "rag__ollama__llama3.1-8b__hybrid__nograde": "relevance check off",
    "rag__ollama__llama3.1-8b__hybrid__verify": "support check: strict",
    "rag__ollama__llama3.1-8b__hybrid__verify-concept": "support check: concept",
    "no_rag__ollama__llama3.1-8b": "Llama, no course",
    "rag__flan__google-flan-t5-base__hybrid": "flan-t5 + course",
    "no_rag__flan__google-flan-t5-base": "flan-t5, no course",
}
MAIN = "rag__ollama__llama3.1-8b__hybrid"
COMPARE = [  # (a, b): is a different from b?
    (MAIN, "no_rag__ollama__llama3.1-8b"),
    (MAIN, "rag__ollama__llama3.1-8b__dense"),
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__nograde"),
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__verify"),
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__verify-concept"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """(a right & b wrong, a wrong & b right, two-sided exact p)."""
    only_a = sum(a[i] and not b[i] for i in a)
    only_b = sum(b[i] and not a[i] for i in a)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(comb(n, j) for j in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)


def auto_correct(rec: dict) -> bool:
    """The main metric: judge v2 = 2 and key-fact recall >= 0.5."""
    return not rec["abstained"] and rec.get("judge2_score") == 2 and (rec.get("key_fact_recall") or 0) >= 0.5


def outcomes(run: str) -> dict[str, dict[str, bool]] | None:
    path = RESULTS / f"{run}.jsonl"
    if not path.exists():
        return None
    recs = {json.loads(line)["id"]: json.loads(line) for line in open(path) if line.strip()}
    hand = HAND.get(run, {})
    out = {"auto": {}, "hand": {}, "abstain": {}}
    for qid, rec in recs.items():
        if QUESTIONS[qid]["type"] == "out_of_scope":
            out["abstain"][qid] = rec["abstained"]
        else:
            out["auto"][qid] = auto_correct(rec)
            out["hand"][qid] = hand[qid][0] == "correct" if qid in hand else out["auto"][qid]
    return out


def pct(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def main() -> None:
    data = {run: o for run in LABELS if (o := outcomes(run)) is not None}
    print("Answerable questions correct (auto = main metric; hand = after re-reading, eval/hand_labels.json),")
    print("and not-covered questions abstained. 95% Wilson intervals in brackets.\n")
    print(f"| configuration | correct (auto) | correct (hand) | not covered → abstained | all 50 handled (hand) |")
    print("|---|---|---|---|---|")
    for run, o in data.items():
        a, h, ab = o["auto"], o["hand"], o["abstain"]
        handled = sum(h.values()) + sum(ab.values())
        hand_cell = pct(sum(h.values()), len(h)) if run in HAND else "not re-checked"
        all_cell = pct(handled, len(h) + len(ab)) if run in HAND else "—"
        print(f"| {LABELS[run]} | {pct(sum(a.values()), len(a))} | {hand_cell} | {pct(sum(ab.values()), len(ab))} | {all_cell} |")

    print("\nPaired comparisons with the main system (exact McNemar test, hand labels; all 50 questions:")
    print("an answerable question counts if answered correctly, a not-covered one if the system abstained).\n")
    print("| main vs | main right, other wrong | main wrong, other right | p |")
    print("|---|---|---|---|")
    for a_run, b_run in COMPARE:
        if a_run not in data or b_run not in data or b_run not in HAND:
            continue
        a = {**data[a_run]["hand"], **data[a_run]["abstain"]}
        b = {**data[b_run]["hand"], **data[b_run]["abstain"]}
        x, y, p = mcnemar_exact(a, b)
        print(f"| {LABELS[b_run]} | {x} | {y} | {p:.3g} |")


if __name__ == "__main__":
    main()
