"""Hand-corrected scores, 95% confidence intervals and paired significance tests for the saved runs.

    uv run python eval/analysis.py                          # the run reported in docs/report.md
    uv run python eval/analysis.py --results eval/runs/X    # any other run (e.g. a fresh reproduction)

Reads the result files and eval/hand_labels.json. A hand label is applied only to the exact answer it was
made for (answer hash), so for a fresh run, changed answers fall back to the automatic metric (reported).
"hand" counts borderline labels as correct; "strict" counts them as not correct.
With 38 answerable and 12 not-covered questions every rate is uncertain by several points, so we
report Wilson 95% intervals, and compare systems with an exact McNemar test (both systems answered
the same questions, so only the questions where they disagree carry information).
"""

from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path

from study_assistant.hand_labels import answer_hash, correct, load_hand_labels, lookup

HERE = Path(__file__).parent
RESULTS = HERE / "results"
QUESTIONS = {json.loads(line)["id"]: json.loads(line) for line in open(HERE / "questions.jsonl") if line.strip()}
HAND = load_hand_labels(HERE / "hand_labels.json")

LABELS = {
    "rag__ollama__llama3.1-8b__hybrid": "Llama + course, hybrid (main)",
    "rag__ollama__llama3.1-8b__dense": "Llama + course, dense",
    "rag__ollama__llama3.1-8b__hybrid__nograde": "relevance check off",
    "rag__ollama__llama3.1-8b__hybrid__verify": "support check: strict",
    "rag__ollama__llama3.1-8b__hybrid__verify-concept": "support check: concept",
    "rag__ollama__llama3.1-8b__hybrid__nofloor": "similarity floor off",
    "rag__ollama__llama3.1-8b__hybrid__ctx-small": "less context (3 passages)",
    "rag__ollama__llama3.1-8b__hybrid__ctx-large": "more context (8 passages)",
    "no_rag__ollama__llama3.1-8b": "Llama, no course",
    "no_rag__ollama__llama3.1-8b__syllabus": "Llama, no course, sees the syllabus",
    "no_rag__ollama__llama3.1-8b__syllabus-lenient": "…syllabus, lenient wording",
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
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__nofloor"),
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__ctx-small"),
    (MAIN, "rag__ollama__llama3.1-8b__hybrid__ctx-large"),
    (MAIN, "no_rag__ollama__llama3.1-8b__syllabus"),
    (MAIN, "no_rag__ollama__llama3.1-8b__syllabus-lenient"),
]

# Course-specific questions (audit idea C2): the expected answer is a fact about THIS course's own material
# (a particular study, the course's tool choice, a toy example, how a lab is structured), so general ML
# knowledge cannot produce it. q09, q29 and q38 mention the course but general knowledge answers them.
# Classified after the results were known, so the audit's own list (which adds q38) is reported as a check.
COURSE_SPECIFIC = {"q01", "q03", "q34", "q35", "q37"}
COURSE_SPECIFIC_AUDIT = COURSE_SPECIFIC | {"q38"}
SPLIT_RUNS = [MAIN, "rag__ollama__llama3.1-8b__dense", "no_rag__ollama__llama3.1-8b", "no_rag__ollama__llama3.1-8b__syllabus",
              "no_rag__ollama__llama3.1-8b__syllabus-lenient"]


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


def outcomes(run: str, results: Path = RESULTS) -> dict | None:
    path = results / f"{run}.jsonl"
    if not path.exists():
        return None
    recs = {json.loads(line)["id"]: json.loads(line) for line in open(path) if line.strip()}
    out = {"auto": {}, "hand": {}, "strict": {}, "abstain": {}, "stale": []}
    for qid, rec in recs.items():
        if QUESTIONS[qid]["type"] == "out_of_scope":
            out["abstain"][qid] = rec["abstained"]
            continue
        auto = auto_correct(rec)
        label = lookup(HAND, run, qid, rec["answer"])
        if label is None and qid in HAND.get(run, {}):
            out["stale"].append(qid)  # labelled, but this answer differs from the one we judged
        out["auto"][qid] = auto
        out["hand"][qid] = correct(auto, label)
        out["strict"][qid] = correct(auto, label, strict=True)
    return out


def pct(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=RESULTS, help="folder with the run's result files")
    results = p.parse_args(argv).results
    data = {run: o for run in LABELS if (o := outcomes(run, results)) is not None}
    shown = results.resolve().relative_to(HERE.parent.resolve()) if results.resolve().is_relative_to(HERE.parent.resolve()) else results.name
    print(f"Results: {shown}\n")
    print("Answerable questions correct (auto = main metric; hand = after re-reading, eval/hand_labels.json;")
    print("strict = hand, but borderline answers counted as not correct), and not-covered questions abstained.")
    print("95% Wilson intervals in brackets.\n")
    print("| configuration | correct (auto) | correct (hand) | correct (strict) | not covered → abstained | all 50 handled (hand) |")
    print("|---|---|---|---|---|---|")
    for run, o in data.items():
        a, h, st, ab = o["auto"], o["hand"], o["strict"], o["abstain"]
        checked = run in HAND
        hand_cell = pct(sum(h.values()), len(h)) if checked else "not re-checked"
        strict_cell = pct(sum(st.values()), len(st)) if checked else "—"
        all_cell = pct(sum(h.values()) + sum(ab.values()), len(h) + len(ab)) if checked else "—"
        print(f"| {LABELS[run]} | {pct(sum(a.values()), len(a))} | {hand_cell} | {strict_cell} "
              f"| {pct(sum(ab.values()), len(ab))} | {all_cell} |")
    stale = {LABELS[r]: o["stale"] for r, o in data.items() if o["stale"]}
    if stale:
        print(f"\nWARNING: answers changed since hand-labelling, automatic verdict used instead: {stale}")

    print("\nPaired comparisons with the main system (exact McNemar test, hand labels; all 50 questions:")
    print("an answerable question counts if answered correctly, a not-covered one if the system abstained).\n")
    print("| main vs | main right, other wrong | main wrong, other right | p | strict reading: wins / losses, p |")
    print("|---|---|---|---|---|")
    for a_run, b_run in COMPARE:
        if a_run not in data or b_run not in data or b_run not in HAND:
            continue
        a = {**data[a_run]["hand"], **data[a_run]["abstain"]}
        b = {**data[b_run]["hand"], **data[b_run]["abstain"]}
        x, y, p = mcnemar_exact(a, b)
        sa = {**data[a_run]["strict"], **data[a_run]["abstain"]}
        sb = {**data[b_run]["strict"], **data[b_run]["abstain"]}
        sx, sy, sp = mcnemar_exact(sa, sb)
        print(f"| {LABELS[b_run]} | {x} | {y} | {p:.3g} | {sx} / {sy}, p = {sp:.3g} |")

    print("\nCourse-specific vs general-knowledge questions (hand labels). Does the course help beyond course trivia?\n")
    print("| configuration | general knowledge (33) | course-specific (5) | course-specific, audit's list (6) |")
    print("|---|---|---|---|")
    for run, o in data.items():
        if run not in HAND or run not in SPLIT_RUNS:
            continue
        h = o["hand"]
        cells = [pct(sum(v for q, v in h.items() if keep(q)), sum(1 for q in h if keep(q)))
                 for keep in (lambda q: q not in COURSE_SPECIFIC, lambda q: q in COURSE_SPECIFIC,
                              lambda q: q in COURSE_SPECIFIC_AUDIT)]
        print(f"| {LABELS[run]} | " + " | ".join(cells) + " |")
    if MAIN in data and "no_rag__ollama__llama3.1-8b" in data:
        general = lambda o: {q: v for q, v in o["hand"].items() if q not in COURSE_SPECIFIC}
        x, y, p = mcnemar_exact(general(data[MAIN]), general(data["no_rag__ollama__llama3.1-8b"]))
        print(f"\nGeneral-knowledge questions only, main vs no course: main wins {x}, loses {y}, exact McNemar p = {p:.3g}")
        general_strict = lambda o: {q: v for q, v in o["strict"].items() if q not in COURSE_SPECIFIC}
        x, y, p = mcnemar_exact(general_strict(data[MAIN]), general_strict(data["no_rag__ollama__llama3.1-8b"]))
        print(f"Same, strict reading: main wins {x}, loses {y}, exact McNemar p = {p:.3g}")


if __name__ == "__main__":
    main()
