"""Experiment log: one row per (build, configuration), appended after every evaluation run.

A *build* is one state of the system (code + knowledge base + question set). Logging each build's
scores lets us compare models and versions over time without overwriting earlier results.

    uv run python -m study_assistant.experiment_log --build "final (after audit)"
    uv run python -m study_assistant.experiment_log --show

The log is `eval/experiment_log.csv` (readable on GitHub). Re-logging a (build, configuration) pair
replaces its row, so the file never holds duplicates. `notebooks/06_experiment_log.ipynb` charts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .evaluation import QUESTIONS_PATH, RESULTS_DIR, load_results, summarize

LOG_PATH = PROJECT_ROOT / "eval" / "experiment_log.csv"
HAND_LABELS_PATH = PROJECT_ROOT / "eval" / "hand_labels.json"

METRICS = {  # summarize() column -> log column
    "correct (main)": "correct_main",
    "correct (judge v2)": "correct_judge_v2",
    "correct (judge v1)": "correct_judge_v1",
    "key-fact recall": "key_fact_recall",
    "false refusals": "false_refusals",
    "not covered → abstained": "not_covered_declined",
    "cites right lecture": "cites_right_lecture",
    "citation precision": "citation_precision",
    "sec / question": "sec_per_question",
}
COLUMNS = ["build", "logged_at", "commit", "kb_documents", "questions_hash", "model_digest", "run", "system",
           "backend", "model", "retriever", "variant", "n_answerable", "n_not_covered", "correct_hand",
           *METRICS.values(), "near_miss_declined", "notes"]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _model_digest(results_dir: Path) -> str:
    """The Ollama model digest written by scripts/run_evaluation.sh, if the run recorded one."""
    info = results_dir / "RUN_INFO.md"
    m = re.search(r"digest ([0-9a-f]+)", info.read_text()) if info.exists() else None
    return m.group(1) if m else ""


def _kb_documents() -> int:
    from .ingest import load_documents

    return len(load_documents())


def hand_corrected(results: pd.DataFrame, labels: dict) -> dict[str, float]:
    """Share of answerable questions correct after the hand check, for runs that have hand labels.

    Answers without a hand label keep the main metric's verdict (see eval/hand_labels.json)."""
    out = {}
    for run, g in results[results.type != "out_of_scope"].groupby("run"):
        if run not in labels:
            continue
        auto = (~g.abstained) & (g.judge2_score == 2) & (g.key_fact_recall.fillna(0) >= 0.5)
        hand = [labels[run][i][0] == "correct" if i in labels[run] else bool(a) for i, a in zip(g.id, auto)]
        out[run] = round(sum(hand) / len(hand), 3)
    return out


def record(build: str, results_dir: Path = RESULTS_DIR, *, runs: list[str] | None = None, commit: str | None = None,
           kb_documents: int | None = None, questions_path: Path = QUESTIONS_PATH, hand_labels: Path | None = None,
           notes: str = "", log_path: Path = LOG_PATH) -> pd.DataFrame:
    """Add (or replace) one row per configuration found in `results_dir`; returns the new rows."""
    results = load_results(results_dir)
    if runs:
        results = results[results.run.isin(runs)]
    if results.empty:
        raise SystemExit(f"no results in {results_dir}")
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]
    near_miss = {q["id"] for q in questions if q["type"] == "out_of_scope" and q["source"].startswith(("near-miss", "trap"))}
    labels = {k: v for k, v in json.loads(hand_labels.read_text()).items() if not k.startswith("_")} if hand_labels else {}
    hand = hand_corrected(results, labels)
    summary = summarize(results)
    meta = {
        "build": build,
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": commit if commit is not None else _git_commit(),
        "kb_documents": kb_documents if kb_documents is not None else _kb_documents(),
        "questions_hash": hashlib.sha1(questions_path.read_bytes()).hexdigest()[:8],
        "model_digest": _model_digest(results_dir),
        "notes": notes,
    }
    rows = []
    for run, g in results.groupby("run", sort=False):
        first = g.iloc[0]
        oos = g[g.type == "out_of_scope"]
        nm = oos[oos.id.isin(near_miss)]
        rows.append({
            **meta, "run": run, "system": first.system, "backend": first.backend, "model": first.model,
            "retriever": first.get("retriever") or "", "variant": first.get("variant") or "",
            "n_answerable": int((g.type != "out_of_scope").sum()), "n_not_covered": len(oos),
            "correct_hand": hand.get(run),
            **{col: summary.loc[run, name] for name, col in METRICS.items()},
            "near_miss_declined": round(nm.abstained.mean(), 3) if len(nm) else None,
        })
        if first.backend != "ollama":
            rows[-1]["model_digest"] = ""
    new = pd.DataFrame(rows, columns=COLUMNS)
    log = pd.read_csv(log_path, dtype=str) if log_path.exists() else pd.DataFrame(columns=COLUMNS)
    log = log[~(log.build.isin(new.build) & log.run.isin(new.run))]
    pd.concat([log, new.astype(str).replace({"None": "", "nan": ""})], ignore_index=True)[COLUMNS].to_csv(log_path, index=False)
    return new


def load_log(log_path: Path = LOG_PATH) -> pd.DataFrame:
    log = pd.read_csv(log_path)
    log["variant"] = log["variant"].fillna("")
    log["retriever"] = log["retriever"].fillna("")
    return log


def main() -> None:
    p = argparse.ArgumentParser(description="Record evaluation results in the experiment log")
    p.add_argument("--build", help="name of this build, e.g. 'final (after audit)'")
    p.add_argument("--results", type=Path, default=RESULTS_DIR, help="folder with the run's result files")
    p.add_argument("--runs", help="comma list: only log these configurations")
    p.add_argument("--commit", help="code version (default: current git HEAD)")
    p.add_argument("--kb-documents", type=int, help="knowledge-base size (default: current knowledge base)")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="question set the run used")
    p.add_argument("--hand", type=Path, help="hand labels for these results (e.g. eval/hand_labels.json)")
    p.add_argument("--notes", default="")
    p.add_argument("--show", action="store_true", help="print the log")
    args = p.parse_args()
    if args.build:
        new = record(args.build, args.results, runs=args.runs.split(",") if args.runs else None, commit=args.commit,
                     kb_documents=args.kb_documents, questions_path=args.questions, hand_labels=args.hand,
                     notes=args.notes)
        print(f"logged {len(new)} configuration(s) as build {args.build!r} in {LOG_PATH.relative_to(PROJECT_ROOT)}")
    if args.show or not args.build:
        pd.set_option("display.width", 250)
        print(load_log()[["build", "commit", "run", "correct_main", "correct_hand", "not_covered_declined",
                          "sec_per_question"]].to_string(index=False))


if __name__ == "__main__":
    main()
