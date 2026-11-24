"""Step 5 – Evaluation: the assistant (RAG) vs a chatbot that can't see the course (no-RAG).

Question set: eval/questions.jsonl, 50 questions written for this step (none reused from
development): 38 answerable (30 using the course's terms, 8 paraphrased) and 12 not covered
(9 near-misses that sound like course topics, 3 off-topic). Every reference answer was written
from the cited slide/notebook, and a check confirmed each key fact appears in the gold material
and each not-covered topic does not appear anywhere in the course.

What is measured (following the tiers of the class llm_evals notebook):

  Abstention   not-covered questions answered "not covered" (higher = better)
               answerable questions wrongly refused (false refusals; lower = better)
  Correctness  tier 1 (rules): share of the question's key facts found in the answer
               tier 4 (LLM-as-judge): 0 wrong / 1 partial / 2 correct against the reference answer
  Citations    (RAG only) the answer cites at least one source from the right lecture/lab,
               and what share of its cited sources are from the right lecture/lab

Usage:
    uv run python -m study_assistant.evaluation --backend ollama --systems rag,no_rag
    uv run python -m study_assistant.evaluation --report
Runs are saved to eval/results/<run>.jsonl one question at a time, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import prompts
from .config import PROJECT_ROOT
from .llm import LLM, get_llm

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# --- the no-RAG baseline ---------------------------------------------------------------------

BASELINE_SYSTEM = f"""You are a study assistant for the university course Applied Generative AI.
Answer the student's question clearly and concisely (at most about 150 words).
If the question is not something the course would cover, or you don't know, reply exactly: "{prompts.NOT_COVERED}" """


def answer_without_course(llm: LLM, question: str) -> dict:
    r = llm.generate(question, system=BASELINE_SYSTEM, max_tokens=400)
    abstained = is_refusal(r.text)
    return {"answer": prompts.NOT_COVERED if abstained else r.text, "abstained": abstained,
            "sources": [], "latency_s": r.latency_s}


_REFUSAL = re.compile(r"(isn't|is not) covered in the course|\bnot covered\b|\bi don'?t know\b", re.I)


def is_refusal(text: str) -> bool:
    t = text.strip().strip('"').strip()
    return not t or bool(_REFUSAL.search(t[:200]))


# --- the assistant (RAG) ---------------------------------------------------------------------------


def answer_with_course(bot, question: str) -> dict:
    start = time.perf_counter()
    r = bot.ask(question)
    return {
        "answer": r.answer,
        "abstained": r.abstained,
        "abstain_reason": r.abstain_reason,
        "uncited": r.uncited,
        "sources": [{"lecture": p.lecture, "doc_id": p.doc_id, "label": p.label} for p in r.sources],
        "retrieved": [{"lecture": p.lecture, "doc_id": p.doc_id} for p in r.passages],
        "trace": r.trace,
        "latency_s": round(time.perf_counter() - start, 3),
    }


# --- scoring ---------------------------------------------------------------------------------------


def key_fact_recall(answer: str, key_facts: list[list[str]]) -> float | None:
    """Tier 1: share of key facts (each a list of acceptable phrasings) present in the answer."""
    if not key_facts:
        return None
    a = answer.lower()
    return sum(any(alt.lower() in a for alt in fact) for fact in key_facts) / len(key_facts)


def _source_file(doc_id: str) -> str:
    return doc_id.split("::")[0]


def citation_scores(sources: list[dict], q: dict) -> tuple[bool | None, float | None]:
    """(cites at least one gold source, share of cited sources that are gold); None if nothing cited."""
    if not sources:
        return None, None
    gold = [s["lecture"] in q["gold_lectures"] or _source_file(s["doc_id"]) in q["gold_files"] for s in sources]
    return any(gold), sum(gold) / len(gold)


JUDGE_SYSTEM = "You are a strict, fair grader for a university course on Applied Generative AI."

JUDGE_PROMPT = """Grade the assistant's answer against the reference answer, which was written from the course materials.

Question: {question}

Reference answer: {reference}

Assistant's answer: {answer}

Scoring:
2 = correct: agrees with the reference on the main point(s); it may be shorter or worded differently, and extra detail is fine if it does not contradict the reference.
1 = partially correct: gets some of the main point right but misses or gets wrong an important part.
0 = incorrect: wrong, contradicts the reference, or does not answer the question.
Ignore citation markers like [1].

Reply with exactly two lines:
SCORE: <0, 1 or 2>
REASON: <one short sentence>"""


def judge(judge_llm: LLM, q: dict, answer: str) -> tuple[int | None, str]:
    """Tier 4: LLM-as-judge score 0/1/2 against the reference answer (None if unparseable)."""
    reply = judge_llm.generate(
        JUDGE_PROMPT.format(question=q["question"], reference=q["reference"], answer=answer),
        system=JUDGE_SYSTEM, max_tokens=80,
    ).text
    m = re.search(r"SCORE\s*:\s*([012])", reply)
    reason = re.search(r"REASON\s*:\s*(.*)", reply)
    return (int(m.group(1)) if m else None), (reason.group(1).strip() if reason else reply.strip()[:200])


# --- running ---------------------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class RunConfig:
    system: str  # "rag" | "no_rag"
    backend: str
    model: str
    retriever: str | None = None

    @property
    def name(self) -> str:
        model = re.sub(r"[^A-Za-z0-9.]+", "-", self.model).strip("-")
        return "__".join([self.system, self.backend, model] + ([self.retriever] if self.retriever else []))


def run(config: RunConfig, llm: LLM, questions: list[dict], *, bot=None, judge_llm: LLM | None = None,
        out_dir: Path = RESULTS_DIR, log=print) -> Path:
    """Answer (and judge) every question, appending one JSON line per question. Resumes if interrupted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config.name}.jsonl"
    results: dict[str, dict] = {}  # question id -> record; previous results are reused (resume)
    if path.exists():
        results = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}

    for i, q in enumerate(questions, start=1):
        rec = results.get(q["id"])
        if rec is None:
            res = answer_with_course(bot, q["question"]) if config.system == "rag" else answer_without_course(llm, q["question"])
            rec = {"id": q["id"], "type": q["type"], "question": q["question"], **config.__dict__, "run": config.name, **res}
        if q["type"] != "out_of_scope" and not rec["abstained"]:
            rec["key_fact_recall"] = key_fact_recall(rec["answer"], q["key_facts"])
            rec["cites_gold"], rec["citation_precision"] = citation_scores(rec.get("sources", []), q)
            if judge_llm is not None and rec.get("judge_model") != judge_llm.model:
                rec["judge_score"], rec["judge_reason"] = judge(judge_llm, q, rec["answer"])
                rec["judge_model"] = judge_llm.model
        results[q["id"]] = rec
        # save after every question so an interrupted run loses nothing
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results.values()), encoding="utf-8")
        log(f"[{config.name}] {i}/{len(questions)} {q['id']} {'ABSTAIN' if rec['abstained'] else 'answered'}"
            + (f" judge={rec.get('judge_score')}" if "judge_score" in rec else ""))
    return path


# --- reporting -------------------------------------------------------------------------------------


def load_results(out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(out_dir.glob("*.jsonl")):
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run. Correctness is over ALL answerable questions (a refusal counts as not correct)."""
    out = {}
    for run_name, g in df.groupby("run", sort=False):
        ans = g[g.type != "out_of_scope"]
        oos = g[g.type == "out_of_scope"]
        answered = ans[~ans.abstained]
        js = ans["judge_score"] if "judge_score" in ans else pd.Series(dtype=float)
        row = {
            "answerable": len(ans),
            "correct (judge=2)": (js == 2).sum() / len(ans) if js.notna().any() else None,
            "correct or partial": (js >= 1).sum() / len(ans) if js.notna().any() else None,
            "key-fact recall": ans["key_fact_recall"].fillna(0).mean() if "key_fact_recall" in ans else None,
            "false refusals": ans.abstained.mean(),
            "not covered → abstained": oos.abstained.mean() if len(oos) else None,
            "near-miss → abstained": oos[oos.question.isin(_NEAR_MISS)].abstained.mean() if len(oos) else None,
            "cites right lecture": answered["cites_gold"].mean() if "cites_gold" in answered and answered["cites_gold"].notna().any() else None,
            "citation precision": answered["citation_precision"].mean() if "citation_precision" in answered and answered["citation_precision"].notna().any() else None,
            "sec / question": g.latency_s.mean(),
        }
        out[run_name] = row
    return pd.DataFrame(out).T.astype(float).round(3)


_NEAR_MISS = {q["question"] for q in load_questions() if q["type"] == "out_of_scope" and q["source"].startswith(("near-miss", "trap"))} \
    if QUESTIONS_PATH.exists() else set()


# --- CLI ---------------------------------------------------------------------------------------------


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description="Evaluate the study assistant (RAG) against a no-RAG chatbot")
    p.add_argument("--backend", default="ollama", help="LLM that answers")
    p.add_argument("--model", help="override the backend's default model")
    p.add_argument("--systems", default="rag,no_rag", help="comma list of: rag, no_rag")
    p.add_argument("--retriever", default=None, help="retriever for the RAG system (default: the assistant's)")
    p.add_argument("--judge", default="ollama", help="backend for LLM-as-judge ('none' to skip)")
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--report", action="store_true", help="only print the summary of saved runs")
    args = p.parse_args()

    if not args.report:
        from .assistant import DEFAULT_RETRIEVER, StudyAssistant
        from .retrieval import build_retriever

        questions = load_questions()[: args.limit]
        llm = get_llm(args.backend, args.model)
        judge_llm = None if args.judge == "none" else get_llm(args.judge)
        for system in args.systems.split(","):
            bot, retriever = None, None
            if system == "rag":
                retriever = args.retriever or DEFAULT_RETRIEVER
                grading = "threshold" if args.backend == "flan" else "llm"
                bot = StudyAssistant(llm, build_retriever(retriever), grading=grading)
            run(RunConfig(system, args.backend, llm.model, retriever), llm, questions, bot=bot,
                judge_llm=judge_llm, out_dir=args.out)

    pd.set_option("display.width", 250)
    print(summarize(load_results(args.out)).to_string())


if __name__ == "__main__":
    main()
