from __future__ import annotations

import json
import re

import pytest

from study_assistant import prompts
from study_assistant.evaluation import (
    RunConfig, citation_scores, is_refusal, judge, key_fact_recall, load_questions, load_results, run, summarize,
)
from study_assistant.ingest import load_documents
from study_assistant.llm import LLM, LLMResponse


class Scripted(LLM):
    name = "scripted"

    def __init__(self, reply):
        super().__init__("fake")
        self.reply, self.calls = reply, 0

    def _generate(self, prompt, system, max_tokens, temperature):
        self.calls += 1
        return LLMResponse(self.reply if isinstance(self.reply, str) else self.reply(prompt), self.name, self.model, 0.0)


# --- the question set itself ------------------------------------------------------------------


def test_question_set_shape():
    qs = load_questions()
    assert len(qs) == 50 and len({q["id"] for q in qs}) == 50
    types = [q["type"] for q in qs]
    assert types.count("out_of_scope") == 12 and types.count("paraphrase") == 8
    for q in qs:
        if q["type"] != "out_of_scope":
            assert q["gold_lectures"] or q["gold_files"], q["id"]
            assert q["key_facts"] and q["reference"], q["id"]


def test_questions_not_reused_from_development():
    dev = {json.loads(l)["query"].lower() for l in open("eval/retrieval_queries.jsonl")}
    assert not dev & {q["question"].lower() for q in load_questions()}


def test_key_facts_appear_in_gold_material():
    docs = load_documents()
    for q in load_questions():
        if q["type"] == "out_of_scope":
            continue
        text = " ".join(d.text.lower() for d in docs if d.lecture in q["gold_lectures"] or d.source_file in q["gold_files"])
        for fact in q["key_facts"]:
            assert any(alt.lower() in text for alt in fact), (q["id"], fact)


@pytest.mark.parametrize("term", ["vanishing", "ppo", "vq-vae", "stable diffusion", "mixture of experts", "fid",
                                  "l2 regular", "dpo", "grpo", "sourdough", "boiling point", "wimbledon"])
def test_not_covered_topics_are_absent(term):
    text = " ".join(d.text.lower() for d in load_documents())
    assert not re.search(r"\b" + re.escape(term) + r"\b", text)


# --- scoring ------------------------------------------------------------------------------------


def test_key_fact_recall():
    facts = [["14%", "14 percent"], ["novice", "34%"]]
    assert key_fact_recall("Productivity rose 14 percent; novices gained most.", facts) == 1.0
    assert key_fact_recall("Productivity rose 14%.", facts) == 0.5
    assert key_fact_recall("anything", []) is None


def test_citation_scores():
    q = {"gold_lectures": [5], "gold_files": ["lora_finetuning-1.ipynb"]}
    srcs = [{"lecture": 5, "doc_id": "x.pptx::slide-1"}, {"lecture": None, "doc_id": "other.ipynb::section-2"}]
    assert citation_scores(srcs, q) == (True, 0.5)
    assert citation_scores([{"lecture": None, "doc_id": "lora_finetuning-1.ipynb::section-1"}], q) == (True, 1.0)
    assert citation_scores([], q) == (None, None)


@pytest.mark.parametrize("text,expected", [
    (prompts.NOT_COVERED, True), ('"This isn\'t covered in the course materials."', True), ("I don't know.", True),
    ("", True), ("GANs have a generator and a discriminator.", False),
])
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected


def test_judge_parses_score_and_reason():
    q = {"question": "Q?", "reference": "R."}
    assert judge(Scripted("SCORE: 2\nREASON: matches."), q, "A") == (2, "matches.")
    assert judge(Scripted("I would say it's fine"), q, "A")[0] is None


# --- running ------------------------------------------------------------------------------------


def test_run_baseline_scores_and_resumes(tmp_path):
    qs = [{"id": "a1", "type": "keyword", "question": "What gates does a GRU have?", "gold_lectures": [6], "gold_files": [],
           "reference": "Reset and update.", "key_facts": [["reset"], ["update"]]},
          {"id": "o1", "type": "out_of_scope", "question": "How do I make bread?", "gold_lectures": [], "gold_files": [],
           "reference": prompts.NOT_COVERED, "key_facts": []}]
    answerer = Scripted(lambda p: prompts.NOT_COVERED if "bread" in p else "A reset gate and an update gate.")
    grader = Scripted("SCORE: 2\nREASON: ok")
    cfg = RunConfig("no_rag", "scripted", "fake")
    run(cfg, answerer, qs, judge_llm=grader, out_dir=tmp_path, log=lambda *_: None)

    df = load_results(tmp_path)
    a1, o1 = df.set_index("id").loc["a1"], df.set_index("id").loc["o1"]
    assert not a1.abstained and a1.key_fact_recall == 1.0 and a1.judge_score == 2
    assert o1.abstained
    s = summarize(df).iloc[0]
    assert s["correct (judge=2)"] == 1.0 and s["not covered → abstained"] == 1.0 and s["false refusals"] == 0.0

    calls = answerer.calls, grader.calls
    run(cfg, answerer, qs, judge_llm=grader, out_dir=tmp_path, log=lambda *_: None)  # resume: nothing redone
    assert (answerer.calls, grader.calls) == calls
