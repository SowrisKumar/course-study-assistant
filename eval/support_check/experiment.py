"""Step 6 experiment: does a support check after answering remove unsupported answers?

Runs on SAVED answers (no regeneration), so it takes minutes, not an hour. Needs Ollama + llama3.1:8b.

    uv run python eval/support_check/experiment.py collect-dev                 # save development answers (~8 min)
    uv run python eval/support_check/experiment.py dev                         # compare check variants on dev
    uv run python eval/support_check/experiment.py eval strict                 # apply the chosen variant to Step 5 answers

Variants: "strict" (the prompt in study_assistant/prompts.py: is the answer's main content stated in the
passages?) and "concept" (do the passages discuss the specific concept asked about?). The variant applied to
the evaluation set was fixed in advance; the development set was used for all trying-out.
Results: eval/support_check/*.json(l); discussion in docs/report.md, Section 7.
"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
from study_assistant import prompts
from study_assistant.assistant import group_into_passages
from study_assistant.llm import get_llm

CONCEPT_PROMPT = """Question: {question}

Course passages:
{passages}

Do these passages discuss the specific concept, method or system the question asks about (by name, or by clearly describing it)?
Passages about a related or more general topic do NOT count: for example, passages about transformers do not discuss "rotary embeddings" unless they mention or describe them.

Reply with exactly one line:
DISCUSSED: yes
or
DISCUSSED: no"""
CONCEPT_SYSTEM = "You check whether course passages cover the specific thing a student asked about."

llm = get_llm("ollama")


def passages_text(relevant):
    return "\n\n".join(f"({p['label']})\n{p['text']}" for p in relevant)


def check(variant, question, answer, relevant):
    """True = keep the answer, False = reject (abstain)."""
    if variant == "strict":
        reply = llm.generate(prompts.VERIFY_PROMPT.format(question=question, passages=passages_text(relevant),
                             answer=re.sub(r"\s*\[\d+\]", "", answer)), system=prompts.VERIFY_SYSTEM, max_tokens=10).text
        m = re.search(r"SUPPORTED\s*:\s*(yes|no)", reply, re.I)
    else:
        reply = llm.generate(CONCEPT_PROMPT.format(question=question, passages=passages_text(relevant)),
                             system=CONCEPT_SYSTEM, max_tokens=10).text
        m = re.search(r"DISCUSSED\s*:\s*(yes|no)", reply, re.I)
    return not (m and m.group(1).lower() == "no")


HERE = __import__("pathlib").Path(__file__).parent

if sys.argv[1] == "collect-dev":
    from study_assistant.assistant import StudyAssistant
    from study_assistant.retrieval import build_retriever
    bot = StudyAssistant(llm, build_retriever("hybrid"), verify=False)
    with open(HERE / "dev_answers.jsonl", "w") as out:
        for line in open(HERE.parent / "retrieval_queries.jsonl"):
            q = json.loads(line)
            s = bot.graph.invoke({"question": q["query"], "trace": []})
            out.write(json.dumps({"id": q["id"], "question": q["query"], "oos": q.get("out_of_scope", False),
                "abstained": s.get("abstained", False), "answer": s.get("answer", ""),
                "relevant": [{"label": p.label, "text": p.text} for p in s.get("relevant", [])]}) + "\n")

elif sys.argv[1] == "dev":
    rows = [json.loads(l) for l in open(HERE / "dev_answers.jsonl")]
    answered = [r for r in rows if not r["abstained"]]
    print(f"{len(rows)} dev questions, {len(answered)} answered ({sum(r['oos'] for r in answered)} of them out-of-scope)")
    for variant in ["strict", "concept"]:
        rejected = [r for r in answered if not check(variant, r["question"], r["answer"], r["relevant"])]
        print(f"  {variant:8} rejects {len(rejected)}: " + ", ".join(f"{r['id']}{'(OOS)' if r['oos'] else ''}" for r in rejected))

else:  # eval: rebuild the passages each saved Step 5 answer was written from
    variant = sys.argv[2]
    from study_assistant.evaluation import load_questions, load_results
    from study_assistant.retrieval import build_retriever
    Q = {q["id"]: q for q in load_questions()}
    df = load_results()
    out = []
    for run, ret_name in [("rag__ollama__llama3.1-8b__hybrid", "hybrid"), ("rag__ollama__llama3.1-8b__dense", "dense")]:
        retriever = build_retriever(ret_name)
        for _, r in df[(df.run == run) & ~df.abstained].iterrows():
            passages = group_into_passages(retriever.search(r.question, k=8), 5)
            # check that the rebuilt passages match the saved trace (retrieval is deterministic)
            saved = [l.strip().split(" ", 2)[-1] for l in r.trace if re.match(r"\s+\[\d+\] \d", l)]  # "[1] 0.63 <label>"
            assert saved == [p.label for p in passages], (run, r.id)
            m = re.search(r"relevant=\[([\d, ]+)\]", " ".join(r.trace))
            keep = [int(x) for x in m.group(1).split(",")] if m else [p.number for p in passages]
            relevant = [{"label": p.label, "text": p.text} for p in passages if p.number in keep]
            ok = (r.judge2_score == 2) and (r.key_fact_recall or 0) >= 0.5 if r.type != "out_of_scope" else None
            out.append({"run": run, "id": r.id, "type": r.type, "was_correct": ok,
                        "kept": check(variant, r.question, r.answer, relevant)})
    json.dump(out, open(HERE / f"eval_{variant}.json", "w"), indent=1)
    for run in sorted({o["run"] for o in out}):
        rs = [o for o in out if o["run"] == run]
        print(run)
        print("  out-of-scope answers now rejected:", [o["id"] for o in rs if o["type"] == "out_of_scope" and not o["kept"]],
              "of", [o["id"] for o in rs if o["type"] == "out_of_scope"])
        print("  covered answers now rejected:", [(o["id"], "was correct" if o["was_correct"] else "was wrong") for o in rs if o["type"] != "out_of_scope" and not o["kept"]])
