"""Regenerate eval/QUESTIONS.md (the readable review copy) from eval/questions.jsonl.

    uv run python eval/make_questions_md.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()  # --help prints this and exits

HERE = Path(__file__).parent
qs = [json.loads(line) for line in open(HERE / "questions.jsonl") if line.strip()]
n = Counter(q["type"] for q in qs)
oos = [q for q in qs if q["type"] == "out_of_scope"]
kinds = Counter(q["source"].split(" ")[0].rstrip(":") for q in oos)

label = {"keyword": "uses course term", "paraphrase": "paraphrased", "out_of_scope": "not covered"}


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")  # keep "P(wn | wn-1)" readable inside a table


lines = [
    "# Evaluation questions (for review)", "",
    f"Generated from [`questions.jsonl`](questions.jsonl) by `make_questions_md.py`. "
    f"The JSONL file is what the evaluation reads. {len(qs)} questions, none reused from development:", "",
    f"- **{n['keyword']}** answerable, using the course's own terms",
    f"- **{n['paraphrase']}** answerable but **paraphrased**: they describe the idea without its name",
    f"- **{n['out_of_scope']} not covered**:",
    f"  - {kinds['near-miss']} near-misses that sound like course topics",
    f"  - {kinds['trap']} trap",
    f"  - {kinds['obviously']} off-topic", "",
    "Each reference answer was written from the source listed. Tests check two things:",
    "- every key fact appears in the gold lecture's text, and none just repeats the question",
    "- no not-covered topic appears anywhere in the course", "",
    "| id | type | question | where it's taught | reference answer |", "|---|---|---|---|---|",
]
for q in qs:
    where = ", ".join([f"Lecture {lec}" for lec in q["gold_lectures"]] + [f"`{f}`" for f in q["gold_files"][:1]]) or "n/a"
    lines.append(f"| {q['id']} | {label[q['type']]} | {cell(q['question'])} | {where}<br><sub>{cell(q['source'])}</sub> "
                 f"| {cell(q['reference'])} |")
(HERE / "QUESTIONS.md").write_text("\n".join(lines) + "\n")
print(f"wrote {HERE / 'QUESTIONS.md'} ({len(qs)} questions)")
