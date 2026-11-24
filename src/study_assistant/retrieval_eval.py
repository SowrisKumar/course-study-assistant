"""Measure how well a retriever finds the right course material.

A retrieved chunk counts as relevant if it comes from one of the question's gold
lectures or gold files (see eval/retrieval_queries.jsonl).

Metrics (computed over in-scope questions only):
  hit@k  – share of questions with at least one relevant chunk in the top k
  MRR@10 – mean of 1/rank of the first relevant chunk (0 if none in the top 10);
           rewards putting the right material first, not just somewhere in the list
For out-of-scope questions we record the best `dense_score`, to see whether a
similarity threshold can separate "covered" from "not covered" questions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .chunking import Chunk
from .config import PROJECT_ROOT
from .retrieval import Retriever

QUERIES_PATH = PROJECT_ROOT / "eval" / "retrieval_queries.jsonl"


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_relevant(chunk: Chunk, q: dict) -> bool:
    return chunk.lecture in q["gold_lectures"] or chunk.source_file in q["gold_files"]


def evaluate(retriever: Retriever, queries: list[dict], depth: int = 10) -> pd.DataFrame:
    """One row per question: rank of the first relevant hit, top hit, best dense score."""
    rows = []
    for q in queries:
        hits = retriever.search(q["query"], k=depth)
        first = next((h.rank for h in hits if is_relevant(h.chunk, q)), None)
        rows.append({
            "method": retriever.name,
            "id": q["id"],
            "query": q["query"],
            "out_of_scope": q.get("out_of_scope", False),
            "first_relevant_rank": first,
            "top_hit": hits[0].chunk.citation,
            "max_dense_score": max(h.dense_score for h in hits),
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """hit@k and MRR@10 per method (in-scope questions only)."""
    inscope = results[~results.out_of_scope]
    out = {}
    for method, g in inscope.groupby("method", sort=False):
        r = g.first_relevant_rank
        row = {f"hit@{k}": (r <= k).mean() for k in ks}
        row["MRR@10"] = (1.0 / r).fillna(0).mean()
        out[method] = row
    return pd.DataFrame(out).T.round(3)
