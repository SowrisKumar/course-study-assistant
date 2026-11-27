# Evaluation run
- started: 2026-11-25 02:33 UTC
- code: git 91cc6b3
- answer + judge model: llama3.1:8b (digest 46e0c10c039e)
- machine: Apple M2

| configuration | minutes |
|---|---|
| Llama + course, hybrid (main) | 14 |
| Llama + course, dense | 13 |
| Llama, no course | 8 |
| Ablation: relevance check off | 11 |
| Ablation: support check on | 17 |
| flan-t5, with and without course | 3 |
| Judge v2 (with course excerpts), all runs | 16 |
- finished: 2026-11-25 03:59 UTC

## Added later
- code: git c026e50

| configuration | minutes |
|---|---|
| Ablation: support check on (concept) | 17 |

Judge v2 was then re-run with `--rejudge` (it only scores answers that have no v2 score yet).

## Added later (quick improvements)
- code: git 308ebb8

| configuration | minutes |
|---|---|
| Llama, no course, sees the syllabus | 2 |
| Ablation: similarity floor off | 14 |
| Ablation: less context (3 passages) | 9 |
| Ablation: more context (8 passages) | 18 |
| Judge v2 for the new runs | 9 |
| …syllabus, lenient wording (code 902ebee) | 7 |
| Judge v2 for it | 2 |

## Re-scored after the question review
- code: git 31f816c
- `evaluation --rescore`: six questions corrected and whole-word key-fact matching (eval/QUESTION_REVIEW.md).
  Key-fact recall and citation scores were recomputed from the saved answers; no model was asked again.
  Original scores: eval/archive/before_question_review/.

