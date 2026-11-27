# Review of the 50 evaluation questions

**What this is:** an AI-assisted review (Claude, in a separate session with no part in writing the questions) of `eval/questions.jsonl`. It is not an independent human review; a second person checking the flagged questions is still the next step.

- **Reviewed:** commit `138b6e7`
- **Fixes applied:** commit `31f816c`
- **Ground truth:** only the course text in `data/processed/documents.jsonl` (538 documents), not general knowledge of the topics.
- **Re-scoring:** the saved answers in `eval/results/` were re-scored with the corrected questions; no model was asked again. The original questions and every original per-answer score are archived in `eval/archive/before_question_review/`.

## Headline

| Verdict at review time | Count | Questions | Status now |
|---|---|---|---|
| Fine | 34 | 23 answerable, 11 not covered | unchanged |
| Wording only (no effect on scores) | 10 | q02, q04, q06, q09, q16, q26, q28, q34, q37, q47 | left as they are (see "Not applied") |
| Grading problem (changes a score) | 6 | q07, q08, q13, q20, q29, q32 | **fixed** |

- **No not-covered topic is actually taught.** Three are partly touched: GRPO (the RL practical computes a mean/std-normalised advantage), PPO (named in the extended Lecture 11 deck, never explained) and L1/L2 (only `L1Loss` and `weight_decay` in code). All three are still fair near-misses.
- **The fixes change only the automatic metric and the citation scores.** The hand-checked scores, the strict reading, the significance tests and the ranking of systems are unchanged, because the hand labels already judged these answers by reading them.

## How the review was done

1. **Answerable questions (q01–q38):**
   - Find the passage that teaches the answer, and compare it with `gold_lectures` / `gold_files`, the reference answer and each key fact.
   - Search the whole corpus for each question's term, to find teaching outside the gold location.
2. **Not-covered questions (q39–q50):** case-insensitive regex searches over all 538 documents, including variants, abbreviations and related wording.
3. **Checking the scorer:** recompute key-fact recall for every saved answer with the project's own matcher. It reproduced the saved values exactly.
4. **Measuring the effect:** for each proposed fix, re-score all 13 saved runs and list every answer whose verdict would flip.

## Fixes applied

| Question | Problem | Evidence (course text) | Fix | Saved answers affected |
|---|---|---|---|---|
| **q29** top-k sampling | Key facts rejected the course's own explanation | GPT practical, exercises: *"Top-k truncates the long tail of garbage words before sampling"* | 2nd fact: `-inf`, `exclude`, `discard`, `remov*`, `truncat*`, `long tail`, `only the k` | 7 Llama + course runs: correct (automatic), 0 → 1 |
| **q20** normalizing flow | Core idea (change of variables) was not a key fact | Lecture 7, slide 7: *"construct a change of variables… to some simple distribution (e.g. Gaussian)… inverse is simple… Jacobian is tractable"* | `[change of variables, transform*] [gaussian, simple distribution, normal distribution] [invert*, inverse, jacobian]` | dense: 0 → 1 |
| **q07** zero/one/few-shot | Too strict (a correct answer failed); `an example` too loose | Lecture 3, slide 8: *"Directly ask without examples… Provide one example… Give multiple examples"* | Added alternatives (`no specific`, `zero examples`, `small set of examples`, `few examples`, `several`); dropped `an example` | no course: 0 → 1 |
| **q08** compound AI systems | Too loose (`multiple`, `performance` pass a generic answer) | Lecture 3, slide 12: *"retrievers, external tools… faster than scaling alone… real-time retrieval… static training data"* | `[retriev*, tool] [scaling, real-time, real time, static, training data, knowledge]` | lenient-syllabus baseline: 1 → 0 |
| **q13** shared CNN weights | Gold location missed Lecture 6, which explains *why* | Lecture 6, slide 16: *"share weights across spatial positions… recognized no matter where it is located"* | `gold_lectures: [4, 6]` | relevance-check-off run: q13 now counts as citing the right lecture |
| **q32** Q-learning | Labelled a paraphrase but uses the course term and wording | RL practical: *"No model required — … when the agent has no map. This is why Q-learning…"* | `type: keyword` (now 31 keyword, 7 paraphrased) | none (moves between subgroups) |
| **Matcher** | Matched word beginnings only, so `wer` ~ "were", `uv` ~ "uvicorn", `inf` ~ "information", `norm` ~ "normal" | `fact_present` in `src/study_assistant/evaluation.py` | Whole words (with -s/-es/-ed/-d/-ing) by default; stems marked with `*` (55 phrases) | none flipped; latent false matches removed |

**Effect on the automatic main metric** (same answers, re-scored):

| Run | Before | After |
|---|---|---|
| Llama + course, hybrid (main) | 36/38 = 95% | 37/38 = 97% |
| Llama + course, dense | 30/38 = 79% | 32/38 = 84% |
| …hybrid, relevance check off | 37/38 = 97% | 37/38 = 97% |
| …hybrid, similarity floor off | 36/38 = 95% | 37/38 = 97% |
| …hybrid, support check strict / concept | 35 / 36 of 38 | 36 / 37 of 38 |
| …hybrid, less / more context | 32 / 33 of 38 | 33 / 34 of 38 |
| Llama, no course | 18/38 = 47% | 19/38 = 50% |
| …sees the syllabus (strict / lenient) | 3 / 11 of 38 | 3 / 10 of 38 |
| flan-t5 with / without course | 9 / 0 of 38 | 9 / 0 of 38 |

The strict reading still counts q29 as *borderline* (not correct). The key-fact fix corrects the automatic metric, not our judgement that the answer never says sampling is limited to the k most likely tokens. So the strict scores (main 92%, dense 87%) are unchanged.

## Not applied, and why

These have no effect on any saved score. Changing a reference answer would also make the judge's scores for that question out of date.

| Question | Note |
|---|---|
| q02 | The intrinsic-hallucination key fact also matches the extrinsic wording (`contradict*`). The reference says "contradicts / wrongly summarizes"; the course says only "wrongly summarized". This is why q02 is borderline in the strict reading. |
| q04 | Key facts check the definition but not *why* the Markov assumption is used. |
| q06, q16, q28 | Weak paraphrases: they reuse the slide's distinctive wording ("self-driving car… noisy sensor", "lower-dimensional surface", attention in an image GAN). |
| q09 | The reference leaves out "environmental impact" (Lecture 3, slide 23). |
| q26 | The reference describes value embeddings in standard terms, not the course's ("gives the relative weight"). |
| q34 | `generat*` also matches "Generative AI" and `reward` matches "Reward Modeling"; a tighter list would be fairer. |
| q37 | Writing out "Retrieval-Augmented Generation" satisfies 3 of the 5 stage facts. |
| q47 | The source note should say the RL practical partly covers advantages (not GRPO's group baseline). |

## Not-covered questions (q39–q50)

| id | Label | Searched for | Result |
|---|---|---|---|
| q39 label smoothing | near-miss | `label.?smooth`, `smooth(ed\|ing)? label`, `soft (label\|target)` | no hits (only n-gram "smoothing") |
| q40 PPO's clipped objective | near-miss | `PPO`, `proximal`, `clip… objective/ratio`, `surrogate`, `trust region` | PPO named (extended Lecture 11 deck, slides 11, 14, 29); no clipped objective |
| q41 VQ-VAE codebook | near-miss | `VQ`, `vector quanti`, `codebook`, `discrete latent` | no hits (only 4-bit model quantization) |
| q42 Stable Diffusion latent space | near-miss | `stable diffusion`, `latent diffusion`, `LDM` | no hits; the diffusion practical works on pixels |
| q43 mixture of experts | near-miss | `mixture of experts`, `MoE`, `gating network`, `load balanc` | no hits ("router" is the LangGraph router) |
| q44 FID | near-miss | `FID`, `Fréchet`, `inception` | no hits ("Inception" is a movie in a toy dataset) |
| q45 L1 vs L2 regularization | near-miss | `L1`, `L2`, `regulari[sz]`, `weight decay`, `lasso`, `ridge` | not explained; only `nn.L1Loss`, `weight_decay=…`, dropout |
| q46 DPO | near-miss | `DPO`, `direct preference`, `preference optim` | no hits (RLHF reward models are covered) |
| q47 GRPO advantages | near-miss | `GRPO`, `group relative`, `advantage`, `baseline` | no GRPO; the RL practical normalises an advantage |
| q48 sourdough | off-topic | `sourdough`, `dough`, `yeast`, `bak(e\|ing)` | only "bread" in a toy vocabulary |
| q49 Everest boiling point | trap | `everest`, `boil` | Everest is in toy RAG data; no "boil" anywhere |
| q50 Wimbledon | off-topic | `wimbledon`, `tennis` | no hits |

## Coverage, overlap and difficulty

**Coverage**
- Every lecture has at least one question: Lecture 7 (normalizing flows) has only one and Lecture 11 two; the others three to five. The add-on labs have one each (evals, LangGraph, RAG primer, Whisper fine-tuning).
- Topics with no evaluation question are mostly covered by the 35 development queries (VAE reparameterization, GAN losses, LSTM, Real NVP, diffusion process, Langevin sampling, Bellman, RLHF reward model, LoRA, BLEU, …).
- Covered by neither set: temperature sampling, dropout and batch norm, credit assignment, REINFORCE, the DistilBERT fine-tuning lab, the Docker/FastAPI activity, and multimodal prompting.

**Overlap**
- No duplicates, and no question repeats a development query.
- A few share topics with development queries (q25 positional encoding, q20 normalizing flows, q24 diffusion noising, q35 LLM evaluation, q33/q34 RLHF). Hybrid retrieval was chosen on the development set, so these may be slightly easier for it.

**Difficulty**
- Mostly single-passage lookups: about 30 of 38 answers sit in one slide or notebook cell that uses the question's own term.
- No question needs combining two lectures.
- The report already notes the paraphrases were easier than the development ones.
- A harder set would add cross-lecture comparisons (e.g. how VAEs, GANs and diffusion models generate images) and paraphrases that avoid the slide's wording as well as its term.

## All 50 questions

| id | type | verdict | status |
|---|---|---|---|
| q01 | keyword | fine | – |
| q02 | keyword | wording only | not applied (borderline in strict reading) |
| q03 | keyword | fine | `uv` no longer matches "uvicorn" |
| q04 | keyword | wording only | not applied |
| q05 | keyword | fine | – |
| q06 | paraphrase | wording only | not applied |
| q07 | keyword | grading problem | **fixed** |
| q08 | keyword | grading problem | **fixed** |
| q09 | keyword | wording only | not applied |
| q10–q12 | keyword | fine | – |
| q13 | paraphrase | grading problem | **fixed** (gold [4, 6]) |
| q14, q15 | keyword, paraphrase | fine | – |
| q16 | paraphrase | wording only | not applied |
| q17–q19 | keyword, keyword, paraphrase | fine | `norm` no longer matches "normal" (q17) |
| q20 | keyword | grading problem | **fixed** |
| q21–q25 | keyword | fine | – |
| q26 | keyword | wording only | not applied |
| q27 | keyword | fine | – |
| q28 | paraphrase | wording only | not applied |
| q29 | keyword | grading problem | **fixed** (still borderline in strict reading) |
| q30, q31 | paraphrase, keyword | fine | – |
| q32 | keyword (was paraphrase) | grading problem | **fixed** (relabelled) |
| q33 | keyword | fine | – |
| q34 | keyword | wording only | not applied |
| q35, q36 | keyword | fine | – |
| q37 | keyword | wording only | not applied |
| q38 | keyword | fine | `wer` no longer matches "were" |
| q39–q46 | not covered | fine | – |
| q47 | not covered | wording only | not applied |
| q48–q50 | not covered | fine | – |
