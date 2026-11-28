# Review of the 50 evaluation questions

This is a review of `eval/questions.jsonl`. Claude did the first pass, in a separate session that had no part in writing the questions. Sowris Kumar Sripathy then went through every question it flagged (see "Human review" below). No one outside the team has reviewed the set.

The review looked at commit `138b6e7`, and the fixes went in at commit `31f816c`. The only ground truth used was the course text in `data/processed/documents.jsonl` (538 documents), not general knowledge of the topics. After the fixes, we re-scored the saved answers in `eval/results/` against the corrected questions without asking any model again. The original questions and every original per-answer score are kept in `eval/archive/before_question_review/`.

## Summary

| Verdict at review time | Count | Questions | Status now |
|---|---|---|---|
| Fine | 34 | 23 answerable, 11 not covered | unchanged |
| Wording only (no effect on scores) | 10 | q02, q04, q06, q09, q16, q26, q28, q34, q37, q47 | left as they are (see "Not applied") |
| Grading problem (changes a score) | 6 | q07, q08, q13, q20, q29, q32 | **fixed** |

None of the not-covered topics is actually taught. Three are touched on in passing: the RL practical computes a mean/std-normalised advantage (close to GRPO), the extended Lecture 11 deck names PPO without explaining it, and L1/L2 only show up in code as `L1Loss` and `weight_decay`. All three are still fair near-misses.

The fixes only change the automatic metric and the citation scores. The hand-checked scores, the strict reading, the significance tests and the ranking of the systems stay the same, because the hand labels had already judged these answers by reading them.

## Human review

On 27 November 2026, Sowris Kumar Sripathy reviewed the 16 questions that this review flagged: the six grading problems (q07, q08, q13, q20, q29, q32) and the ten wording-only notes (q02, q04, q06, q09, q16, q26, q28, q34, q37, q47). Sowris agreed with all six fixes and with leaving the ten wording-only questions as they are. No other changes were made.

## How the review was done

For the answerable questions (q01–q38), the review found the passage that teaches each answer and compared it with `gold_lectures` / `gold_files`, the reference answer and each key fact. It also searched the whole corpus for each question's term, to catch teaching outside the expected location.

For the not-covered questions (q39–q50), it ran case-insensitive regex searches over all 538 documents, including variants, abbreviations and related wording.

To check the scorer, it recomputed key-fact recall for every saved answer with the project's own matcher, and got exactly the saved values. Finally, for each proposed fix it re-scored all 13 saved runs and listed every answer whose verdict would change.

## Fixes applied

| Question | Problem | Evidence (course text) | Fix | Saved answers affected |
|---|---|---|---|---|
| **q29** top-k sampling | The key facts rejected the course's own explanation | GPT practical, exercises: *"Top-k truncates the long tail of garbage words before sampling"* | 2nd fact: `-inf`, `exclude`, `discard`, `remov*`, `truncat*`, `long tail`, `only the k` (this also drops the loose `inf`, `zero` and `only`) | 7 Llama + course runs: correct (automatic) goes from 0 to 1 |
| **q20** normalizing flow | The core idea (change of variables) wasn't a key fact | Lecture 7, slide 7: *"construct a change of variables… to some simple distribution (e.g. Gaussian)… inverse is simple… Jacobian is tractable"* | `[change of variables, transform*] [gaussian, simple distribution, normal distribution] [invert*, inverse, jacobian]` | dense: 0 to 1 |
| **q07** zero/one/few-shot | Too strict (a correct answer failed), and `an example` was too loose | Lecture 3, slide 8: *"Directly ask without examples… Provide one example… Give multiple examples"* | Added alternatives (`no specific`, `zero examples`, `small set of examples`, `few examples`, `several`) and dropped `an example` | no course: 0 to 1 |
| **q08** compound AI systems | Too loose (`multiple` and `performance` let a generic answer pass) | Lecture 3, slide 12: *"retrievers, external tools… faster than scaling alone… real-time retrieval… static training data"* | `[retriev*, tool] [scaling, real-time, real time, static, training data, knowledge]` | lenient-syllabus baseline: 1 to 0 |
| **q13** shared CNN weights | The expected location missed Lecture 6, which explains *why* | Lecture 6, slide 16: *"share weights across spatial positions… recognized no matter where it is located"* | `gold_lectures: [4, 6]` | relevance-check-off run: q13 now counts as citing the right lecture |
| **q32** Q-learning | Labelled a paraphrase, but it uses the course's term and wording | RL practical: *"No model required"*, then *"This is why Q-learning…"* | `type: keyword` (now 31 keyword, 7 paraphrased) | none (it only moves between subgroups) |
| **Matcher** | Matched the start of words only, so `wer` matched "were", `uv` "uvicorn", `inf` "information" and `norm` "normal" | `fact_present` in `src/study_assistant/evaluation.py` | Whole words by default (allowing -s/-es/-ed/-d/-ing), with stems marked by `*` (55 phrases) | no verdict changed, but hidden false matches are gone |

Here is the effect on the automatic main metric, with the same answers re-scored:

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

The strict reading still counts q29 as *borderline*, meaning not correct. Fixing the key facts corrects the automatic metric, but it doesn't change our judgement that the answer never says sampling is limited to the k most likely tokens. So the strict scores (main 92%, dense 87%) are unchanged.

## Not applied, and why

None of these affects a saved score. Changing a reference answer would also make the judge's scores for that question out of date.

| Question | Note |
|---|---|
| q02 | The key fact for an intrinsic hallucination also matches the wording for an extrinsic one (`contradict*`). The reference says "contradicts / wrongly summarizes", while the course only says "wrongly summarized". This is why q02 is borderline in the strict reading. |
| q04 | The key facts check the definition, but not *why* the Markov assumption is used. |
| q06, q16, q28 | Weak paraphrases: they reuse the slide's distinctive wording ("self-driving car… noisy sensor", "lower-dimensional surface", attention in an image GAN). |
| q09 | The reference leaves out "environmental impact" (Lecture 3, slide 23). |
| q26 | The reference describes value embeddings in standard terms rather than the course's ("gives the relative weight"). |
| q34 | `generat*` also matches "Generative AI", and `reward` matches "Reward Modeling"; a tighter list would be fairer. |
| q37 | Just writing out "Retrieval-Augmented Generation" satisfies 3 of the 5 stage facts. |
| q47 | The source note should say that the RL practical partly covers advantages (though not GRPO's group baseline). |

## Not-covered questions (q39–q50)

| id | Label | Searched for | Result |
|---|---|---|---|
| q39 label smoothing | near-miss | `label.?smooth`, `smooth(ed\|ing)? label`, `soft (label\|target)` | no hits (only n-gram "smoothing") |
| q40 PPO's clipped objective | near-miss | `PPO`, `proximal`, `clip… objective/ratio`, `surrogate`, `trust region` | PPO is named (extended Lecture 11 deck, slides 11, 14, 29), but there is no clipped objective |
| q41 VQ-VAE codebook | near-miss | `VQ`, `vector quanti`, `codebook`, `discrete latent` | no hits (only 4-bit model quantization) |
| q42 Stable Diffusion latent space | near-miss | `stable diffusion`, `latent diffusion`, `LDM` | no hits; the diffusion practical works on pixels |
| q43 mixture of experts | near-miss | `mixture of experts`, `MoE`, `gating network`, `load balanc` | no hits ("router" is the LangGraph router) |
| q44 FID | near-miss | `FID`, `Fréchet`, `inception` | no hits ("Inception" is a movie in a toy dataset) |
| q45 L1 vs L2 regularization | near-miss | `L1`, `L2`, `regulari[sz]`, `weight decay`, `lasso`, `ridge` | not explained; only `nn.L1Loss`, `weight_decay=…` and dropout |
| q46 DPO | near-miss | `DPO`, `direct preference`, `preference optim` | no hits (RLHF reward models are covered) |
| q47 GRPO advantages | near-miss | `GRPO`, `group relative`, `advantage`, `baseline` | no GRPO; the RL practical normalises an advantage |
| q48 sourdough | off-topic | `sourdough`, `dough`, `yeast`, `bak(e\|ing)` | only "bread" in a toy vocabulary |
| q49 Everest boiling point | trap | `everest`, `boil` | Everest is in toy RAG data, and "boil" appears nowhere |
| q50 Wimbledon | off-topic | `wimbledon`, `tennis` | no hits |

## Coverage, overlap and difficulty

**Coverage.** Every lecture has at least one question. Lecture 7 (normalizing flows) has only one and Lecture 11 has two; the others have three to five. The add-on labs have one each (evals, LangGraph, the RAG primer and Whisper fine-tuning). Most topics without an evaluation question are covered by the 35 development questions instead (VAE reparameterization, GAN losses, LSTM, Real NVP, the diffusion process, Langevin sampling, Bellman, the RLHF reward model, LoRA, BLEU and others). Neither set covers temperature sampling, dropout and batch norm, credit assignment, REINFORCE, the DistilBERT fine-tuning lab, the Docker/FastAPI activity or multimodal prompting.

**Overlap.** There are no duplicates, and no question repeats a development question. A few share a topic with a development question (q25 positional encoding, q20 normalizing flows, q24 diffusion noising, q35 LLM evaluation, q33/q34 RLHF). Since we chose hybrid retrieval on the development set, these may be slightly easier for it.

**Difficulty.** Most questions are lookups in a single passage: about 30 of the 38 answers sit in one slide or notebook cell that uses the question's own term, and no question needs two lectures combined. The report already notes that the paraphrased questions were easier than the development ones. A harder set would add comparisons across lectures (for example, how VAEs, GANs and diffusion models each generate images) and paraphrases that avoid the slide's wording as well as its term.

## All 50 questions

| id | type | verdict | status |
|---|---|---|---|
| q01 | keyword | fine | no change |
| q02 | keyword | wording only | not applied (borderline in strict reading) |
| q03 | keyword | fine | `uv` no longer matches "uvicorn" |
| q04 | keyword | wording only | not applied |
| q05 | keyword | fine | no change |
| q06 | paraphrase | wording only | not applied |
| q07 | keyword | grading problem | **fixed** |
| q08 | keyword | grading problem | **fixed** |
| q09 | keyword | wording only | not applied |
| q10–q12 | keyword | fine | no change |
| q13 | paraphrase | grading problem | **fixed** (gold [4, 6]) |
| q14, q15 | keyword, paraphrase | fine | no change |
| q16 | paraphrase | wording only | not applied |
| q17–q19 | keyword, keyword, paraphrase | fine | `norm` no longer matches "normal" (q17) |
| q20 | keyword | grading problem | **fixed** |
| q21–q25 | keyword | fine | no change |
| q26 | keyword | wording only | not applied |
| q27 | keyword | fine | no change |
| q28 | paraphrase | wording only | not applied |
| q29 | keyword | grading problem | **fixed** (still borderline in strict reading) |
| q30, q31 | paraphrase, keyword | fine | no change |
| q32 | keyword (was paraphrase) | grading problem | **fixed** (relabelled) |
| q33 | keyword | fine | no change |
| q34 | keyword | wording only | not applied |
| q35, q36 | keyword | fine | no change |
| q37 | keyword | wording only | not applied |
| q38 | keyword | fine | `wer` no longer matches "were" |
| q39–q46 | not covered | fine | no change |
| q47 | not covered | wording only | not applied |
| q48–q50 | not covered | fine | no change |
