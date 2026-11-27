# Design log

This file records the decisions we made and why, step by step. It is the raw material for the report (docs/report.md), Sections 3–7.

## Goals (from the proposal, refined)

1. Answer students' questions **using only course materials**.
2. **Cite the source precisely**: lecture + slide / notebook section / PDF page.
3. **Abstain** ("not covered in the course") instead of guessing.
4. Extra modes: **explain more simply** and **make a short practice quiz**.
5. **Evaluate** on ~50 questions with known answers: correctness, citation accuracy, abstention, compared with a no-RAG chatbot.

Improvements over the proposal:
- Location-level citations (not just a file name)
- Chunking that follows the material's structure
- Hybrid retrieval with an ablation study
- A grounding check on citations
- Several LLM backends compared (OpenAI, Gemini, local Llama 3.1 via Ollama, flan-t5)

## Step 1 — Ingestion

### Corpus
- **Included:** 11 lecture decks (`.pptx`), 2 extended decks exported as images (`module10_rl_slides.pdf` for Lecture 10, `module11_rl_genai_slides.pdf` for Lecture 11), 28 notebooks (practicals + agents/LLM add-on labs), 6 activity PDFs, 1 setup-guide PDF. That is 48 files → 538 documents.
- The extended Lecture 11 deck was missed at first and added after the audit (see "Audit and final re-run").
- **Excluded:** our own assignment submissions (`Assignment_*.docx`), because the assistant should answer from what was *taught*, not from our answers. Also excluded: `shared/*.py` helper code (supporting code for the labs, not teaching material).

### Unit of a document
Each document is the material's *natural* unit, so citations are meaningful:

| Source | Unit | Why |
|---|---|---|
| Slides | one slide | Students navigate lectures by slide; a slide is one idea. |
| Notebooks | one section = an explanation + the code that follows it | A section starts at a heading or at a markdown cell after code. Splitting only at headings collapsed some notebooks into 2–3 sections of 20k+ chars. |
| PDFs | one page | Continuation pages inherit the previous heading, e.g. "GAN Class Activity (continued)". |

### Lecture mapping
Practical notebooks are numbered differently from lectures. For example, `Module_4_Practical_VAE.ipynb` is the practical for the **Lecture 5** VAE slides, and `Module_8_Practical_GPT.ipynb` is for **Lecture 9** (Transformers). We compared notebook content with every deck's slide titles and wrote the mapping explicitly in `ingest/catalog.py` (`NOTEBOOK_LECTURE`), so "which lecture is this from?" has the right answer.

| Notebook(s) | Lecture |
|---|---|
| Module_2_Practical_1–3 | 2 – Basics of GenAI (probability, sampling, embeddings) |
| M3_Prompting_Strategies | 3 – Applications of GenAI |
| Module_3_Practical_1–3 (FCNN, convolutions, CNN) | 4 – Modern DL Part 1 |
| Module_4_Practical_VAE | 5 – Modern DL Part 2 (autoencoders, VAE) |
| Module_6_Practical_1–2 (GAN, RNN) | 6 – Modern DL Part 3 |
| Module_7_Practical_1–2 (EBM, diffusion) | 8 – Energy-based and diffusion models |
| Module_8_Practical_GPT | 9 – Transformers and advanced GANs |
| Module_9_Practical_RL | 10 – Basics of RL |
| Module_10_RL_and_LLM | 11 – RL applied to GenAI |
| RAG / LangGraph / ReAct / evals / fine-tuning labs | add-on labs (no single lecture) |

Lecture 7 (Normalizing Flows) has slides only (10 slides, no practical), so it is the thinnest topic.

### Extraction choices and problems found
| Problem found | Fix |
|---|---|
| Slide-number fields (`‹#›`) in slide text | Skip slide-number/date/footer placeholders |
| Body text placed inside the title placeholder | Title = first line, the rest becomes body |
| pypdf removed spaces in code (`importtorch.nnasnn`) | Switched to **PyMuPDF** |
| PDF reading order put every code listing after all the headings | `get_text(sort=True)` (position-based order) |
| Course-website header, sidebar, URL and page-counter lines in PDFs | Remove lines that repeat at the top/bottom of most pages, plus regex for URLs, counters and the sidebar prefix |
| The extended decks `module10_rl_slides.pdf` and `module11_rl_genai_slides.pdf` have **no text layer** (28 and 31 images) | OCR with **RapidOCR** (pip-only, no system install), cached by file hash |
| OCR read two side-by-side callout boxes line by line across both, interleaving sentences | Rebuild the layout from OCR boxes: group lines into blocks (same left edge, small gap), then order blocks by row and left-to-right |
| Marimo export notice cells and `pip install` lines in notebooks | Dropped |
| Section-divider / title slides with no content | Kept but flagged `low_content`; the index will skip them so they don't out-rank real content |

### Known limitations
- Equations and diagrams that are images are lost (`extra.n_images` flags such slides). A possible extension is captioning them with a vision model.
- OCR handles subscripts and Greek letters only approximately.
- PDF code loses its indentation; the notebooks keep exact code.
- Notebook outputs are dropped.

### Numbers
- 538 documents: 273 slides, 250 notebook sections, 11 activity pages, 4 guide pages.
- 43 are flagged low-content, which leaves 495 to index.
- Median length: slides about 510 chars, notebook sections about 1,470 chars.
- 173 documents are longer than about 1,000 chars, the point at which `all-MiniLM-L6-v2` stops reading (256 word-pieces). This drives Step 2's chunking.

## Step 2 — Chunking and retrieval

### Chunking (`chunking.py`)
- `all-MiniLM-L6-v2` reads at most 256 tokens and silently drops the rest. 185 of 249 notebook sections were longer than that (up to 4,800 tokens).
- Documents are split on paragraph and code-block boundaries. Pieces are packed up to 254 tokens, counted with the model's own tokenizer.
- A paragraph that is too long on its own is split by lines, then by words.
- Neighbouring chunks share one short paragraph (≤ 60 tokens) as overlap.
- Every chunk's embedded text starts with a header `title › section`, so a code fragment from the middle of a notebook keeps its topic.
- Low-content documents are skipped.
- Result: **1,237 chunks** (970 notebook, 237 slides, 18 activity, 12 guide). None is over the limit, and no text is lost (tested).

### Retrievers (`retrieval.py`)
| Method | Implementation |
|---|---|
| TF-IDF | scikit-learn, unigrams + bigrams, sublinear tf, English stop words |
| BM25 | `rank-bm25` (Okapi), same tokenisation |
| Dense | `all-MiniLM-L6-v2`, normalised embeddings, cosine (embeddings cached in `data/cache/`) |
| Hybrid | Reciprocal Rank Fusion of the BM25 and dense rankings (k = 60, top 50 of each) |

Every hit also carries its dense cosine score, so the assistant has one consistent similarity number.

### Retrieval check (`eval/retrieval_queries.jsonl`, notebook 02)
We wrote 35 questions and verified each answer key by grepping the knowledge base:
- 20 *keyword* questions, which contain the topic's term
- 10 *paraphrased* questions, which describe the idea without the term
- 5 out-of-scope questions, including the near-miss "mode collapse"

A hit counts if it comes from the gold lecture or file.

| hit@5 | keyword (20) | paraphrase (10) | all (30) | MRR@10 (all) |
|---|---|---|---|---|
| TF-IDF | 1.00 | 0.60 | 0.87 | 0.84 |
| BM25 | 1.00 | 0.80 | 0.93 | 0.86 |
| Dense | 1.00 | **1.00** | **1.00** | 0.90 |
| Hybrid | 1.00 | 0.80 | 0.93 | **0.92** |

(Numbers on the final knowledge base. Before the extended Lecture 11 deck was added, paraphrase hit@5 was 0.50 / 0.80 / 0.90 / 0.70; the conclusions are the same.)

Findings:
- **Keyword search only works when the student already knows the course's term.** Dense retrieval handles paraphrases best.
- **Hybrid (RRF) did not beat dense on paraphrases.** In the LoRA question, BM25 and dense each found a *different part* of the same notebook section, so neither part got enough fused votes. Candidate fix for Step 6: merge votes per section before fusing.
- **A similarity threshold cannot detect "not covered".** "Capital of France" scores 0.43, *above* the weakest real question (0.37), because the extended Lecture 11 deck uses it as an example prompt. The near-miss "mode collapse" scores 0.48, above 6 of the 30 real questions. **Decision:** Step 4 adds an LLM relevance check before answering. The threshold is only used to drop obvious off-topic questions.
- **We don't tune on these 30 questions**, because we wrote them while building the system. The default retriever is chosen on the Step 5 evaluation.

Limitations: the set is small (one question is 3–10 points), and relevance is judged at lecture level, not slide level.

## Step 3 — LLM backends

One interface, `LLM.generate(prompt, system=None, max_tokens, temperature=0) -> LLMResponse`. The response carries the text, the model, the latency and the token counts. The assistant and the evaluation only call `generate`, so changing backend is one word.

| Backend | Default model | Why it is included |
|---|---|---|
| OpenAI | `gpt-5.6-luna` (cheapest text model) | strong cloud model; the class labs used the OpenAI SDK |
| Gemini | `gemini-3.6-flash` | free tier, so the whole project can run at no cost |
| Ollama | `llama3.1:8b` | local, private, free; already installed |
| flan-t5 | `google/flan-t5-base` | the model from the class RAG notebooks; weak baseline |

Model names change often, so defaults can be overridden in `.env` (`OPENAI_MODEL`, ...).

Choices:
- **temperature = 0** (and a fixed seed for Ollama) everywhere, so evaluation runs are as reproducible as the backend allows. Re-running notebook 03, and a 2-question evaluation, gave identical answers. Some long free-form answers can still vary slightly between runs (seen once in notebook 04's "simplify" cell), so exact wording is not guaranteed.
- **Retries with exponential backoff** on rate limits and overloads. Gemini's free tier is limited per minute, so it waits 10 / 20 / 40 / 80 s; the others wait 2 / 4 / 8 / 16 s.
- **OpenAI reasoning models reject `temperature`**: if the API says so, we drop it and remember that for later calls.
- **flan-t5 reads at most 512 tokens**: we keep the *end* of long prompts, where the question is.
- **Missing keys or a stopped Ollama** raise `LLMUnavailable` with a how-to-fix message. `python -m study_assistant.llm --check` lists what works on the current machine.

First observations (notebook 03, one question, so anecdotal; Step 5 measures it properly):
- Llama 3.1 from memory gives generic answers whose correctness depends on wording. One phrasing said the VAE latent is discrete, which is wrong. With course context it answers in the course's own terms and can be cited.
- flan-t5 can't answer from memory, and even with context it gives answers of a few words.

## Step 4 — The assistant (LangGraph)

`assistant.py` is a `StateGraph` (class pattern: `campus_help_desk_rag.ipynb`) with nodes **route → retrieve → check → generate → cite**, plus **abstain** and **chitchat**. Prompts are in `prompts.py`.

| Node | What it does | Why |
|---|---|---|
| route | Rule-based mode detection: quiz / simplify / chitchat / answer. Removes the request words ("quiz me on") from the search query. | Cheap and predictable, like the class router. Only explicit phrases trigger simplify, so "simple RNNs" stays a normal question. |
| retrieve | Top 8 chunks (dense retriever at this step; hybrid became the default in Step 5), grouped into at most 5 passages, one per document | Each source is cited once, and the model sees chunks of the same document together |
| check | Off-topic floor (best cosine < 0.25 means abstain with no LLM call), then **one** LLM call grading all passages (`RELEVANT: 1, 3`) | Step 2 showed thresholds can't separate in-scope from out-of-scope. One call instead of one per passage keeps it fast. If the reply can't be parsed, all passages are kept (fail open), because the next layer can still abstain. |
| generate | Mode-specific system prompt. Only the passages, cite `[n]`, reply exactly with the NOT_COVERED sentence if they're insufficient | Grounding plus a second abstention layer |
| cite | Drops citations to passages that weren't shown, renumbers `[n]` in order of use, maps them to "Lecture N · title, slide/section". A NOT_COVERED reply becomes an abstention; an answer with no citations is flagged. | Citations are checked, not trusted |

Other choices:
- **Citation labels show the *lecture*** ("Lecture 5 · Module 4: Practical – VAE …"), because the notebooks' own titles have different module numbers.
- **flan-t5 uses a similarity threshold (0.45) instead of the LLM check**, since it can't follow the grader format.
- **Ollama context window raised to 8,192 tokens.** Its default is small and silently cuts long prompts.

First observations with Llama 3.1 8B (7 examples in notebook 04; measured properly in Step 5):
- In-scope questions get answers citing the right slide. Quiz and simplify modes work and cite their sources.
- **The 8B grader is unreliable on out-of-scope questions.** For both "mode collapse" and "capital of France" it marked one irrelevant passage as relevant. The answering prompt still abstained, so the second layer is needed.
- **A false "not covered" on a paraphrased RLHF question.** Retrieval never surfaced Lecture 11 (it ranked 6th in Step 2), so the assistant correctly refused to answer from the wrong material. Retrieval recall is the bottleneck to watch.

## Step 5 — Evaluation (set-up)

### Question set (`eval/questions.jsonl`, readable copy in `eval/QUESTIONS.md`)
50 new questions; none are reused from the 35 development queries (this is tested).

| Type | n | Purpose |
|---|---|---|
| Answerable, uses the course's term | 30 (31 after the question review relabelled q32) | Spread over Lectures 1–11 and the add-on labs |
| Answerable, paraphrased | 8 (7 after the review) | The student doesn't know the term (Step 2 showed this is where retrieval struggles) |
| Not covered, near-miss | 9 | Sounds like course content but isn't in it: vanishing gradients (replaced by label smoothing after the audit), PPO (now: its clipped objective), VQ-VAE, Stable Diffusion, MoE, FID, L1/L2, DPO, GRPO |
| Not covered, trap | 1 | Everest's boiling point: "Everest" appears in toy demo data, but the answer doesn't |
| Not covered, off-topic | 2 | Sourdough, Wimbledon |

How the set was built and checked:
- Every reference answer was written from the cited slide or notebook text, not from memory.
- Each answerable question has *key facts* (lists of acceptable phrasings). A test checks that every key fact appears in the gold lecture's text.
- A test also checks that each not-covered topic appears nowhere in the course.
- The checks found two mistakes in the first draft: "Mount Everest" and "Pride and Prejudice" both appear in toy knowledge bases inside class notebooks. Everest was kept as a deliberate trap; Pride and Prejudice was replaced.
- One question was dropped because its answer lives in a formula *image* (normalizing-flow loss), so the text can't support it.

### Metrics (`evaluation.py`, tiers from the class `llm_evals` notebook)
- **Abstention:** the share of not-covered questions answered "not covered" (reported separately for near-misses), and the share of answerable questions wrongly refused (false refusals).
- **Correctness, measured over all answerable questions** (a refusal counts as not correct):
  - Tier 1: key-fact recall (deterministic).
  - Tier 4: LLM-as-judge, 0/1/2 against the reference answer.
- **Citations (RAG only):** whether the answer cites the right lecture or lab, and what share of its cited sources are right.
- **Baseline ("no-RAG"):** the same LLM with a system prompt saying it's the course's study assistant. It is allowed to say "not covered", but it can't see the materials.

### Smoke test (5 questions; not results)
- The RAG assistant answered the near-miss "vanishing gradient" question with a `[1]` citation to a Lecture 4 passage that doesn't mention it. Both abstention layers missed it. The full run measures how often this happens.
- The no-RAG chatbot refused the MIT/Stanford study question, which only the course material can answer.
- The judge (Llama 3.1 8B) gave full marks to a no-RAG answer that missed the course's main point (convergence), whose key-fact recall was 0.5. Judges can be lenient, which is why key-fact recall is reported alongside. If the judge is the same model as the answerer, that also biases the result; a different judge model is preferable.

## Step 5 — Results (before the audit)

> **Superseded.** These are the first results, kept as a record of how decisions were made. Raw answers are archived in `eval/archive/results_before_audit/`. The audit found problems with the data and the scoring, and every number was re-measured; see "Audit and final re-run" below.

Five configurations were run on all 50 questions: Llama 3.1 8B with the course (hybrid and dense retrieval) and without it, and flan-t5-base with and without the course. The grader is Llama 3.1 8B (chosen by the team).

| | Llama + course (hybrid) | Llama + course (dense) | Llama, no course | flan-t5 + course | flan-t5, no course |
|---|---|---|---|---|---|
| Correct (main metric) | **97%** | 82% | 55% | 32% | 0% |
| Key-fact recall | 96% | 92% | 72% | 35% | 2% |
| False refusals | 0% | 5% | 3% | 5% | 50% |
| Not covered → "not covered" | **92%** | 83% | 33% | 67% | 58% |
| Near-miss → "not covered" | **90%** | 80% | 20% | 60% | 70% |
| Cites the right lecture | 97% | 100% | – | (2 cited answers) | – |
| Seconds / question (M2, local) | 14 | 13 | 7 | 1 | 1 |

### The grader had to be fixed: a finding in itself
- **Judge v1** (reference answer only) marked down correct course details that the short reference omitted. Six of these were checked and every one is in the course text: the diffusion `denoise(noisy_images, noise_rates, signal_rates)` inputs, the "lower-triangular" causal mask, "conditional edges", and three others. This biased the comparison against RAG.
- **Judge v2** adds the 3 most relevant excerpts of the gold lecture to the grader's prompt, identically for every system. It fixed those cases and still catches the two known-wrong answers, **but it is too lenient on one-word answers** (flan-t5's "squeeze" got full marks).
- **Main metric = v2 says correct AND ≥ half the key facts are present.** v1 and v2 are still reported.
- **The grader is also inconsistent:** the same wrong clause scored 1 in one run and 2 in another. With an 8B grader, differences of 1–3 questions are noise, so the dense-vs-hybrid comparison was also read by hand.
- **Bug found:** flan-t5 declines with a bare "unanswerable", a convention from its training data, which the refusal detector missed. It was fixed, and the one affected record was re-labelled without re-running the model.

### Decisions
- **Default retriever changed to hybrid.** Reading the 8 questions where dense and hybrid differ by hand: hybrid wins 4 and loses 1; the other 3 were grader noise. It also refused one more near-miss. *Caveat:* this choice was made on the evaluation set, so hybrid's numbers are slightly optimistic.
- **Retrieval isn't the bottleneck on this set.** Every method has hit@3 = 100%. The paraphrased evaluation questions were easier than the development ones.

### Main failure for Step 6: confident answers with unsupporting citations
"Vanishing gradients" (both retrievers) and "Stable Diffusion" (dense): the relevance check accepted loosely related passages, and Llama answered from its own knowledge while citing those passages. Citations make this look trustworthy. Candidate fixes:
- a stricter relevance prompt or a stronger grader model
- a per-sentence support check after generation

Other errors:
- one subtly wrong definition (q02)
- generic answers missing the course's specific point (q19, q20)
- over-cautious refusals with the right material in hand (dense: q17, q30)

### Limitations
- 38 + 12 questions (one question is 2.6 / 8 points)
- the grader is the same 8B model as the answerer
- hybrid was selected on this set
- the questions were not independently reviewed

## Step 6 — Support check (before the audit)

> **Partly wrong; corrected below.** The development-set conclusion below treated every one of the 27 answered development questions as correct. Labelling them by hand showed 3 were wrong, which reverses the comparison of the two variants (see "Audit and final re-run").

**Goal:** remove the Step 5 failure where an answer cites passages that don't support it.

**Built:** a `verify` node after `cite`. The LLM judges whether the answer's main content is stated in the passages; if not, the assistant abstains. It applies only to plain answers, because simplify adds analogies and quizzes add distractors on purpose.

**Method:**
- All trying-out used the 35 development questions.
- The experiment ran on *saved* answers. Retrieval is deterministic and relevance verdicts are in the traces, so passages were rebuilt exactly; the script asserts this.
- The variant applied to the evaluation set ("strict") was fixed in advance. Script and outputs: `eval/support_check/`.

**Bugs found on the development set:**
1. The check must see all passages shown, not only the cited ones; otherwise imprecise citations caused false rejections.
2. The "not covered" sentence must be detected anywhere in a reply, because Llama sometimes hedged with general knowledge first. This was a general bug; it affected no saved Step 5 result.

**Results:**
- **Development:** it wrongly rejected 3 of 27 correct answers (11%), as did a "concept" variant. There was nothing out-of-scope left to catch.
- **Evaluation, dense run:** removed 1 of 2 unsupported out-of-scope answers (Stable Diffusion) and 2 known-wrong answers, plus 1 correct answer.
- **Evaluation, hybrid run:** removed 1 flawed answer the grader had passed, and missed "vanishing gradients".

**Decision: off by default; available as `verify=True` / `chat --verify`.** It is a precision-for-coverage trade-off with an 8B checker, not a clear win. A stronger checker model, or checking sentence by sentence, is the next step.

**Final report:** `docs/report.md` (figure in `docs/figures/`).

## Audit and final re-run

After Step 6 we asked an AI model (Claude, in a separate session with no stake in the earlier decisions) for a read-only review of the whole project, looking for errors in the data, the scoring and the write-up. It is an AI-assisted review, not an independent human one. Its findings are summarised in the table below; a second review later found four more problems (see "Second review"). It found real problems; we fixed them and re-ran every evaluation. All numbers in `docs/report.md` and notebook 05 come from that final run (`eval/results/RUN_INFO.md`).

### What was wrong, and the fix
| Problem | Fix |
|---|---|
| **The extended Lecture 11 deck was not in the knowledge base.** It names PPO, which a "not covered" question asked about. Without a catalogue entry it would also have been labelled a "guide". | Catalogue entry, re-ingest (48 files, 538 documents, 1,237 chunks). q40 now asks about PPO's *clipped objective*, which the deck does not explain. |
| **q39 ("vanishing gradients and residual connections") was half covered:** the GPT practical says residuals keep gradients flowing. Our headline failure example was partly a grounded answer to a mislabelled question. | Replaced with "label smoothing", which appears nowhere in the course (tested). Failure analysis rewritten around dense's Stable Diffusion answer. |
| **The headline 97% included an answer we knew was wrong** (we had hand-checked only where dense and hybrid differed). | Every failed answer in every Llama run re-read by hand, plus all passes of the main and no-course runs (`eval/hand_labels.json`). Both automatic and hand scores reported. |
| **The relevance check never declined a question**, so "two independent abstention layers" was not true. | Said so, and ran an ablation with the check off: same results. |
| **The reproduce instructions re-ran nothing** (results resume by file name) and covered 2 of 5 configurations. | `scripts/run_evaluation.sh` runs every configuration, then judge v2, and records code version, model digest and timings. |
| **Key facts were loose:** some matched inside other words or just repeated the question. | Word-start matching (later replaced by whole-word matching in the question review), 11 key-fact lists tightened, a test that no key fact repeats the question. |
| **Citation handling broke code:** `outputs[0]` was read as a citation, and renumbering could crash. | Citations are only parsed outside code spans (tested). |
| **Two refusal detectors** (assistant and evaluation) disagreed on hedged refusals. | One shared detector, `refusal.py`. |
| Stale documentation, notebook outputs showing local paths, missing notebook dependencies. | Fixed; notebooks re-executed on the final knowledge base. |

The pre-audit results are archived in `eval/archive/results_before_audit/`. The conclusions did not change direction, but the numbers did: hybrid is 95% by the automatic metric and 97% by hand, the no-course baseline 47% and 61% (automatic figures as of the audit; the later question review moved them to 97% and 50%).

### Statistics
With 38 + 12 questions we now report 95% Wilson intervals and exact McNemar tests (`eval/analysis.py`). Only course vs no course is significant (p < 0.001); hybrid vs dense is not (4 vs 1 disagreements, p = 0.38).

### Support check, re-assessed
- **The Step 6 development result was wrong.** It counted every rejected development answer as a correct answer lost. Hand labels (`eval/support_check/dev_labels.json`) show strict rejected 2 correct answers and 1 of the 3 wrong ones; concept rejected all 3 wrong ones and no correct ones. So concept is the better variant, chosen on development data only, and it was added as `verify="concept"`.
- **On the evaluation set neither helps:** the main system already declines all 12 not-covered questions. Strict rejected q02, a borderline answer (correct on our main reading, wrong on the strict one); concept changed nothing. Both add about 4 s per answer.
- **Decision: off by default** (the default system is the evaluated one). `chat --verify` uses the concept variant.

## Making the evaluation stronger (after the audit)

The audit also listed cheaper ideas to make the results more convincing. We did these:

- **Experiment log.** Every run is appended to `eval/experiment_log.csv`: one row per build and configuration, with commit, knowledge-base size, question-set hash, model digest, and automatic and hand scores. Notebook 06 charts it. The pre-audit build is backfilled from the archive.
- **Course-specific vs general-knowledge split.** Five questions need this course's own material; the course still helps a lot on the other 33 (97% vs 70% by hand, p = 0.012). Classified after seeing results, so the audit's six-question list is reported as a check.
- **Fairer baseline: the plain model sees the syllabus** (lecture and lab titles with their section titles, built from the knowledge base by `course_outline()`). With a strict instruction it refused 32 of the 38 covered questions (11% correct, 100% declined); with a lenient one, 42% correct and 83% declined. Either way it handles about half the 50 questions or fewer, vs 98% for the assistant. Both wordings are reported: helping the baseline is the conservative direction.
- **More ablations:** similarity floor off (no change in results: of the two off-topic questions, the answer prompt caught one and the relevance check the other), 3 vs 8 passages (less context costs accuracy, more costs time). Nothing beats the default significantly.
- **Real sessions in the report** (`docs/examples.md`), the workflow as an image (`docs/figures/make_workflow.py`) so it survives PDF export, the embedding model pinned to a fixed Hugging Face revision, and the query embedded once per search instead of twice.

Still open, and needing people or keys: an independent grader (Gemini key), a second human grader with an agreement score, and a review of the question set by someone outside the team. (Sowris Kumar Sripathy has reviewed the 16 questions flagged by the question review and agreed with its fixes; see `eval/QUESTION_REVIEW.md`.)

## Second review

A second AI-assisted review (Claude, separate session) re-checked the finished project. It confirmed the numbers, traces and not-covered labels, and found four problems:

| Problem | Fix |
|---|---|
| **The public repo includes the instructor's course text** (`documents.jsonl`, and quotes in saved answers), while the report said only that "the raw course files" don't ship. | The report now says plainly that the extracted text is included. The team decided to keep the repo public. |
| **The reproduce script reproduced nothing.** It wrote to `eval/results/`, so saved answers were reused, `RUN_INFO.md` was overwritten, and the same numbers were logged as a new build. Hand labels would also silently apply to new, different answers. | The script writes to a new `eval/runs/<date>-<commit>/` folder and refuses to reuse results unless `RESUME=1`. `analysis.py --results` analyses any run. Every hand label stores a hash of the answer it judged and applies only to that exact answer (`hand_labels.py`, tested). |
| **The headline 97% rested on two generous labels** (q02's general definition of an intrinsic hallucination, q29's "truncates the long tail"). | Marked *borderline* wherever they occur. A **strict reading** that counts them as wrong is reported next to every hand-checked number: 92% for the main system, 87% for dense. Course vs no course stays significant (p < 0.001); the general-knowledge-only comparison does not (p = 0.065). |
| **The first audit was called "independent".** It was an AI review. | Described as an AI-assisted review, with a note on AI use in the report (Section 8). |


## Step 7: HTTP API and Docker (the proposal's last item)

**API** (`src/study_assistant/api.py`, FastAPI, the class `sps_genai` pattern): `POST /ask`, `/explain`, `/quiz` and `GET /health`, with typed request and response models.
- Each reply carries the answer, `abstained` and its reason, an `uncited` warning, and the numbered sources (lecture + slide). It includes the workflow trace on request.
- **The assistant is built on the first request, not at import.** The server starts even when the model isn't reachable yet, and answers 503 with the reason instead of crashing.
- **One question at a time (a lock).** A local model answers sequentially anyway, and the graph isn't guaranteed to be thread-safe.
- Configuration comes from environment variables (`LLM_BACKEND`, `OLLAMA_HOST`, `RETRIEVER`, `VERIFY`), so the same code runs locally and in the container.
- **Tests use a scripted fake model** (`tests/test_api.py`), so they need no Ollama.

**Docker** (`Dockerfile`, `docker-compose.yml`):
- **What's in the image:** the code, the knowledge base and the pinned embedding model. The knowledge base is embedded at build time, so the container starts fast and needs no internet (`HF_HUB_OFFLINE=1`).
- **What isn't:** the language model. On a Mac, Ollama inside Docker can't use the GPU, so by default the container calls Ollama on the host (`host.docker.internal`). Gemini/OpenAI work with a key, and an optional `ollama` service runs everything in Docker.
- **CPU-only PyTorch on Linux** (`[tool.uv.sources]` in `pyproject.toml`): the default Linux wheels pull several GB of CUDA libraries a CPU container never uses. macOS installs are unchanged.
- **Separate Docker variable for the Ollama address.** Compose uses its own `DOCKER_OLLAMA_HOST`, so `OLLAMA_HOST=localhost` in `.env` (right for the CLI) can't leak into the container, where localhost is the container itself.

## Final audit

A last AI-assisted review (Claude, separate session) cloned the repo fresh from GitHub, with an empty model cache, and confirmed:
- it runs with nothing outside the repo: `uv sync`, all tests, all six notebooks, the CLI, the API and `docker compose up --build`
- every number in the README and the report matches the analysis output
- a 2-question evaluation reproduced the saved answers byte for byte

It found, and we fixed:
- **"What is the capital of France?" gets answered.** The extended Lecture 11 deck uses it as an example prompt, so it passes the similarity floor and the relevance check. We can't fix this without re-running the evaluation, so it is now a stated limitation, and notebook 04 shows it as a known failure.
- **Notebook 04 was stale** (from before the Lecture 11 deck and the verify node). It now runs on the default system, and its text matches the outputs.
- **"The relevance check never declined" was too broad.** With the similarity floor off, it declined Wimbledon. The claim now says "with the floor on (the default)".
- **The API's 503 only held before the first request.** It now checks the model is reachable before every answer and on `/health`, and turns errors mid-answer into 503 (tested).
- Smaller fixes:
  - `VERIFY` accepts on/off/concept/strict and rejects anything else at startup.
  - The evaluation CLI requires `--out` when answering, and an explicit `--verify` value.
  - The CLI no longer prints library warnings and progress bars.
  - Wording and stale counts fixed; the README reordered, with its requirements spelled out.

## Question review (after the results)

An AI-assisted review of all 50 questions against the course text (`eval/QUESTION_REVIEW.md`) confirmed every not-covered label and found six grading problems, which we fixed:
- q29 and q20 key facts that rejected correct, course-worded answers
- q07 too strict, q08 too loose
- q13 missing Lecture 6 as a gold lecture
- q32 mislabelled as a paraphrase

It also found that the key-fact matcher matched word beginnings ("wer" ~ "were"). Phrases now match as whole words unless marked as stems (`*`).

- **The saved answers were re-scored, not re-run** (`evaluation --rescore`). The originals are archived in `eval/archive/before_question_review/`, and the experiment log has a third build, "after question review (re-scored)".
- **Automatic main metric:** hybrid 95% → 97%, dense 79% → 84%, no course 47% → 50%, lenient syllabus 29% → 26%.
- **Unchanged:** the hand-checked and strict scores, every significance test, and the ranking. q29 stays borderline in the strict reading.
- **Both versions are reported,** because these corrections came after we had seen the results.

