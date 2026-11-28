# Design log

This is where we wrote down what we decided at each step and why. The report ([`report.md`](report.md), Sections 3–7) is built from it.

## Goals

These come from the proposal, with a little more detail:

1. Answer students' questions **using only the course materials**.
2. **Cite the source precisely**: the lecture plus the slide, notebook section or PDF page.
3. **Say "not covered in the course"** instead of guessing.
4. Offer two extra modes: **explain more simply**, and **make a short practice quiz**.
5. **Evaluate** it on about 50 questions with known answers (correctness, citation accuracy, declining when it should) and compare it with a chatbot that can't see the course.

We also went a bit further than the proposal. Citations point to a location, not just a file name. Chunking follows the structure of the material. We compared retrieval methods, including a hybrid one. Citations are checked rather than trusted. And we tried several language models: OpenAI, Gemini, a local Llama 3.1 through Ollama, and flan-t5.

## Step 1: Ingestion

### What goes into the knowledge base

We included 11 lecture decks (`.pptx`), 2 extended decks that only exist as images (`module10_rl_slides.pdf` for Lecture 10 and `module11_rl_genai_slides.pdf` for Lecture 11), 28 notebooks (the practicals and the agent/LLM add-on labs), 6 activity PDFs and 1 setup-guide PDF. That is 48 files, which become 538 documents. We missed the extended Lecture 11 deck at first and only added it after the audit (see "Audit and final re-run").

We left out our own assignment submissions (`Assignment_*.docx`), because the assistant should answer from what was taught, not from what we wrote. We also left out the `shared/*.py` helper code, which supports the labs but isn't teaching material.

### What counts as one document

Each document is the material's natural unit, so that a citation points somewhere a student would actually look:

| Source | Unit | Why |
|---|---|---|
| Slides | one slide | Students find their way around a lecture by slide, and a slide usually holds one idea. |
| Notebooks | one section: an explanation plus the code after it | A section starts at a heading, or at a markdown cell that follows code. Splitting only at headings left some notebooks with 2 or 3 sections of over 20,000 characters. |
| PDFs | one page | A continuation page takes the heading of the page before, e.g. "GAN Class Activity (continued)". |

### Which lecture each notebook belongs to

The practical notebooks are numbered differently from the lectures. `Module_4_Practical_VAE.ipynb`, for example, goes with the **Lecture 5** VAE slides, and `Module_8_Practical_GPT.ipynb` goes with **Lecture 9** (Transformers). We compared each notebook with the slide titles of every deck and wrote the mapping out by hand in `ingest/catalog.py` (`NOTEBOOK_LECTURE`), so the question "which lecture is this from?" gets the right answer.

| Notebook(s) | Lecture |
|---|---|
| Module_2_Practical_1–3 | 2: Basics of GenAI (probability, sampling, embeddings) |
| M3_Prompting_Strategies | 3: Applications of GenAI |
| Module_3_Practical_1–3 (FCNN, convolutions, CNN) | 4: Modern DL Part 1 |
| Module_4_Practical_VAE | 5: Modern DL Part 2 (autoencoders, VAE) |
| Module_6_Practical_1–2 (GAN, RNN) | 6: Modern DL Part 3 |
| Module_7_Practical_1–2 (EBM, diffusion) | 8: Energy-based and diffusion models |
| Module_8_Practical_GPT | 9: Transformers and advanced GANs |
| Module_9_Practical_RL | 10: Basics of RL |
| Module_10_RL_and_LLM | 11: RL applied to GenAI |
| RAG / LangGraph / ReAct / evals / fine-tuning labs | add-on labs (no single lecture) |

Lecture 7 (Normalizing Flows) has only slides, 10 of them, and no practical, so it is the thinnest topic.

### Problems we ran into while extracting text

| Problem | What we did |
|---|---|
| Slide-number fields (`‹#›`) showed up in the slide text | Skip the slide-number, date and footer placeholders |
| Some slides put their body text inside the title placeholder | Use the first line as the title and the rest as the body |
| pypdf removed the spaces in code (`importtorch.nnasnn`) | Switched to **PyMuPDF** |
| The PDF reading order put every code listing after all the headings | `get_text(sort=True)`, which orders text by position |
| PDFs carried the course website's header, sidebar, URL and page counter | Remove lines that repeat at the top or bottom of most pages, plus regexes for URLs, counters and the sidebar prefix |
| The extended decks `module10_rl_slides.pdf` and `module11_rl_genai_slides.pdf` have **no text layer** (28 and 31 images) | OCR with **RapidOCR**, which installs with pip alone; results are cached by file hash |
| OCR read two callout boxes that sit side by side as one block, mixing their sentences | Rebuild the layout from the OCR boxes: group lines into blocks (same left edge, small gap), then order the blocks by row and from left to right |
| Notebooks contained Marimo export notices and `pip install` lines | Dropped |
| Title and section-divider slides have no content | Kept, but flagged `low_content`, so the index skips them and they can't outrank real content |

### What we know is lost

Equations and diagrams that are images don't survive (`extra.n_images` flags those slides); captioning them with a vision model would be a possible extension. OCR only gets subscripts and Greek letters roughly right. Code from PDFs loses its indentation, though the notebooks keep the exact code. Notebook outputs are dropped.

### Numbers

There are 538 documents: 273 slides, 250 notebook sections, 11 activity pages and 4 guide pages. 43 of them are flagged as low-content, which leaves 495 to index. A typical slide is about 510 characters and a typical notebook section about 1,470 (medians). 173 documents are longer than roughly 1,000 characters, which is about where `all-MiniLM-L6-v2` stops reading (256 word-pieces). That is why Step 2 needs real chunking.

## Step 2: Chunking and retrieval

### Chunking (`chunking.py`)

`all-MiniLM-L6-v2` reads at most 256 tokens and quietly ignores the rest, and 185 of the 249 notebook sections were longer than that (some reached 4,800 tokens). So we split documents at paragraph and code-block boundaries and pack the pieces up to 254 tokens, counted with the model's own tokenizer. A paragraph that is too long by itself is split by lines, and then by words if needed. Neighbouring chunks share one short paragraph (at most 60 tokens) as overlap.

The text we embed for each chunk starts with a `title › section` header, so a piece of code from the middle of a notebook still carries its topic. Low-content documents are skipped. The result is **1,237 chunks** (970 from notebooks, 237 from slides, 18 from activities and 12 from the guide). The tests confirm that none is over the limit and that no text goes missing.

### Retrievers (`retrieval.py`)

| Method | How it's built |
|---|---|
| TF-IDF | scikit-learn, unigrams and bigrams, sublinear tf, English stop words |
| BM25 | `rank-bm25` (Okapi), with the same tokenisation |
| Dense | `all-MiniLM-L6-v2`, normalised embeddings and cosine similarity (embeddings cached in `data/cache/`) |
| Hybrid | Reciprocal Rank Fusion of the BM25 and dense rankings (k = 60, top 50 from each) |

Every hit also carries its dense cosine score, so the assistant always has one consistent similarity number to work with.

### Checking retrieval (`eval/retrieval_queries.jsonl`, notebook 02)

We wrote 35 development questions and checked each answer key by searching the knowledge base. 20 are keyword questions that use the topic's term, 10 are paraphrased questions that describe the idea without naming it, and 5 are out of scope, including the near-miss "mode collapse". A hit counts if it comes from the right lecture or file.

| hit@5 | keyword (20) | paraphrase (10) | all (30) | MRR@10 (all) |
|---|---|---|---|---|
| TF-IDF | 1.00 | 0.60 | 0.87 | 0.84 |
| BM25 | 1.00 | 0.80 | 0.93 | 0.86 |
| Dense | 1.00 | **1.00** | **1.00** | 0.90 |
| Hybrid | 1.00 | 0.80 | 0.93 | **0.92** |

These numbers are on the final knowledge base. Before the extended Lecture 11 deck was added, paraphrase hit@5 was 0.50, 0.80, 0.90 and 0.70; the conclusions don't change.

What we learned:

- **Keyword search only works if the student already knows the course's term.** Dense retrieval handles paraphrases best.
- **Hybrid didn't beat dense on paraphrases.** On the LoRA question, BM25 and dense each found a different part of the same notebook section, so neither part collected enough votes after fusion. One fix to try in Step 6 would be to merge votes per section before fusing.
- **A similarity threshold can't tell when something isn't covered.** "Capital of France" scores 0.43, which is higher than the weakest real question (0.37), because the extended Lecture 11 deck uses it as an example prompt. The near-miss "mode collapse" scores 0.48, higher than 6 of the 30 real questions. So in Step 4 we added an LLM relevance check before answering, and kept the threshold only for dropping questions that are obviously off-topic.
- **We don't tune on these 30 questions**, since we wrote them while building the system. The default retriever is chosen on the Step 5 evaluation instead.

The set is small, so one question moves the numbers by 3 to 10 points, and relevance is judged by lecture, not by slide.

## Step 3: LLM backends

Every model sits behind one interface, `LLM.generate(prompt, system=None, max_tokens, temperature=0) -> LLMResponse`, and the response carries the text, the model name, the latency and the token counts. The assistant and the evaluation only ever call `generate`, so switching models means changing one word.

| Backend | Default model | Why it's here |
|---|---|---|
| OpenAI | `gpt-5.6-luna` (the cheapest text model) | a strong cloud model, and the class labs used the OpenAI SDK |
| Gemini | `gemini-3.6-flash` | has a free tier, so the whole project can run at no cost |
| Ollama | `llama3.1:8b` | local, private and free, and it was already installed |
| flan-t5 | `google/flan-t5-base` | the model from the class RAG notebooks, used as a weak baseline |

Model names change often, so the defaults can be overridden in `.env` (`OPENAI_MODEL` and so on).

A few choices worth explaining:

- **Temperature is 0 everywhere**, and Ollama also gets a fixed seed, so evaluation runs are as repeatable as each backend allows. Re-running notebook 03, and a 2-question evaluation, gave the same answers. Long free-form answers can still change slightly between runs (we saw it once in the "simplify" cell of notebook 04), so the exact wording isn't guaranteed.
- **Rate limits and overloads are retried with exponential backoff.** Gemini's free tier has a per-minute limit, so it waits 10, 20, 40 and then 80 seconds; the other backends wait 2, 4, 8 and 16.
- **OpenAI's reasoning models reject `temperature`.** When the API says so, we drop it and remember that for later calls.
- **flan-t5 reads at most 512 tokens**, so we keep the end of long prompts, which is where the question is.
- **A missing key or a stopped Ollama** raises `LLMUnavailable` with a message saying how to fix it, and `python -m study_assistant.llm --check` lists what works on the current machine.

First impressions from notebook 03 (just one question, so only anecdotal; Step 5 measures this properly): Llama 3.1 answering from memory gives generic answers, and whether they're right depends on the wording. One phrasing claimed the VAE latent is discrete, which is wrong. With course context it answers in the course's own terms and can cite them. flan-t5 can't answer from memory at all, and even with context its answers are a few words long.

## Step 4: The assistant (LangGraph)

`assistant.py` is a LangGraph `StateGraph`, following the class's `campus_help_desk_rag.ipynb`, with the nodes **route → retrieve → check → generate → cite** plus **abstain** and **chitchat**. The prompts live in `prompts.py`.

| Node | What it does | Why |
|---|---|---|
| route | Works out the mode with simple rules (quiz, simplify, chitchat or answer) and strips request words like "quiz me on" from the search query. | Cheap and predictable, like the router in class. Only explicit phrases switch to simplify, so a question about "simple RNNs" stays a normal question. |
| retrieve | Takes the top 8 chunks (dense retrieval at this step; hybrid became the default in Step 5) and groups them into at most 5 passages, one per document. | Each source is cited only once, and the model sees chunks from the same document together. |
| check | If the best cosine score is below 0.25, declines without calling the model. Otherwise makes **one** LLM call that grades all the passages at once (`RELEVANT: 1, 3`). | Step 2 showed a threshold alone can't separate covered from not covered. One call instead of one per passage keeps it fast. If the reply can't be parsed, all passages are kept, because the next step can still decline. |
| generate | Uses a system prompt for the mode: use only the passages, cite `[n]`, and reply with exactly the "not covered" sentence if the passages aren't enough. | Keeps answers grounded and gives a second chance to decline. |
| cite | Drops citations to passages that weren't shown, renumbers `[n]` in order of use, and maps each one to "Lecture N · title, slide/section". A "not covered" reply becomes a decline, and an answer with no citations is flagged. | Citations are checked, not trusted. |

Some smaller decisions:

- **Citation labels show the lecture number** ("Lecture 5 · Module 4: Practical – VAE …"), because the notebooks' own titles use different module numbers.
- **flan-t5 uses a similarity threshold (0.45) instead of the LLM check**, since it can't follow the grader's format.
- **Ollama's context window is raised to 8,192 tokens.** Its default is small and silently cuts long prompts.

First impressions with Llama 3.1 8B (7 examples in notebook 04; Step 5 measures this properly): covered questions get answers that cite the right slide, and the quiz and simplify modes work and cite their sources too. **The 8B grader isn't reliable on questions the course doesn't cover.** For both "mode collapse" and "capital of France" it marked an irrelevant passage as relevant, and it was the answering prompt that declined, which is why that second layer matters. We also got **a wrong "not covered" on a paraphrased RLHF question**: retrieval never brought up Lecture 11 (it ranked 6th in Step 2), so the assistant rightly refused to answer from the wrong material. Retrieval recall is the thing to watch.

## Step 5: Evaluation set-up

### The questions (`eval/questions.jsonl`, readable copy in `eval/QUESTIONS.md`)

We wrote 50 new questions. None of them repeats one of the 35 development questions, and a test checks that.

| Type | n | Purpose |
|---|---|---|
| Answerable, uses the course's term | 30 (31 after the question review relabelled q32) | Spread over Lectures 1–11 and the add-on labs |
| Answerable, paraphrased | 8 (7 after the review) | The student doesn't know the term, which Step 2 showed is where retrieval struggles |
| Not covered, near-miss | 9 | Sounds like course content but isn't in it: vanishing gradients (replaced by label smoothing after the audit), PPO (now its clipped objective), VQ-VAE, Stable Diffusion, MoE, FID, L1/L2, DPO, GRPO |
| Not covered, trap | 1 | Everest's boiling point: "Everest" appears in toy demo data, but the answer doesn't |
| Not covered, off-topic | 2 | Sourdough and Wimbledon |

Every reference answer was written from the slide or notebook text it cites, not from memory. Each answerable question also has key facts, which are lists of acceptable phrasings, and a test checks that each key fact really appears in the right lecture's text. Another test checks that no not-covered topic appears anywhere in the course. Those checks caught two mistakes in the first draft: "Mount Everest" and "Pride and Prejudice" both show up in toy knowledge bases inside class notebooks. We kept Everest as a deliberate trap and replaced Pride and Prejudice. We also dropped one question because its answer (the normalizing-flow loss) only exists as a formula image, so the text can't support it.

### Metrics (`evaluation.py`, using the tiers from the class `llm_evals` notebook)

For declining, we measure the share of not-covered questions answered with "not covered" (near-misses reported on their own) and the share of answerable questions wrongly refused. Correctness is measured over all answerable questions, and a refusal counts as wrong. It has two tiers: key-fact recall, which is deterministic (Tier 1), and an LLM judge scoring 0, 1 or 2 against the reference answer (Tier 4). For citations (RAG only) we check whether the answer cites the right lecture or lab, and what share of its cited sources are right.

The baseline is the same model with a system prompt saying it is the course's study assistant. It may say "not covered", but it can't see the materials.

### Smoke test (5 questions, not results)

The RAG assistant answered the near-miss "vanishing gradient" question with a `[1]` citation to a Lecture 4 passage that never mentions it; both chances to decline missed it, and the full run would show how often that happens. The no-course chatbot refused the question about the MIT/Stanford study, which only the course material can answer. And the judge (Llama 3.1 8B) gave full marks to a no-course answer that missed the course's main point (convergence), even though its key-fact recall was 0.5. Judges can be lenient, which is why we report key-fact recall next to them. Using the same model as both answerer and judge is a bias too, and a different judge model would be better.

## Step 5: First results (before the audit)

> **Superseded.** These are the first results, kept to show how decisions were made. The raw answers are in `eval/archive/results_before_audit/`. The audit later found problems with the data and the scoring, and every number was measured again (see "Audit and final re-run").

We ran five configurations on all 50 questions: Llama 3.1 8B with the course (hybrid and dense retrieval) and without it, and flan-t5-base with and without it. The judge was Llama 3.1 8B, as the team chose.

| | Llama + course (hybrid) | Llama + course (dense) | Llama, no course | flan-t5 + course | flan-t5, no course |
|---|---|---|---|---|---|
| Correct (main metric) | **97%** | 82% | 55% | 32% | 0% |
| Key-fact recall | 96% | 92% | 72% | 35% | 2% |
| False refusals | 0% | 5% | 3% | 5% | 50% |
| Not covered, answered "not covered" | **92%** | 83% | 33% | 67% | 58% |
| Near-miss, answered "not covered" | **90%** | 80% | 20% | 60% | 70% |
| Cites the right lecture | 97% | 100% | n/a | (2 cited answers) | n/a |
| Seconds per question (M2, local) | 14 | 13 | 7 | 1 | 1 |

### The judge needed fixing

The first judge (v1) only saw the reference answer, so it marked down correct course details that the short reference left out. We checked six of these and all six are in the course text: the diffusion `denoise(noisy_images, noise_rates, signal_rates)` inputs, the "lower-triangular" causal mask, "conditional edges" and three others. That biased the comparison against RAG.

The second judge (v2) also gets the 3 most relevant excerpts from the right lecture, the same for every system. That fixed those cases and still caught the two answers we knew were wrong, **but it is too generous with one-word answers** (flan-t5's "squeeze" got full marks). So the **main metric counts an answer as correct when v2 says so and at least half the key facts are present.** We still report v1 and v2.

The judge is also inconsistent: the same wrong clause scored 1 in one run and 2 in another. With an 8B judge, a difference of 1 to 3 questions is noise, so we also read the dense-versus-hybrid comparison by hand.

We found one bug as well. flan-t5 declines with a bare "unanswerable", a habit from its training data, and the refusal detector missed it. We fixed the detector and relabelled the one affected record without running the model again.

### Decisions

We **switched the default retriever to hybrid.** Reading the 8 questions where dense and hybrid differ, hybrid wins 4 and loses 1, and the other 3 were judge noise; it also declined one more near-miss. The catch is that we made this choice on the evaluation set, so hybrid's numbers are slightly optimistic.

**Retrieval isn't the bottleneck on this set.** Every method has hit@3 = 100%, and the paraphrased evaluation questions turned out easier than the development ones.

### The main failure: confident answers with citations that don't support them

On "vanishing gradients" (both retrievers) and "Stable Diffusion" (dense), the relevance check accepted loosely related passages, and Llama answered from its own knowledge while citing them. The citations make those answers look trustworthy. Two fixes seemed worth trying: a stricter relevance prompt or a stronger grader model, or a sentence-by-sentence support check after generation.

The other errors were one subtly wrong definition (q02), generic answers that missed the course's specific point (q19, q20), and over-cautious refusals when the right material was in hand (dense: q17, q30).

### Limitations at this point

There were only 38 answerable and 12 not-covered questions, so a single question is worth 2.6 or 8 points. The judge was the same 8B model as the answerer, hybrid was picked on this set, and no one outside had reviewed the questions.

## Step 6: Support check (before the audit)

> **Partly wrong, corrected below.** The development-set result here assumed all 27 answered development questions were correct. Labelling them by hand showed 3 were wrong, which reverses the comparison between the two variants (see "Audit and final re-run").

The goal was to fix the Step 5 failure where an answer cites passages that don't support it. We added a `verify` node after `cite`: the model judges whether the main content of the answer is stated in the passages, and if not, the assistant declines. It only applies to plain answers, since simplify adds analogies and quizzes add wrong options on purpose.

All the trying-out happened on the 35 development questions. The experiment worked on saved answers: retrieval is deterministic and the relevance verdicts are in the traces, so the passages could be rebuilt exactly (the script checks this). We fixed the variant to apply to the evaluation set ("strict") in advance. The script and its outputs are in `eval/support_check/`.

The development set turned up two bugs. First, the check has to see every passage shown, not only the cited ones, or imprecise citations cause false rejections. Second, the "not covered" sentence has to be detected anywhere in a reply, because Llama sometimes hedged with general knowledge before saying it. That second bug was general, but it didn't affect any saved Step 5 result.

Results: on development, it wrongly rejected 3 of 27 correct answers (11%), and so did a "concept" variant, with nothing out of scope left to catch. On the evaluation set's dense run it removed 1 of the 2 unsupported answers to not-covered questions (Stable Diffusion) and 2 answers we knew were wrong, but also 1 correct answer. On the hybrid run it removed 1 flawed answer the judge had passed, and missed "vanishing gradients".

So we left it **off by default**, available as `verify=True` or `chat --verify`. With an 8B checker it trades coverage for precision rather than being a clear win. A stronger checker, or checking sentence by sentence, would be the next thing to try.

The final report is `docs/report.md`, with figures in `docs/figures/`.

## Audit and final re-run

After Step 6 we asked an AI model (Claude, in a separate session that had no part in the earlier decisions) to review the whole project without changing anything, looking for errors in the data, the scoring and the write-up. This was an AI-assisted review, not an independent human one. It found real problems, summarised below, and we fixed them and re-ran every evaluation. A second review later found four more (see "Second review"). Every number in `docs/report.md` and notebook 05 comes from that final run (`eval/results/RUN_INFO.md`).

### What was wrong, and how we fixed it

| Problem | Fix |
|---|---|
| **The extended Lecture 11 deck wasn't in the knowledge base.** It names PPO, which one "not covered" question asked about. Without a catalogue entry it would also have been labelled a "guide". | Added the catalogue entry and re-ingested (48 files, 538 documents, 1,237 chunks). q40 now asks about PPO's *clipped objective*, which the deck doesn't explain. |
| **q39 ("vanishing gradients and residual connections") was half covered**: the GPT practical says residuals keep gradients flowing. Our headline failure example was partly a grounded answer to a mislabelled question. | Replaced it with "label smoothing", which appears nowhere in the course (tested), and rewrote the failure analysis around dense's Stable Diffusion answer. |
| **The headline 97% included an answer we knew was wrong**, because we had only hand-checked the questions where dense and hybrid differed. | Re-read every failed answer in every Llama run by hand, plus every pass in the main and no-course runs (`eval/hand_labels.json`), and reported both the automatic and the hand-checked scores. |
| **The relevance check never declined a question**, so "two independent layers for declining" wasn't true. | Said so, and ran the evaluation with the check turned off: same results. |
| **The instructions for reproducing the results re-ran nothing** (results resume by file name) and covered only 2 of the 5 configurations. | `scripts/run_evaluation.sh` runs every configuration and then judge v2, and records the code version, the model digest and timings. |
| **Some key facts were loose**: they matched inside other words, or just repeated the question. | Matched at word starts (later changed to whole words in the question review), tightened 11 key-fact lists, and added a test that no key fact repeats its question. |
| **Citation handling broke code**: `outputs[0]` was read as a citation, and renumbering could crash. | Citations are only parsed outside code spans (tested). |
| **The assistant and the evaluation had two different refusal detectors** that disagreed on hedged refusals. | One shared detector, `refusal.py`. |
| Out-of-date documentation, notebook outputs showing local paths, and missing notebook dependencies. | Fixed, and the notebooks were re-run on the final knowledge base. |

The results from before the audit are in `eval/archive/results_before_audit/`. The conclusions point the same way, but the numbers moved: hybrid scores 95% on the automatic metric and 97% by hand, and the no-course baseline 47% and 61%. (Those automatic figures are from the audit; the later question review moved them to 97% and 50%.)

### Statistics

With 38 answerable and 12 not-covered questions, we now report 95% Wilson intervals and exact McNemar tests (`eval/analysis.py`). Only course versus no course is significant (p < 0.001). Hybrid versus dense isn't (4 against 1 disagreements, p = 0.38).

### The support check, looked at again

**The Step 6 development result was wrong.** It counted every rejected development answer as a lost correct answer. The hand labels (`eval/support_check/dev_labels.json`) show that strict rejected 2 correct answers and 1 of the 3 wrong ones, while concept rejected all 3 wrong ones and no correct ones. So concept is the better variant, chosen on development data only, and we added it as `verify="concept"`.

**On the evaluation set neither one helps**, because the main system already declines all 12 not-covered questions. Strict rejected q02, a borderline answer (correct on our main reading, wrong on the strict one), and concept changed nothing. Both add about 4 seconds per answer. So it stays **off by default**, since the default system is the one we evaluated, and `chat --verify` uses the concept variant.

## Making the evaluation stronger (after the audit)

The audit also suggested some cheaper ways to make the results more convincing, and we did them.

We started an **experiment log**: every run adds a row per build and configuration to `eval/experiment_log.csv`, with the commit, knowledge-base size, question-set hash, model digest, and the automatic and hand-checked scores. Notebook 06 charts it, and the build from before the audit is filled in from the archive.

We **split course-specific questions from general-knowledge ones.** Five questions need this course's own material, and the course still helps a lot on the other 33 (97% against 70% by hand, p = 0.012). We sorted the questions after seeing the results, so the audit's own six-question list is reported as a check.

We gave the baseline a fairer shot by **letting the plain model see the syllabus**: the lecture and lab titles with their section titles, built from the knowledge base by `course_outline()`. With a strict instruction it refused 32 of the 38 covered questions (11% correct, every not-covered question declined). With a lenient one it got 42% right and declined 83%. Either way it handles about half of the 50 questions or fewer, against 98% for the assistant. We report both wordings, since helping the baseline is the safe direction to err in.

We ran **more ablations.** Turning the similarity floor off changed no results (of the two off-topic questions, the answer prompt caught one and the relevance check the other). Using 3 instead of 8 passages cost accuracy, and using more cost time. Nothing beat the default by a significant margin.

We also added **real sessions** to the report (`docs/examples.md`), drew the workflow as an image (`docs/figures/make_workflow.py`) so it survives PDF export, pinned the embedding model to a fixed Hugging Face revision, and embedded the query once per search instead of twice.

What's still open needs people or keys: an independent judge (a Gemini key), a second human grader with an agreement score, and a review of the questions by someone outside the team. Sowris Kumar Sripathy has reviewed the 16 questions flagged by the question review and agreed with its fixes (see `eval/QUESTION_REVIEW.md`).

## Second review

A second AI-assisted review (Claude, in a separate session) went over the finished project again. It confirmed the numbers, the traces and the not-covered labels, and found four problems:

| Problem | Fix |
|---|---|
| **The public repo includes the instructor's course text** (`documents.jsonl`, and quotes inside saved answers), while the report only said that "the raw course files" aren't shipped. | The report now says plainly that the extracted text is included. The team decided to keep the repo public. |
| **The reproduce script didn't reproduce anything.** It wrote to `eval/results/`, so it reused the saved answers, overwrote `RUN_INFO.md` and logged the same numbers as a new build. Hand labels would also have quietly applied to new, different answers. | The script now writes to a new `eval/runs/<date>-<commit>/` folder and refuses to reuse results unless `RESUME=1`, and `analysis.py --results` analyses any run. Each hand label stores a hash of the answer it judged and only applies to that exact answer (`hand_labels.py`, tested). |
| **The headline 97% leaned on two generous labels** (q02's general definition of an intrinsic hallucination, and q29's "truncates the long tail"). | Both are marked *borderline* everywhere they appear, and a **strict reading** that counts them as wrong is reported next to every hand-checked number: 92% for the main system and 87% for dense. Course versus no course is still significant (p < 0.001); the comparison on general-knowledge questions alone isn't (p = 0.065). |
| **The first audit was called "independent"**, but it was an AI review. | It's now described as an AI-assisted review, and the report has a note on AI use (Section 8). |

## Step 7: HTTP API and Docker

This was the last item in the proposal.

**The API** (`src/study_assistant/api.py`) uses FastAPI, following the class's `sps_genai` pattern. It has `POST /ask`, `/explain` and `/quiz`, plus `GET /health`, with typed request and response models. Each reply carries the answer, whether it declined and why, an `uncited` warning, and the numbered sources (lecture and slide), and it can include the workflow trace on request.

The assistant is built on the first request rather than at import, so the server starts even when the model can't be reached yet, and it answers with a 503 and the reason instead of crashing. It handles one question at a time behind a lock: a local model answers one at a time anyway, and the graph isn't guaranteed to be thread-safe. Settings come from environment variables (`LLM_BACKEND`, `OLLAMA_HOST`, `RETRIEVER`, `VERIFY`), so the same code runs on a laptop and in the container. The tests use a scripted fake model (`tests/test_api.py`), so they don't need Ollama.

**Docker** (`Dockerfile`, `docker-compose.yml`): the image holds the code, the knowledge base and the pinned embedding model. The knowledge base is embedded at build time, so the container starts quickly and needs no internet (`HF_HUB_OFFLINE=1`). The language model isn't in the image. On a Mac, Ollama inside Docker can't use the GPU, so by default the container calls Ollama on the host (`host.docker.internal`). Gemini and OpenAI work with a key, and an optional `ollama` service runs everything inside Docker.

On Linux we install CPU-only PyTorch (`[tool.uv.sources]` in `pyproject.toml`), because the default Linux wheels bring several GB of CUDA libraries that a CPU container never uses; installs on macOS are unchanged. Compose also has its own variable for the Ollama address, `DOCKER_OLLAMA_HOST`, so `OLLAMA_HOST=localhost` in `.env` (right for the CLI) can't leak into the container, where localhost means the container itself.

## Final audit

A last AI-assisted review (Claude, in a separate session) cloned the repo fresh from GitHub with an empty model cache. It confirmed that the project runs with nothing from outside the repo (`uv sync`, all the tests, all six notebooks, the CLI, the API and `docker compose up --build`), that every number in the README and the report matches the analysis output, and that a 2-question evaluation reproduced the saved answers byte for byte.

It found a few things, which we fixed:

- **"What is the capital of France?" gets answered.** The extended Lecture 11 deck uses it as an example prompt, so it gets past both the similarity floor and the relevance check. Fixing that would mean re-running the evaluation, so it's now a stated limitation, and notebook 04 shows it as a known failure.
- **Notebook 04 was out of date**, from before the Lecture 11 deck and the verify node. It now runs on the default system, and its text matches its outputs.
- **"The relevance check never declined" was too broad.** With the similarity floor off, it declined Wimbledon, so the claim now says "with the floor on (the default)".
- **The API's 503 only worked before the first request.** It now checks that the model is reachable before every answer and on `/health`, and turns errors in the middle of an answer into a 503 (tested).
- Some smaller things: `VERIFY` accepts on, off, concept or strict and rejects anything else at startup; the evaluation CLI requires `--out` when answering, and an explicit `--verify` value; the CLI no longer prints library warnings and progress bars; and we fixed some wording and old counts, and reordered the README with its requirements spelled out.

## Question review (after the results)

An AI-assisted review checked all 50 questions against the course text (`eval/QUESTION_REVIEW.md`). It confirmed every not-covered label and found six grading problems, which we fixed. The key facts for q29 and q20 rejected correct answers worded the way the course words them, q07 was too strict and q08 too loose, q13 was missing Lecture 6 as a correct lecture, and q32 was wrongly labelled as a paraphrase. It also found that the key-fact matcher matched the start of words, so "wer" matched "were". Phrases now have to match whole words unless they're marked as stems with `*`.

We **re-scored the saved answers rather than re-running them** (`evaluation --rescore`). The originals are in `eval/archive/before_question_review/`, and the experiment log has a third build, "after question review (re-scored)". On the automatic main metric, hybrid went from 95% to 97%, dense from 79% to 84%, no course from 47% to 50%, and the lenient syllabus baseline from 29% to 26%. The hand-checked and strict scores, every significance test and the ranking stayed the same, and q29 is still borderline on the strict reading. We report both versions, because these corrections were made after we had seen the results.
