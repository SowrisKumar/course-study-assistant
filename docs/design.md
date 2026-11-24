# Design log

This file records the decisions we made and why, step by step. It is the raw material for the report's "Method" section.

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
- **Included:** 11 lecture decks (`.pptx`), 1 extended Module 10 deck exported as images (`module10_rl_slides.pdf`), 28 notebooks (practicals + agents/LLM add-on labs), 6 activity PDFs, 1 setup-guide PDF. That is 47 files → 507 documents.
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
| `module10_rl_slides.pdf` has **no text layer** (28 images) | OCR with **RapidOCR** (pip-only, no system install), cached by file hash |
| OCR read two side-by-side callout boxes line by line across both, interleaving sentences | Rebuild the layout from OCR boxes: group lines into blocks (same left edge, small gap), then order blocks by row and left-to-right |
| Marimo export notice cells and `pip install` lines in notebooks | Dropped |
| Section-divider / title slides with no content | Kept but flagged `low_content`; the index will skip them so they don't out-rank real content |

### Known limitations
- Equations and diagrams that are images are lost (`extra.n_images` flags such slides). A possible extension is captioning them with a vision model.
- OCR handles subscripts and Greek letters only approximately.
- PDF code loses its indentation; the notebooks keep exact code.
- Notebook outputs are dropped.

### Numbers
- 507 documents: 242 slides, 250 notebook sections, 11 activity pages, 4 guide pages.
- 43 are flagged low-content, which leaves 464 to index.
- Median length: slides about 400 chars, notebook sections about 1,450 chars.
- 171 documents are longer than about 1,000 chars, the point at which `all-MiniLM-L6-v2` stops reading (256 word-pieces). This drives Step 2's chunking.

## Step 2 — Chunking and retrieval

### Chunking (`chunking.py`)
- `all-MiniLM-L6-v2` reads at most 256 tokens and silently drops the rest. 185 of 249 notebook sections were longer than that (up to 4,800 tokens).
- Documents are split on paragraph and code-block boundaries. Pieces are packed up to 254 tokens, counted with the model's own tokenizer.
- A paragraph that is too long on its own is split by lines, then by words.
- Neighbouring chunks share one short paragraph (≤ 60 tokens) as overlap.
- Every chunk's embedded text starts with a header `title › section`, so a code fragment from the middle of a notebook keeps its topic.
- Low-content documents are skipped.
- Result: **1,202 chunks** (971 notebook, 201 slides, 18 activity, 12 guide). None is over the limit, and no text is lost (tested).

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
| TF-IDF | 1.00 | 0.50 | 0.83 | 0.81 |
| BM25 | 1.00 | 0.80 | 0.93 | 0.86 |
| Dense | 0.95 | **0.90** | 0.93 | **0.89** |
| Hybrid | 1.00 | 0.70 | 0.90 | 0.89 |

Findings:
- **Keyword search only works when the student already knows the course's term.** Dense retrieval handles paraphrases best.
- **Hybrid (RRF) did not beat dense on paraphrases.** In the LoRA question, BM25 and dense each found a *different part* of the same notebook section, so neither part got enough fused votes. Candidate fix for Step 6: merge votes per section before fusing.
- **A similarity threshold cannot detect "not covered".** "Capital of France" scores 0.36, while the weakest real question scores 0.37. The near-miss "mode collapse" scores 0.48, above 7 of the 30 real questions. **Decision:** Step 4 adds an LLM relevance check before answering. The threshold is only used to drop obvious off-topic questions.
- **We don't tune on these 30 questions**, because we wrote them while building the system. The default retriever is chosen on the Step 5 evaluation.

Limitations: the set is small (one question is 3–10 points), and relevance is judged at lecture level, not slide level.
