# Course Study Assistant — Applied Generative AI (Final Project)

A study assistant that answers questions **only from our course materials** (lecture slides, practical notebooks, activity PDFs), **cites the lecture and slide/page** each answer came from, and **says "this isn't covered in the course"** instead of guessing. It can also explain a topic more simply and generate a short practice quiz.

Built with the tools from class: RAG (embeddings + vector search) and a LangGraph workflow, evaluated on ~50 questions with known answers against a chatbot that can't see the materials.

Team: Sowris Kumar Sripathy, Alfonso Garcia. The original proposal is in [`docs/proposal.png`](docs/proposal.png).

## Status

| Step | What | Status |
|---|---|---|
| 0 | Project scaffold, GitHub | ✅ |
| 1 | Ingestion: slides / notebooks / PDFs → documents with citation metadata | ✅ |
| 2 | Chunking, vector index, retrieval (TF-IDF vs dense vs hybrid) | ⏳ next |
| 3 | LLM backends: OpenAI, Gemini, Ollama (Llama 3.1), flan-t5 | |
| 4 | LangGraph assistant (answer / simplify / quiz / abstain) + CLI | |
| 5 | Evaluation set (50 Qs) and harness: RAG vs no-RAG | |
| 6 | Iterate on failures, write-up | |
| later | FastAPI + Docker | |

Design decisions and their reasons are logged in [`docs/design.md`](docs/design.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
git clone <this repo>
cd <repo>
uv sync                      # creates .venv with all dependencies
cp .env.example .env         # then edit COURSE_DIR (see below)
```

**Course materials are not in this repo**: they are the instructor's files. Put your copy of the class files folder anywhere and point `COURSE_DIR` in `.env` at it. The default `../Class Files` works if the folder sits next to this repo.

Rules that keep it that way:
- `.gitignore` blocks `.pptx`/`.pdf` files, `data/` (extracted text, OCR cache, indexes) and `.env`.
- Notebooks are committed **without outputs**, because outputs print course text. Clear them before committing:
  `uv run --with nbconvert jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
  `tests/test_repo_hygiene.py` fails if you forget.

**macOS + iCloud Desktop/Documents sync:** iCloud marks files inside `.venv` as hidden, and Python then ignores the project's install (`ModuleNotFoundError: No module named 'study_assistant'`). Keep the environment out of iCloud with a `.nosync` folder:

```bash
rm -rf .venv && uv venv .venv.nosync --python 3.12 && ln -s .venv.nosync .venv && uv sync
```

## Usage

```bash
uv run python -m study_assistant.ingest     # Step 1: build data/processed/documents.jsonl
uv run pytest                               # run the tests
```

The first ingest OCRs one image-only PDF (`module10_rl_slides.pdf`, ~1 minute); the result is cached in `data/cache/ocr/`.

Notebooks in `notebooks/` walk through each step. In VS Code, pick the `.venv` interpreter as the kernel, or run `uv run --with jupyter jupyter lab`.

## Layout

```
src/study_assistant/
  config.py              paths + settings (reads .env)
  ingest/                Step 1: loaders for .pptx / .ipynb / .pdf (+ OCR), catalog of files -> lectures
notebooks/01_ingest_explore.ipynb   what's in the knowledge base, quality checks
tests/                   unit tests (synthetic files) + corpus checks (skipped if COURSE_DIR missing)
docs/                    proposal, design log
data/                    generated files (git-ignored)
```
