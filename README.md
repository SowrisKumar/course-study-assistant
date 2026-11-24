# Course Study Assistant — Applied Generative AI (Final Project)

A study assistant that answers questions **only from our course materials** (lecture slides, practical notebooks, activity PDFs), **cites the lecture and slide/page** each answer came from, and **says "this isn't covered in the course"** instead of guessing. It can also explain a topic more simply and generate a short practice quiz.

Built with the tools from class: RAG (embeddings + vector search) and a LangGraph workflow, evaluated on ~50 questions with known answers against a chatbot that can't see the materials.

Team: Sowris Kumar Sripathy, Alfonso Garcia. Original proposal: [`docs/proposal.png`](docs/proposal.png).

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SowrisKumar/course-study-assistant.git
cd course-study-assistant
uv sync
uv run pytest
```

Everything needed is in the repo. The knowledge base (the course materials' text, already extracted and cleaned) is `data/processed/documents.jsonl`. Walk through it in [`notebooks/01_ingest_explore.ipynb`](notebooks/01_ingest_explore.ipynb).

## Status

| Step | What | Status |
|---|---|---|
| 1 | Knowledge base: slides / notebooks / PDFs → 507 documents with citation info | ✅ |
| 2 | Chunking, vector index, retrieval (TF-IDF vs dense vs hybrid) | ⏳ next |
| 3 | LLM backends: OpenAI, Gemini, Ollama (Llama 3.1), flan-t5 | |
| 4 | LangGraph assistant (answer / simplify / quiz / abstain) + CLI | |
| 5 | Evaluation (50 questions): RAG vs no-RAG | |
| 6 | Improvements and write-up | |

Design decisions and their reasons: [`docs/design.md`](docs/design.md).

## Layout

```
data/processed/documents.jsonl   the knowledge base
src/study_assistant/             the code (ingest/ builds the knowledge base)
notebooks/                       step-by-step walkthroughs
tests/                           tests
docs/                            proposal, design log
```
