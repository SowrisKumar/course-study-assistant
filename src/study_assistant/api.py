"""HTTP API for the study assistant (FastAPI), following the class `sps_genai` pattern.

    uv run uvicorn study_assistant.api:app --port 8000      # then open http://localhost:8000/docs

Endpoints:
    GET  /          what this is
    GET  /health    is the language model reachable? which model, retriever, knowledge base
                    (indexed_documents: 495 of the 538 documents; title-only slides are not indexed)
    POST /ask       {"question": "..."}  -> answer with sources, or "not covered"
    POST /explain   {"topic": "..."}     -> a simpler explanation of a course topic
    POST /quiz      {"topic": "..."}     -> a 3-question practice quiz

Configuration (environment variables, or .env):
    LLM_BACKEND   ollama (default) | gemini | openai | flan
    OLLAMA_HOST   where Ollama runs (default http://localhost:11434; in Docker http://host.docker.internal:11434)
    RETRIEVER     hybrid (default) | dense | bm25 | tfidf
    VERIFY        off (default) | on (= concept) | concept | strict    the optional support check (report, Section 7)

The assistant is built on the first request, not at import, so the server starts even if the model
is not reachable yet. Before every answer (and on /health) a quick check confirms the model is still
reachable; if not, or if it fails mid-answer, the API replies HTTP 503 with the reason.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException

from . import quiet  # noqa: F401  (clean server logs: no progress bars or library notices)
from pydantic import BaseModel, Field

from .assistant import DEFAULT_RETRIEVER, AssistantResult, StudyAssistant
from .llm import LLMUnavailable, RetryableError, get_llm

VERIFY_SETTINGS = {"off": None, "none": None, "false": None, "0": None,
                   "on": "concept", "true": "concept", "1": "concept", "concept": "concept", "strict": "strict"}


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What does max pooling do?"])
    trace: bool = Field(False, description="also return the workflow trace (one line per step)")


class Topic(BaseModel):
    topic: str = Field(min_length=1, max_length=300, examples=["GANs"])
    trace: bool = False


class Source(BaseModel):
    number: int
    label: str = Field(description="lecture + deck/notebook + slide/section, e.g. 'Lecture 9 · …, slide 7'")
    lecture: int | None
    doc_id: str


class Reply(BaseModel):
    question: str
    mode: Literal["answer", "simplify", "quiz", "chitchat"]
    answer: str
    abstained: bool = Field(description="true when the assistant said the course doesn't cover it")
    abstain_reason: str = ""
    uncited: bool = Field(description="an answer was given but cites nothing (a grounding warning)")
    sources: list[Source]
    latency_s: float
    trace: list[str] | None = None


def _reply(r: AssistantResult, latency: float, with_trace: bool) -> Reply:
    return Reply(question=r.question, mode=r.mode, answer=r.answer, abstained=r.abstained,
                 abstain_reason=r.abstain_reason, uncited=r.uncited, latency_s=round(latency, 2),
                 sources=[Source(number=p.number, label=p.label, lecture=p.lecture, doc_id=p.doc_id) for p in r.sources],
                 trace=r.trace if with_trace else None)


def create_app(bot: StudyAssistant | None = None) -> FastAPI:
    """Build the API. Tests pass a ready-made assistant; the server builds one from the environment."""
    backend = os.getenv("LLM_BACKEND", "ollama")
    retriever_name = os.getenv("RETRIEVER", DEFAULT_RETRIEVER)
    verify_setting = os.getenv("VERIFY", "off").strip().lower()
    if verify_setting not in VERIFY_SETTINGS:
        raise ValueError(f"VERIFY={verify_setting!r}: use off, on, concept or strict")
    verify = VERIFY_SETTINGS[verify_setting]
    state: dict = {"bot": bot}
    lock = threading.Lock()  # one question at a time: a local model answers sequentially anyway

    def assistant() -> StudyAssistant:
        if state["bot"] is None:
            with lock:
                if state["bot"] is None:
                    from .retrieval import build_retriever

                    try:
                        llm = get_llm(backend)
                    except LLMUnavailable as exc:
                        raise HTTPException(503, f"The language model ({backend}) is not available: {exc}") from exc
                    grading = "threshold" if backend == "flan" else "llm"  # flan-t5 can't follow the grader format
                    state["bot"] = StudyAssistant(llm, build_retriever(retriever_name), grading=grading,
                                                  verify=verify)
        return state["bot"]

    def check_reachable(bot: StudyAssistant) -> None:
        """Fast check (a few seconds) that the model still answers, e.g. Ollama wasn't stopped after startup."""
        check = getattr(bot.llm, "_check_available", None)  # Ollama: lists the pulled models
        if check is not None:
            try:
                check()
            except LLMUnavailable as exc:
                raise HTTPException(503, f"The language model ({backend}) is not available: {exc}") from exc

    def ask(question: str, with_trace: bool) -> Reply:
        bot = assistant()
        check_reachable(bot)
        start = time.perf_counter()
        try:
            with lock:
                result = bot.ask(question)
        except (LLMUnavailable, RetryableError, httpx.HTTPError) as exc:  # e.g. the model stopped mid-answer
            raise HTTPException(503, f"The language model failed while answering: {exc}") from exc
        return _reply(result, time.perf_counter() - start, with_trace)

    app = FastAPI(title="Course Study Assistant",
                  description="Answers questions only from the Applied Generative AI course materials, cites the "
                              "lecture and slide, and says when something isn't covered.",
                  version="1.0")

    @app.get("/")
    def root() -> dict:
        return {"message": "Course study assistant. POST /ask, /explain or /quiz; interactive docs at /docs."}

    @app.get("/health")
    def health() -> dict:
        bot = assistant()
        check_reachable(bot)
        chunks = getattr(bot.retriever, "chunks", [])
        return {"status": "ok", "backend": backend, "model": bot.llm.model, "retriever": retriever_name,
                "verify": verify or "off", "chunks": len(chunks),
                "indexed_documents": len({c.doc_id for c in chunks})}

    @app.post("/ask", response_model=Reply, response_model_exclude_none=True)
    def ask_question(q: Question) -> Reply:
        return ask(q.question, q.trace)

    @app.post("/explain", response_model=Reply, response_model_exclude_none=True)
    def explain(t: Topic) -> Reply:
        return ask(f"Explain {t.topic} more simply", t.trace)

    @app.post("/quiz", response_model=Reply, response_model_exclude_none=True)
    def quiz(t: Topic) -> Reply:
        return ask(f"Quiz me on {t.topic}", t.trace)

    return app


app = create_app()
