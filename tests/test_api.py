"""The HTTP API, with a scripted fake LLM (no Ollama or keys needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from study_assistant import prompts
from study_assistant.api import create_app
from study_assistant.assistant import StudyAssistant
from study_assistant.chunking import chunk_documents
from study_assistant.ingest.schema import Document
from study_assistant.llm import LLM, LLMResponse, LLMUnavailable, RetryableError
from study_assistant.retrieval import build_retriever


class FakeLLM(LLM):
    name = "fake"

    max_retries, base_delay_s = 1, 0.0  # don't sleep in tests

    def __init__(self, answer="Max pooling keeps the largest value in each window [1].", fail=None):
        super().__init__("fake-model")
        self.answer, self.fail = answer, fail  # fail: an exception to raise when answering

    def _generate(self, prompt, system, max_tokens, temperature):
        if self.fail:
            raise self.fail
        text = "RELEVANT: 1" if system == prompts.GRADER_SYSTEM else self.answer
        return LLMResponse(text, self.name, self.model, 0.0)


@pytest.fixture(scope="module")
def retriever():
    doc = Document(doc_id="pool.pptx::slide-1", text="Max pooling keeps the largest value in each window.",
                   source_file="pool.pptx", source_type="slides", title="Deck pool", section="pool",
                   locator="slide 1", lecture=4, track="lecture")
    return build_retriever("dense", chunk_documents([doc]))


def client(retriever, llm) -> TestClient:
    return TestClient(create_app(StudyAssistant(llm, retriever, k=3, max_passages=3, off_topic_floor=-1.0)))


def test_ask_returns_answer_with_sources(retriever):
    r = client(retriever, FakeLLM()).post("/ask", json={"question": "What does max pooling do?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "answer" and not body["abstained"] and "[1]" in body["answer"]
    assert body["sources"] == [{"number": 1, "label": "Lecture 4 · Deck pool, slide 1", "lecture": 4,
                                "doc_id": "pool.pptx::slide-1"}]
    assert "trace" not in body  # only on request


def test_trace_on_request_and_not_covered(retriever):
    r = client(retriever, FakeLLM(answer=prompts.NOT_COVERED)).post(
        "/ask", json={"question": "What is mode collapse?", "trace": True})
    body = r.json()
    assert body["abstained"] and body["answer"] == prompts.NOT_COVERED and body["sources"] == []
    assert any(line.startswith("route") for line in body["trace"])


def test_quiz_and_explain_use_their_modes(retriever):
    c = client(retriever, FakeLLM(answer="Q1 ... Answer: B [1]"))
    assert c.post("/quiz", json={"topic": "pooling"}).json()["mode"] == "quiz"
    assert c.post("/explain", json={"topic": "pooling"}).json()["mode"] == "simplify"


def test_bad_input_and_unavailable_model(retriever):
    c = client(retriever, FakeLLM())
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.get("/health").json()["status"] == "ok"
    r = client(retriever, FakeLLM(fail=LLMUnavailable("Ollama is not running"))).post(
        "/ask", json={"question": "What does max pooling do?"})
    assert r.status_code == 503 and "Ollama is not running" in r.json()["detail"]


def test_model_dropping_mid_answer_gives_503(retriever):
    llm = FakeLLM(fail=RetryableError("connection refused"))  # e.g. Ollama stopped after the server started
    r = client(retriever, llm).post("/ask", json={"question": "What does max pooling do?"})
    assert r.status_code == 503 and "connection refused" in r.json()["detail"]


class StoppedOllama(FakeLLM):
    def _check_available(self):
        raise LLMUnavailable("Ollama is not running at http://localhost:11434. Start the Ollama app.")


def test_health_and_ask_check_the_model_is_still_reachable(retriever):
    c = client(retriever, StoppedOllama())
    assert c.get("/health").status_code == 503
    assert c.post("/ask", json={"question": "What does max pooling do?"}).status_code == 503


@pytest.mark.parametrize("value,expected", [("off", "off"), ("on", "concept"), ("Strict", "strict")])
def test_verify_setting(retriever, monkeypatch, value, expected):
    monkeypatch.setenv("VERIFY", value)
    assert client(retriever, FakeLLM()).get("/health").json()["verify"] == expected


def test_bad_verify_setting_fails_at_startup(monkeypatch):
    monkeypatch.setenv("VERIFY", "maybe")
    with pytest.raises(ValueError, match="VERIFY"):
        create_app()
