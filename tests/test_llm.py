"""LLM backends: logic tested with fake clients (no keys or network needed); live Ollama test if running."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from study_assistant.llm import LLM, LLMResponse, LLMUnavailable, RetryableError, get_llm
from study_assistant.llm.gemini_llm import GeminiLLM
from study_assistant.llm.ollama_llm import OllamaLLM
from study_assistant.llm.openai_llm import OpenAILLM

# --- shared retry logic ---------------------------------------------------------


class FlakyLLM(LLM):
    name = "flaky"
    base_delay_s = 0.0

    def __init__(self, failures: int):
        super().__init__("fake")
        self.failures, self.calls = failures, 0

    def _generate(self, prompt, system, max_tokens, temperature):
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError("rate limited")
        return LLMResponse(f"echo: {prompt}", self.name, self.model, 0.0)


def test_retries_then_succeeds():
    llm = FlakyLLM(failures=2)
    assert llm.generate("hi").text == "echo: hi"
    assert llm.calls == 3


def test_gives_up_after_max_retries():
    llm = FlakyLLM(failures=99)
    with pytest.raises(RetryableError):
        llm.generate("hi")
    assert llm.calls == llm.max_retries + 1


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_llm("nope")


# --- OpenAI -----------------------------------------------------------------------


class FakeOpenAI:
    def __init__(self, reject_temperature=False):
        self.calls, self.reject_temperature = [], reject_temperature
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            resp = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com"))
            raise openai.BadRequestError("Unsupported parameter: 'temperature'", response=resp, body=None)
        msg = SimpleNamespace(content="  An answer.  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3))


def test_openai_builds_messages_and_reads_usage():
    fake = FakeOpenAI()
    r = OpenAILLM("test-model", client=fake).generate("Q?", system="Be brief.", max_tokens=50)
    call = fake.calls[0]
    assert call["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Q?"}]
    assert call["model"] == "test-model" and call["max_completion_tokens"] == 50 and call["temperature"] == 0.0
    assert (r.text, r.input_tokens, r.output_tokens, r.backend) == ("An answer.", 12, 3, "openai")


def test_openai_drops_temperature_for_models_that_reject_it():
    fake = FakeOpenAI(reject_temperature=True)
    llm = OpenAILLM("reasoning-model", client=fake)
    assert llm.generate("Q?").text == "An answer."
    assert "temperature" not in fake.calls[-1]
    llm.generate("again")  # remembered: no failing call the second time
    assert len(fake.calls) == 3


def test_openai_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        OpenAILLM()


# --- Gemini -------------------------------------------------------------------------


def test_gemini_passes_system_prompt_and_reads_usage():
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text="Gemini says hi", usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=4))

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    r = GeminiLLM("gemini-test", client=fake).generate("Q?", system="Be brief.", max_tokens=64)
    cfg = calls[0]["config"]
    assert calls[0]["model"] == "gemini-test" and calls[0]["contents"] == "Q?"
    assert cfg.system_instruction == "Be brief." and cfg.max_output_tokens == 64 and cfg.temperature == 0.0
    assert (r.text, r.input_tokens, r.output_tokens) == ("Gemini says hi", 7, 4)


def test_gemini_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        GeminiLLM()


# --- Ollama (live, only if running) --------------------------------------------------------


def _ollama_ready() -> bool:
    try:
        OllamaLLM()
        return True
    except LLMUnavailable:
        return False


def test_ollama_unreachable_host_is_unavailable():
    with pytest.raises(LLMUnavailable, match="not running"):
        OllamaLLM(host="http://localhost:1")


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama with llama3.1:8b not running")
def test_ollama_live():
    r = get_llm("ollama").generate("Reply with exactly the word: pong", max_tokens=10)
    assert "pong" in r.text.lower()
    assert r.output_tokens and r.latency_s > 0
