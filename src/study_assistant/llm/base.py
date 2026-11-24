"""The one interface every LLM backend implements.

    llm = get_llm("ollama")
    reply = llm.generate("Explain KL divergence in one sentence.", system="You are a tutor.")
    reply.text, reply.latency_s, reply.input_tokens, reply.output_tokens

The assistant (Step 4) and the evaluation (Step 5) only ever call `generate`, so
swapping OpenAI for Gemini, local Llama or flan-t5 is a one-word change.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The backend cannot be used here (missing API key, server not running, ...)."""


class RetryableError(RuntimeError):
    """A temporary failure (rate limit, overloaded server) worth retrying."""


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(ABC):
    name: str = "base"
    max_retries: int = 4
    base_delay_s: float = 2.0  # waits 2, 4, 8, 16 s between retries

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _generate(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> LLMResponse:
        """Backend-specific call. Raise RetryableError for rate limits / overload."""

    def generate(
        self, prompt: str, system: str | None = None, *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResponse:
        """Generate a reply. temperature=0 by default so answers are reproducible for evaluation."""
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._generate(prompt, system, max_tokens, temperature)
            except RetryableError as exc:
                if attempt == self.max_retries:
                    raise
                delay = self.base_delay_s * 2**attempt
                log.warning("%s: %s; retrying in %.0fs", self.name, exc, delay)
                time.sleep(delay)
                continue
            resp.latency_s = round(time.perf_counter() - start, 3)
            return resp
        raise AssertionError("unreachable")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
