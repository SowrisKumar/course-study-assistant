"""Step 3 – LLM backends behind one interface.

    from study_assistant.llm import get_llm
    llm = get_llm("ollama")            # or "openai", "gemini", "flan"
    print(llm.generate("What is a VAE?", system="Answer in one sentence.").text)

Command line:
    uv run python -m study_assistant.llm --check                      # which backends work here
    uv run python -m study_assistant.llm -b ollama "What is a GAN?"   # try one
"""

from __future__ import annotations

from .. import config  # noqa: F401  (loads API keys from .env)
from .base import LLM, LLMResponse, LLMUnavailable, RetryableError

BACKENDS = ("openai", "gemini", "ollama", "flan")

__all__ = ["BACKENDS", "LLM", "LLMResponse", "LLMUnavailable", "RetryableError", "get_llm", "available_backends"]


def get_llm(backend: str, model: str | None = None) -> LLM:
    """Create a backend. Raises LLMUnavailable with a how-to-fix message if it can't run here."""
    if backend == "openai":
        from .openai_llm import OpenAILLM

        return OpenAILLM(model)
    if backend == "gemini":
        from .gemini_llm import GeminiLLM

        return GeminiLLM(model)
    if backend == "ollama":
        from .ollama_llm import OllamaLLM

        return OllamaLLM(model)
    if backend == "flan":
        from .flan_llm import FlanT5LLM

        return FlanT5LLM(model)
    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")


def available_backends() -> dict[str, str]:
    """backend -> 'ok (model)' or the reason it is unavailable. Does not call any model."""
    status = {}
    for b in BACKENDS:
        try:
            status[b] = f"ok ({get_llm(b).model})"
        except LLMUnavailable as exc:
            status[b] = f"unavailable: {exc}"
    return status
