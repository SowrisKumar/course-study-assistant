"""Local models through Ollama (free, private, no API key). Default model: llama3.1:8b.

Requires the Ollama app running (https://ollama.com) and the model pulled:
    ollama pull llama3.1:8b
"""

from __future__ import annotations

import os

import httpx

from .base import LLM, LLMResponse, LLMUnavailable, RetryableError

DEFAULT_MODEL = "llama3.1:8b"


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, check: bool = True):
        super().__init__(model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if check:
            self._check_available()

    def _check_available(self) -> None:
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=3).json()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama is not running at {self.host}. Start the Ollama app.") from exc
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names:
            raise LLMUnavailable(f"Model {self.model!r} not pulled. Run: ollama pull {self.model}")

    def _generate(self, prompt, system, max_tokens, temperature) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0},
        }
        try:
            r = httpx.post(f"{self.host}/api/chat", json=payload, timeout=300)
        except httpx.TransportError as exc:
            raise RetryableError(str(exc)) from exc
        if r.status_code >= 500:
            raise RetryableError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"].strip(),
            backend=self.name,
            model=self.model,
            latency_s=0.0,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )
