"""OpenAI chat models (needs OPENAI_API_KEY in .env). Model: OPENAI_MODEL, default gpt-5.6-luna."""

from __future__ import annotations

import os

from .base import LLM, LLMResponse, LLMUnavailable, RetryableError

DEFAULT_MODEL = "gpt-5.6-luna"


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, model: str | None = None, client=None):
        super().__init__(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMUnavailable("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self._send_temperature = True  # some reasoning models reject `temperature`; we learn that once

    def _generate(self, prompt, system, max_tokens, temperature) -> LLMResponse:
        import openai

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.model, messages=messages, max_completion_tokens=max_tokens)
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            if self._send_temperature and "temperature" in str(exc):
                self._send_temperature = False
                return self._generate(prompt, system, max_tokens, temperature)
            raise
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as exc:
            raise RetryableError(str(exc)) from exc

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            backend=self.name,
            model=self.model,
            latency_s=0.0,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
