"""Google Gemini (free tier available; needs GEMINI_API_KEY in .env).

Model: GEMINI_MODEL, default gemini-3.6-flash. Free-tier keys are rate-limited per
minute; rate-limit errors (HTTP 429) are retried with exponential backoff.
"""

from __future__ import annotations

import os

from .base import LLM, LLMResponse, LLMUnavailable, RetryableError

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiLLM(LLM):
    name = "gemini"
    base_delay_s = 10.0  # free-tier limits are per minute: wait 10, 20, 40, 80 s

    def __init__(self, model: str | None = None, client=None):
        super().__init__(model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL)
        if client is None:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise LLMUnavailable("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com and add it to .env.")
            from google import genai

            client = genai.Client(api_key=key)
        self.client = client

    def _generate(self, prompt, system, max_tokens, temperature) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens, temperature=temperature
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.APIError as exc:
            if getattr(exc, "code", None) in (429, 500, 503):
                raise RetryableError(str(exc)) from exc
            raise

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=(resp.text or "").strip(),
            backend=self.name,
            model=self.model,
            latency_s=0.0,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
