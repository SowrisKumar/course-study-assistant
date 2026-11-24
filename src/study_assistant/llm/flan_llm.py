"""Google flan-t5 run locally with transformers (the model used in the class RAG notebooks).

Default: google/flan-t5-base (250M params, ~1 GB download on first use); set FLAN_MODEL
to google/flan-t5-small for the faster class version.

It is a *weak baseline*: it reads at most 512 input tokens, so long prompts are cut
(we keep the end, where the question is), and it writes short answers.
"""

from __future__ import annotations

import os
from functools import cached_property

from .base import LLM, LLMResponse

DEFAULT_MODEL = "google/flan-t5-base"
MAX_INPUT_TOKENS = 512


class FlanT5LLM(LLM):
    name = "flan"

    def __init__(self, model: str | None = None):
        super().__init__(model or os.getenv("FLAN_MODEL") or DEFAULT_MODEL)

    @cached_property
    def _model(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # hides a harmless "tied weights" notice for flan-t5
        tok = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
        model.eval()
        torch.manual_seed(0)
        return tok, model

    def _generate(self, prompt, system, max_tokens, temperature) -> LLMResponse:
        import torch

        tok, model = self._model
        text = f"{system}\n\n{prompt}" if system else prompt
        ids = tok(text, return_tensors="pt").input_ids
        truncated = ids.shape[1] > MAX_INPUT_TOKENS
        if truncated:  # keep the end of the prompt: that's where the question is
            ids = torch.cat([ids[:, :1], ids[:, -(MAX_INPUT_TOKENS - 1):]], dim=1)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=min(max_tokens, 256),
                do_sample=temperature > 0,
                **({"temperature": temperature} if temperature > 0 else {}),
            )
        return LLMResponse(
            text=tok.decode(out[0], skip_special_tokens=True).strip(),
            backend=self.name,
            model=self.model,
            latency_s=0.0,
            input_tokens=int(ids.shape[1]),
            output_tokens=int(out.shape[1]),
        )
