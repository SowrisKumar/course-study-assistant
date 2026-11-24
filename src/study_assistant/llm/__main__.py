"""CLI: check which LLM backends work, or send one prompt to a backend."""

from __future__ import annotations

import argparse

from . import BACKENDS, available_backends, get_llm


def main() -> None:
    p = argparse.ArgumentParser(description="Try the LLM backends")
    p.add_argument("prompt", nargs="?", help="prompt to send")
    p.add_argument("-b", "--backend", choices=BACKENDS, default="ollama")
    p.add_argument("-m", "--model", help="override the backend's default model")
    p.add_argument("-s", "--system", help="optional system prompt")
    p.add_argument("--check", action="store_true", help="list which backends are usable here")
    args = p.parse_args()

    if args.check or not args.prompt:
        for name, status in available_backends().items():
            print(f"{name:7} {status}")
        return

    r = get_llm(args.backend, args.model).generate(args.prompt, system=args.system)
    print(r.text)
    print(f"\n[{r.backend} · {r.model} · {r.latency_s:.1f}s · {r.input_tokens} in / {r.output_tokens} out tokens]")


if __name__ == "__main__":
    main()
