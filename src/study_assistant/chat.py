"""Chat with the study assistant in the terminal.

    uv run python -m study_assistant.chat                      # Llama 3.1 via Ollama
    uv run python -m study_assistant.chat -b gemini -v         # Gemini, showing each graph step
    uv run python -m study_assistant.chat "What is a VAE?"     # one question, then exit
"""

from __future__ import annotations

import argparse
import warnings

from .assistant import DEFAULT_RETRIEVER, StudyAssistant
from .llm import BACKENDS, LLMUnavailable, get_llm
from .retrieval import RETRIEVERS, build_retriever


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with the course study assistant")
    p.add_argument("question", nargs="?", help="ask one question and exit")
    p.add_argument("-b", "--backend", choices=BACKENDS, default="ollama")
    p.add_argument("-m", "--model", help="override the backend's default model")
    p.add_argument("-r", "--retriever", choices=sorted(RETRIEVERS), default=DEFAULT_RETRIEVER)
    p.add_argument("-v", "--verbose", action="store_true", help="show the steps the assistant took")
    p.add_argument("--verify", action="store_true",
                   help="extra support check after answering: fewer unsupported answers, some correct ones refused")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    try:
        llm = get_llm(args.backend, args.model)
    except LLMUnavailable as exc:
        raise SystemExit(f"Can't use {args.backend}: {exc}")
    # flan-t5 is too small to judge relevance reliably; it uses the similarity threshold instead
    grading = "threshold" if args.backend == "flan" else "llm"
    print(f"Loading course index ({args.retriever}) ...")
    bot = StudyAssistant(llm, build_retriever(args.retriever), grading=grading, verify=args.verify)

    def answer(q: str) -> None:
        r = bot.ask(q)
        if args.verbose:
            print("\n".join(f"  · {line}" for line in r.trace) + "\n")
        print(r.format() + "\n")

    if args.question:
        answer(args.question)
        return

    print(f"Study assistant ready ({llm.name}: {llm.model}). Ask about the course, say 'explain X simply' "
          "or 'quiz me on X'. Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        answer(q)


if __name__ == "__main__":
    main()
