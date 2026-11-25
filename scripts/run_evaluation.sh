#!/usr/bin/env bash
# Reproduces every evaluation number in docs/report.md.
# Needs: `uv sync`, Ollama running with `ollama pull llama3.1:8b`. Takes about 2 hours on an M2 laptop.
# Results go to eval/results/ (override with OUT=...); runs resume if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-eval/results}
mkdir -p "$OUT"
INFO="$OUT/RUN_INFO.md"
{
  echo "# Evaluation run"
  echo "- started: $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "- code: git $(git rev-parse --short HEAD)"
  echo "- answer + judge model: $(ollama list | awk '$1=="llama3.1:8b"{print $1" (digest "$2")"}')"
  echo "- machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo
  echo "| configuration | minutes |"
  echo "|---|---|"
} > "$INFO"

run() {  # run <label> <evaluation args...>
  local label=$1; shift
  echo "== $label"
  local start=$(date +%s)
  uv run python -u -m study_assistant.evaluation --out "$OUT" "$@" > /dev/null
  echo "| $label | $(( ($(date +%s) - start) / 60 )) |" >> "$INFO"
}

run "Llama + course, hybrid (main)"            --backend ollama --systems rag --retriever hybrid
run "Llama + course, dense"                    --backend ollama --systems rag --retriever dense
run "Llama, no course"                         --backend ollama --systems no_rag
run "Ablation: relevance check off"            --backend ollama --systems rag --retriever hybrid --grading none
run "Ablation: support check on"               --backend ollama --systems rag --retriever hybrid --verify
run "flan-t5, with and without course"         --backend flan --systems rag,no_rag --retriever hybrid
run "Judge v2 (with course excerpts), all runs" --rejudge

echo "- finished: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$INFO"
uv run python -m study_assistant.evaluation --out "$OUT" --report
