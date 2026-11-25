#!/usr/bin/env bash
# Re-runs every evaluation configuration in docs/report.md from scratch.
# Needs: `uv sync`, Ollama running with `ollama pull llama3.1:8b`. Takes about 2¾ hours on an M2 laptop.
#
# Results go to a NEW folder, eval/runs/<date>-<commit>/ (override with OUT=...). The run reported in
# docs/report.md stays untouched in eval/results/. The script refuses to write into a folder that already
# holds results (saved answers would be reused, so nothing would be re-run), unless RESUME=1 is set to
# continue an interrupted run of your own.
# At the end it prints the analysis for the new folder and appends its scores to eval/experiment_log.csv
# as build "$BUILD" (default: "reproduction <date> <commit>").
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
OUT=${OUT:-eval/runs/$(date +%Y%m%d-%H%M)-$COMMIT}
if [ -n "$(ls -A "$OUT" 2>/dev/null | grep '\.jsonl$' || true)" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "$OUT already holds results; they would be reused rather than re-run."
  echo "Use a new OUT=... folder, or RESUME=1 to continue an interrupted run in that folder."
  exit 1
fi
mkdir -p "$OUT"
INFO="$OUT/RUN_INFO.md"
[ "${RESUME:-0}" = "1" ] && [ -f "$INFO" ] && INFO="$OUT/RUN_INFO_resumed_$(date +%Y%m%d-%H%M).md"
echo "Writing results to $OUT"
{
  echo "# Evaluation run"
  echo "- started: $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "- code: git $COMMIT"
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
run "Ablation: support check on (strict)"      --backend ollama --systems rag --retriever hybrid --verify strict
run "Ablation: support check on (concept)"     --backend ollama --systems rag --retriever hybrid --verify concept
run "Ablation: similarity floor off"           --backend ollama --systems rag --retriever hybrid --no-floor
run "Ablation: less context (3 passages)"      --backend ollama --systems rag --retriever hybrid --context small
run "Ablation: more context (8 passages)"      --backend ollama --systems rag --retriever hybrid --context large
run "Llama, no course, sees the syllabus"      --backend ollama --systems no_rag --syllabus strict
run "…syllabus, lenient wording"              --backend ollama --systems no_rag --syllabus lenient
run "flan-t5, with and without course"         --backend flan --systems rag,no_rag --retriever hybrid
run "Judge v2 (with course excerpts), all runs" --rejudge

echo "- finished: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$INFO"
uv run python -m study_assistant.evaluation --out "$OUT" --report
uv run python eval/analysis.py --results "$OUT"   # hand labels apply only to answers identical to the ones we judged
uv run python -m study_assistant.experiment_log --results "$OUT" --build "${BUILD:-reproduction $(date +%F) $COMMIT}" \
  --hand eval/hand_labels.json --notes "reproduction run (scripts/run_evaluation.sh)"
