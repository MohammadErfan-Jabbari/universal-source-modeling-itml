#!/usr/bin/env bash
set -euo pipefail

PREDICTOR_PATH="${PREDICTOR_PATH:-submissions/template_predictor.py}"
TEST_PATH="${TEST_PATH:-data/public_practice/test.npy}"
NUM_TOKENS="${NUM_TOKENS:-200000}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-600}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-256}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"

if [ ! -f "$PREDICTOR_PATH" ]; then
  echo "[autoresearch] missing predictor: $PREDICTOR_PATH" >&2
  exit 10
fi

if [ ! -f "$TEST_PATH" ]; then
  echo "[autoresearch] missing test data: $TEST_PATH" >&2
  exit 11
fi

RUN_NUM="${RUN_NUM:-$(uv run python - <<'PY'
import json
from pathlib import Path
path = Path('autoresearch.jsonl')
max_run = 0
if path.exists():
    for line in path.read_text(errors='replace').splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        run = obj.get('run')
        if isinstance(run, int):
            max_run = max(max_run, run)
print(max_run + 1)
PY
)}"

RUN_PADDED="$(printf "%03d" "$RUN_NUM")"
EXPLANATION_PATH="explanations/run_${RUN_PADDED}.md"

if [ ! -f "$EXPLANATION_PATH" ]; then
  cat >&2 <<MSG
[autoresearch] Missing required pre-run explanation: $EXPLANATION_PATH
Write it before running. See explanations/README.md for the required template.
If you are intentionally re-running an old experiment, set RUN_NUM explicitly.
MSG
  exit 12
fi

for heading in \
  "## Proposed change" \
  "## Source and evidence" \
  "## Course-material connection" \
  "## Hypothesis" \
  "## Risks" \
  "## Validation plan"; do
  if ! grep -qF "$heading" "$EXPLANATION_PATH"; then
    echo "[autoresearch] $EXPLANATION_PATH is missing required heading: $heading" >&2
    exit 13
  fi
done

echo "[autoresearch] run=${RUN_PADDED}"
echo "[autoresearch] predictor=${PREDICTOR_PATH}"
echo "[autoresearch] test=${TEST_PATH}"
echo "[autoresearch] num_tokens=${NUM_TOKENS}"
echo "[autoresearch] explanation=${EXPLANATION_PATH}"

# ---------------------------------------------------------------------------
# STEP 1: Official full evaluation (200k tokens)
# ---------------------------------------------------------------------------
OUTPUT="$(
  uv run python -m competition.run_live_eval \
    --test-path "$TEST_PATH" \
    --predictor-path "$PREDICTOR_PATH" \
    --time-limit-seconds "$TIME_LIMIT_SECONDS" \
    --max-context-length "$MAX_CONTEXT_LENGTH" \
    --num-tokens "$NUM_TOKENS"
)"

echo "$OUTPUT"

FINAL_LINE="$(echo "$OUTPUT" | grep '^FINAL_SCORE ' || true)"
if [ -z "$FINAL_LINE" ]; then
  echo "[autoresearch] evaluator did not emit FINAL_SCORE" >&2
  exit 14
fi

BITS="$(echo "$FINAL_LINE" | sed -E 's/.*bits_per_symbol=([^ ]+).*/\1/')"
ELAPSED="$(echo "$FINAL_LINE" | sed -E 's/.*elapsed_seconds=([^ ]+).*/\1/')"
TIMED_OUT_RAW="$(echo "$FINAL_LINE" | sed -E 's/.*timed_out=([^ ]+).*/\1/')"
TOKENS="$(echo "$FINAL_LINE" | sed -E 's/.*evaluated_tokens=([^ ]+).*/\1/')"

if [ "$TIMED_OUT_RAW" = "True" ]; then
  TIMED_OUT=1
else
  TIMED_OUT=0
fi

echo "METRIC bits_per_symbol=$BITS"
echo "METRIC elapsed_seconds=$ELAPSED"
echo "METRIC timed_out=$TIMED_OUT"
echo "METRIC evaluated_tokens=$TOKENS"

if [ "$TIMED_OUT" -ne 0 ]; then
  echo "[autoresearch] invalid: timed_out=True" >&2
  exit 15
fi

if [ "$TOKENS" -ne "$NUM_TOKENS" ]; then
  echo "[autoresearch] invalid: evaluated_tokens=$TOKENS expected=$NUM_TOKENS" >&2
  exit 16
fi

# ---------------------------------------------------------------------------
# STEP 2: Stability check + train-derived validation (fast, ~3-4s total)
# ---------------------------------------------------------------------------
if [ "$SKIP_VALIDATION" -eq 0 ] && [ -f "scripts/validate_predictor.py" ]; then
  echo "[autoresearch] Running stability + train-val validation..."
  VAL_OUTPUT="$(uv run python scripts/validate_predictor.py \
    --predictor-path "$PREDICTOR_PATH" \
    --max-context-length "$MAX_CONTEXT_LENGTH" 2>/dev/null || echo '{}')"

  # Extract and print key metrics for agent parsing
  STABILITY_STD="$(echo "$VAL_OUTPUT" | uv run python -c "import sys,json; d=json.load(sys.stdin); print(d.get('stability',{}).get('std','N/A'))" 2>/dev/null || echo 'N/A')"
  TRAIN_VAL="$(echo "$VAL_OUTPUT" | uv run python -c "import sys,json; d=json.load(sys.stdin); print(d.get('train_val',{}).get('score','N/A'))" 2>/dev/null || echo 'N/A')"
  STABILITY_ELAPSED="$(echo "$VAL_OUTPUT" | uv run python -c "import sys,json; d=json.load(sys.stdin); print(d.get('stability',{}).get('elapsed_seconds','N/A'))" 2>/dev/null || echo 'N/A')"
  TV_ELAPSED="$(echo "$VAL_OUTPUT" | uv run python -c "import sys,json; d=json.load(sys.stdin); print(d.get('train_val',{}).get('elapsed_seconds','N/A'))" 2>/dev/null || echo 'N/A')"

  echo "VALIDATION stability_std=$STABILITY_STD train_val_score=$TRAIN_VAL stability_elapsed=${STABILITY_ELAPSED}s train_val_elapsed=${TV_ELAPSED}s"

  # Print individual block scores for detailed inspection
  echo "$VAL_OUTPUT" | uv run python -c "
import sys, json
d = json.load(sys.stdin)
for s in d.get('stability', {}).get('scores', []):
    print(f\"VALIDATION_BLOCK block={s['block']} bits_per_symbol={s['bits_per_symbol']:.10f} elapsed={s['elapsed_seconds']:.3f}s\")
" 2>/dev/null || true
else
  echo "VALIDATION skipped"
fi
