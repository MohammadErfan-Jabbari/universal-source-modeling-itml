#!/usr/bin/env bash
set -euo pipefail

PREDICTOR_PATH="${PREDICTOR_PATH:-submissions/template_predictor.py}"

# Syntax/import check for tracked Python files.
uv run python -m py_compile \
  competition/run_live_eval.py \
  competition/evaluation/harness.py \
  competition/predictors/base.py \
  competition/predictors/ngram.py \
  competition/predictors/uniform.py \
  "$PREDICTOR_PATH"

# Fast smoke test with probability validation.
uv run python -m competition.run_live_eval \
  --test-path data/public_practice/test.npy \
  --predictor-path "$PREDICTOR_PATH" \
  --smoke-test \
  --validate-probabilities >/tmp/itml_usm_smoke_check.out

grep '^FINAL_SCORE ' /tmp/itml_usm_smoke_check.out

# Public practice hash integrity. The bundled competition.verify_hashes.py is broken
# because it imports a missing instructor-side module, so use sha256sum directly.
(cd data/public_practice && sha256sum -c sha256.txt)
