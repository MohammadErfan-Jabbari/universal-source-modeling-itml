#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_CWD:-$(pwd)}"
cd "$ROOT"

cat <<'MSG'
[autoresearch-before] Required loop discipline:
- Review AGENTS.md and autoresearch.md before choosing a change.
- Read relevant course notes in lectures/ and cite them when they inform the idea.
- Write explanations/run_###.md BEFORE running ./autoresearch.sh.
- Use search tools when useful, but cite external sources in the explanation.
- Preserve official evaluator semantics and avoid lookahead/test leakage.
MSG

if [ -d explanations ]; then
  echo "[autoresearch-before] Last up to 5 explanations:"
  find explanations -maxdepth 1 -type f -name 'run_*.md' -printf '%f\n' \
    | sort -V \
    | tail -5 \
    | while read -r f; do
        printf "\n===== explanations/%s =====\n" "$f"
        # Keep hook output compact; stdout is fed back as steer context.
        sed -n '1,90p' "explanations/$f"
      done
else
  echo "[autoresearch-before] No explanations/ directory found yet."
fi
