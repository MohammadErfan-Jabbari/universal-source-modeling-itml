#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import json, sys
try:
    payload = json.loads(sys.stdin.read() or '{}')
except Exception:
    payload = {}
entry = payload.get('run_entry') or {}
session = payload.get('session') or {}
print('[autoresearch-after] Result logged.')
if entry:
    print(f"[autoresearch-after] run={entry.get('run')} status={entry.get('status')} metric={entry.get('metric')}")
if session:
    print(f"[autoresearch-after] best={session.get('best_metric')} baseline={session.get('baseline_metric')} count={session.get('run_count')}")
print('[autoresearch-after] Next step: update autoresearch.md / EXPERIMENTS.md if this run taught a durable lesson, then write the next explanations/run_###.md before benchmarking again.')
PY
