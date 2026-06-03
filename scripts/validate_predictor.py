#!/usr/bin/env python3
"""Fast stability and train-val validation for USM Challenge autoresearch loop.

Usage:
    uv run python scripts/validate_predictor.py \
        --predictor-path submissions/your_predictor.py \
        [--train-path data/generator/train.npy] \
        [--stability-blocks 3] \
        [--stability-block-size 5000] \
        [--train-val-size 50000] \
        [--max-context-length 256]

Output (JSON, one line):
    {"stability":{"scores":[...],"std":0.0012,"elapsed":1.23},
     "train_val":{"score":2.9734,"elapsed":2.45}}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

# Ensure competition package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from competition.evaluation.harness import (
    ContextWindowView,
    EvaluationResult,
)
from competition.predictors.base import Predictor


# ---------------------------------------------------------------------------
# Helper: build predictor from file
# ---------------------------------------------------------------------------

def _build_predictor(predictor_path: str, alphabet_size: int, max_context_length: int) -> Predictor:
    path = Path(predictor_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictor not found: {path}")
    spec = importlib.util.spec_from_file_location("student_predictor_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predictor = mod.build_predictor(alphabet_size, max_context_length)
    if not isinstance(predictor, Predictor):
        raise TypeError("build_predictor must return a Predictor instance")
    return predictor


# ---------------------------------------------------------------------------
# Core: warm up predictor, then evaluate a contiguous slice
# ---------------------------------------------------------------------------

def _warmup_and_evaluate(
    predictor: Predictor,
    warmup: NDArray[np.int64],
    eval_seq: NDArray[np.int64],
    max_context_length: int,
) -> EvaluationResult:
    """Initialize predictor, process warmup prefix (no metrics), then evaluate eval_seq."""
    predictor.initialize()

    # Setup context ring buffer (mirrors harness exactly)
    context_buffer = np.empty(max_context_length, dtype=np.int64) if max_context_length > 0 else None
    context_view = ContextWindowView(context_buffer) if max_context_length > 0 else None
    write_pos = 0
    history_len = 0

    def _advance(symbol: int) -> float:
        nonlocal write_pos, history_len
        if max_context_length == 0:
            log_probs = np.asarray(predictor.predict_next(()), dtype=np.float64)
        else:
            start_idx = (write_pos - history_len) % max_context_length
            context_view._set_state(start=start_idx, length=history_len)
            log_probs = np.asarray(predictor.predict_next(context_view), dtype=np.float64)

        # Skip validation for speed (already checked in smoke test)
        true_log_prob = float(log_probs[symbol])

        predictor.update(symbol)
        if max_context_length > 0:
            context_buffer[write_pos] = symbol
            write_pos = (write_pos + 1) % max_context_length
            if history_len < max_context_length:
                history_len += 1
        return true_log_prob

    # Warmup: feed prefix, no timing, no metrics
    for raw in warmup:
        sym = int(raw)
        if sym < 0 or sym >= predictor.alphabet_size:
            raise ValueError(f"Symbol {sym} out of range")
        _advance(sym)

    # Evaluate: time and track metrics
    total_bits = 0.0
    num_tokens = 0
    t0 = perf_counter()

    for raw in eval_seq:
        sym = int(raw)
        if sym < 0 or sym >= predictor.alphabet_size:
            raise ValueError(f"Symbol {sym} out of range")
        total_bits += -_advance(sym)
        num_tokens += 1

    elapsed = perf_counter() - t0
    bps = total_bits / num_tokens if num_tokens else float("nan")
    tps = num_tokens / elapsed if elapsed > 0 else float("inf")

    return EvaluationResult(
        num_tokens=num_tokens,
        total_bits=total_bits,
        bits_per_symbol=bps,
        elapsed_seconds=elapsed,
        tokens_per_second=tps,
        timed_out=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="USM predictor validation")
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--train-path", default="data/generator/train.npy")
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--stability-blocks", type=int, default=3)
    parser.add_argument("--stability-block-size", type=int, default=5000)
    parser.add_argument("--train-val-size", type=int, default=50000)
    args = parser.parse_args()

    train = np.load(args.train_path)
    if train.ndim != 1:
        raise ValueError(f"train must be 1D, got {train.shape}")
    alphabet_size = int(train.max()) + 1

    # -------------------------------------------------------------------
    # Stability check: N contiguous blocks with proper warmup
    # -------------------------------------------------------------------
    stability_scores: list[dict] = []
    stability_start = perf_counter()

    block_size = args.stability_block_size
    # Fixed warmup + fixed eval block for all runs
    # This detects non-determinism: same data, same predictor, should give identical scores
    warmup_len = block_size * 10  # 50k warmup
    eval_start = warmup_len       # eval at 50k-55k
    eval_end = eval_start + block_size

    if eval_end > len(train):
        eval_end = len(train)

    for i in range(args.stability_blocks):
        # All blocks use the SAME warmup and SAME eval slice

        predictor = _build_predictor(args.predictor_path, alphabet_size, args.max_context_length)
        result = _warmup_and_evaluate(
            predictor,
            train[:eval_start],
            train[eval_start:eval_end],
            args.max_context_length,
        )
        stability_scores.append({
            "block": i,
            "bits_per_symbol": float(result.bits_per_symbol),
            "elapsed_seconds": float(result.elapsed_seconds),
            "tokens": int(result.num_tokens),
        })

    stability_elapsed = perf_counter() - stability_start
    stability_std = float(np.std([s["bits_per_symbol"] for s in stability_scores])) if stability_scores else None

    # -------------------------------------------------------------------
    # Train-derived validation: last N symbols with full prefix warmup
    # -------------------------------------------------------------------
    train_val_score = None
    train_val_elapsed = None

    if len(train) > args.train_val_size:
        tv_start = perf_counter()
        predictor = _build_predictor(args.predictor_path, alphabet_size, args.max_context_length)
        result = _warmup_and_evaluate(
            predictor,
            train[:-args.train_val_size],
            train[-args.train_val_size:],
            args.max_context_length,
        )
        train_val_elapsed = perf_counter() - tv_start
        train_val_score = float(result.bits_per_symbol)

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    output = {
        "stability": {
            "scores": stability_scores,
            "std": stability_std,
            "elapsed_seconds": round(stability_elapsed, 3),
        },
        "train_val": {
            "score": train_val_score,
            "elapsed_seconds": round(train_val_elapsed, 3) if train_val_elapsed else None,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
