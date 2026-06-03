"""
validation_script.py — Train-derived validation + stability checker for USM Challenge predictors.

This script reduces the risk of hallucinated/overfitted public-practice scores by:
1. Creating a held-out validation split from train.npy
2. Evaluating the predictor on both train-val and public practice
3. Running the predictor multiple times to check score stability
4. Reporting a combined reliability score

Usage:
    uv run python validation_script.py --predictor submissions/your_predictor.py [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# Add competition package to path
sys.path.insert(0, str(Path(__file__).parent / "student_bundle_v2026-course.3"))

from competition.evaluation.harness import evaluate_predictor, EvaluationResult
from competition.predictors.base import Predictor


def load_predictor(path: str, alphabet_size: int, max_context_length: int):
    """Dynamically import and build a predictor from a Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("predictor_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_predictor(alphabet_size, max_context_length)


def split_train_val(train_path: str, val_size: int = 50000):
    """Split train.npy into fit and validation suffix."""
    train = np.load(train_path)
    if len(train) <= val_size:
        raise ValueError(f"Train too short ({len(train)}) for val_size {val_size}")
    fit = train[:-val_size]
    val = train[-val_size:]
    return fit, val


def evaluate_on_sequence(predictor: Predictor, sequence: np.ndarray) -> EvaluationResult:
    """Evaluate predictor on a given sequence (no file loading, just the harness logic)."""
    # We use the harness's evaluate_predictor but with a custom iterable
    # Since evaluate_predictor takes a file path, we replicate its core logic here.
    from competition.evaluation.harness import evaluate_predictor
    # Save sequence to temp file
    tmp_path = Path("/tmp/val_sequence.npy")
    np.save(tmp_path, sequence)
    return evaluate_predictor(predictor, str(tmp_path))


def main():
    parser = argparse.ArgumentParser(description="Validate USM predictor reliability")
    parser.add_argument("--predictor", required=True, help="Path to predictor .py file")
    parser.add_argument("--train", default="data/generator/train.npy", help="Train data path")
    parser.add_argument("--test", default="data/public_practice/test.npy", help="Public practice test path")
    parser.add_argument("--val-size", type=int, default=50000, help="Validation suffix size from train")
    parser.add_argument("--runs", type=int, default=2, help="Number of stability runs on test")
    parser.add_argument("--alphabet-size", type=int, default=16)
    parser.add_argument("--max-context-length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("USM Predictor Validation & Stability Check")
    print("=" * 60)
    print(f"Predictor: {args.predictor}")
    print()

    # 1. Split train
    print(f"[1] Splitting train into fit / val (last {args.val_size} symbols)...")
    try:
        fit, val = split_train_val(args.train, args.val_size)
        print(f"    Fit size: {len(fit)}, Val size: {len(val)}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # 2. Evaluate on train-val
    print("[2] Evaluating on train-derived validation...")
    try:
        predictor_val = load_predictor(args.predictor, args.alphabet_size, args.max_context_length)
        t0 = perf_counter()
        result_val = evaluate_on_sequence(predictor_val, val)
        elapsed_val = perf_counter() - t0
        print(f"    Val score:    {result_val.bits_per_symbol:.10f} bits/symbol")
        print(f"    Val elapsed:  {elapsed_val:.3f}s")
    except Exception as e:
        print(f"    ERROR: {e}")
        result_val = None

    # 3. Evaluate on public practice (multiple runs for stability)
    print(f"[3] Evaluating on public practice ({args.runs} stability runs)...")
    test_scores = []
    for r in range(1, args.runs + 1):
        try:
            predictor_test = load_predictor(args.predictor, args.alphabet_size, args.max_context_length)
            t0 = perf_counter()
            result_test = evaluate_on_sequence(predictor_test, np.load(args.test))
            elapsed_test = perf_counter() - t0
            test_scores.append(result_test.bits_per_symbol)
            print(f"    Run {r}: {result_test.bits_per_symbol:.10f} bps  ({elapsed_test:.3f}s)")
        except Exception as e:
            print(f"    Run {r}: ERROR — {e}")

    # 4. Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if result_val:
        print(f"Train-val score:      {result_val.bits_per_symbol:.10f}")
    if test_scores:
        mean_test = np.mean(test_scores)
        std_test = np.std(test_scores)
        min_test = np.min(test_scores)
        print(f"Public practice mean: {mean_test:.10f}")
        print(f"Public practice std:  {std_test:.10f}")
        print(f"Public practice min:  {min_test:.10f}")
        if std_test > 1e-6:
            print(f"WARNING: Score varies by {std_test:.2e} across runs — investigate non-determinism.")
        else:
            print(f"OK: Score is stable across runs.")

    # 5. Reliability check
    if result_val and test_scores:
        gap = abs(result_val.bits_per_symbol - np.mean(test_scores))
        print(f"Train-val / test gap: {gap:.10f}")
        if gap > 0.05:
            print(f"WARNING: Large gap suggests overfitting to public practice.")
        else:
            print(f"OK: Gap is reasonable.")

    print("=" * 60)


if __name__ == "__main__":
    main()
