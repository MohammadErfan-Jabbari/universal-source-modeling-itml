"""Student-side live evaluation runner for competition day."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from competition.evaluation.harness import evaluate_sequence
from competition.predictors.base import Predictor


DEFAULT_NUM_TOKENS = 200_000
SMOKE_TEST_NUM_TOKENS = 5_000


def _load_test_sequence(path: str | Path) -> np.ndarray:
    test_path = Path(path)
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    arr = np.load(test_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D test sequence array, got shape {arr.shape}.")
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("test.npy is empty.")
    return arr


def _infer_alphabet_size(sequence: np.ndarray) -> int:
    min_symbol = int(sequence.min())
    max_symbol = int(sequence.max())
    if min_symbol < 0:
        raise ValueError(f"Negative symbol detected in test sequence: {min_symbol}")
    return max_symbol + 1


def _load_predictor_builder(predictor_path: str | Path):
    path = Path(predictor_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictor file not found: {path}")

    spec = importlib.util.spec_from_file_location("student_predictor_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import predictor module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_predictor = getattr(module, "build_predictor", None)
    if build_predictor is None or not callable(build_predictor):
        raise AttributeError(
            "Predictor module must define callable build_predictor(alphabet_size, max_context_length)."
        )
    return build_predictor


def _build_predictor(
    predictor_path: str | Path,
    *,
    alphabet_size: int,
    max_context_length: int,
) -> Predictor:
    build_predictor = _load_predictor_builder(predictor_path)
    predictor = build_predictor(
        alphabet_size=int(alphabet_size),
        max_context_length=int(max_context_length),
    )
    if not isinstance(predictor, Predictor):
        raise TypeError(
            "build_predictor(...) must return an instance of competition.predictors.base.Predictor."
        )
    return predictor


def format_final_score_line(
    *,
    bits_per_symbol: float,
    elapsed_seconds: float,
    timed_out: bool,
    evaluated_tokens: int,
) -> str:
    return (
        "FINAL_SCORE "
        f"bits_per_symbol={bits_per_symbol:.10f} "
        f"elapsed_seconds={elapsed_seconds:.6f} "
        f"timed_out={timed_out} "
        f"evaluated_tokens={evaluated_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live evaluation on a released test set.")
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--predictor-path", type=str, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=600.0)
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--validate-probabilities", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.max_context_length < 0:
        raise ValueError("--max-context-length must be non-negative.")
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be positive.")
    if args.time_limit_seconds is not None and args.time_limit_seconds < 0:
        raise ValueError("--time-limit-seconds must be non-negative.")

    sequence = _load_test_sequence(args.test_path)
    requested_tokens = SMOKE_TEST_NUM_TOKENS if args.smoke_test else int(args.num_tokens)
    eval_seq = sequence[:requested_tokens]

    alphabet_size = _infer_alphabet_size(eval_seq)
    predictor = _build_predictor(
        args.predictor_path,
        alphabet_size=alphabet_size,
        max_context_length=args.max_context_length,
    )

    result = evaluate_sequence(
        predictor,
        eval_seq,
        max_context_length=args.max_context_length,
        max_seconds=args.time_limit_seconds,
        validate_probabilities=args.validate_probabilities,
    )

    print(
        format_final_score_line(
            bits_per_symbol=result.bits_per_symbol,
            elapsed_seconds=result.elapsed_seconds,
            timed_out=result.timed_out,
            evaluated_tokens=result.num_tokens,
        )
    )


if __name__ == "__main__":
    main()
