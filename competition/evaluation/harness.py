"""Sequential evaluation harness for empirical cross-entropy."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor


class ContextWindowView(Sequence[int]):
    """Read-only chronological view over a NumPy-backed ring buffer.

    The harness mutates this object in-place on each step to avoid per-token
    tuple/list allocations when passing context to predictors.
    """

    __slots__ = ("_buffer", "_capacity", "_start", "_length")

    def __init__(self, buffer: NDArray[np.int64]) -> None:
        if buffer.ndim != 1:
            raise ValueError("Context ring buffer must be one-dimensional.")
        self._buffer = buffer
        self._capacity = int(buffer.shape[0])
        self._start = 0
        self._length = 0

    def _set_state(self, *, start: int, length: int) -> None:
        if length < 0 or length > self._capacity:
            raise ValueError("Invalid context length for view.")
        if self._capacity == 0:
            self._start = 0
            self._length = 0
            return
        self._start = int(start % self._capacity)
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self[i] for i in range(start, stop, step)]

        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("context index out of range")
        if self._capacity == 0:
            raise IndexError("context is empty")
        return int(self._buffer[(self._start + idx) % self._capacity])

    def __iter__(self) -> Iterator[int]:
        for idx in range(self._length):
            yield int(self._buffer[(self._start + idx) % self._capacity])

    def __repr__(self) -> str:
        return f"ContextWindowView(length={self._length}, capacity={self._capacity})"


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of a sequential evaluation run."""

    num_tokens: int
    total_bits: float
    bits_per_symbol: float
    elapsed_seconds: float
    tokens_per_second: float
    timed_out: bool


def _validate_log_probs(log_probs: NDArray[np.float64], alphabet_size: int) -> None:
    if log_probs.shape != (alphabet_size,):
        raise ValueError(
            f"predict_next must return shape ({alphabet_size},), got {log_probs.shape}."
        )
    if np.any(np.isnan(log_probs)) or np.any(np.isposinf(log_probs)):
        raise ValueError("predict_next returned invalid log2 probabilities (NaN or +inf).")
    if np.any(log_probs > 0.0):
        raise ValueError("Log2 probabilities cannot be positive.")

    probs = np.exp2(log_probs)
    prob_sum = float(np.sum(probs))
    if not np.isclose(prob_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Predicted probabilities must sum to 1 (got {prob_sum:.8f})."
        )
    if np.any(probs < 0.0):
        raise ValueError("Predicted probabilities must be non-negative.")


def evaluate_sequence(
    predictor: Predictor,
    sequence: Iterable[int],
    *,
    max_context_length: int = 256,
    max_seconds: float | None = None,
    validate_probabilities: bool = True,
) -> EvaluationResult:
    """Evaluate a predictor on a symbol sequence in strictly sequential mode.

    The harness truncates context before calling `predict_next`, ensuring
    predictors cannot access more than `max_context_length` past symbols.
    """

    if max_context_length < 0:
        raise ValueError("max_context_length must be non-negative.")
    if max_seconds is not None and max_seconds < 0:
        raise ValueError("max_seconds must be non-negative or None.")

    predictor.initialize()

    context_buffer = np.empty(max_context_length, dtype=np.int64)
    context_view = ContextWindowView(context_buffer)
    write_pos = 0
    history_len = 0

    total_bits = 0.0
    num_tokens = 0
    start = perf_counter()
    timed_out = False

    for raw_symbol in sequence:
        if max_seconds is not None and (perf_counter() - start) >= max_seconds:
            timed_out = True
            break

        symbol = int(raw_symbol)
        if symbol < 0 or symbol >= predictor.alphabet_size:
            raise ValueError(
                f"Observed symbol {symbol} out of range [0, {predictor.alphabet_size})."
            )

        if max_context_length == 0:
            context_view._set_state(start=0, length=0)
        else:
            start_idx = (write_pos - history_len) % max_context_length
            context_view._set_state(start=start_idx, length=history_len)

        log_probs = np.asarray(predictor.predict_next(context_view), dtype=np.float64)

        if validate_probabilities:
            _validate_log_probs(log_probs, predictor.alphabet_size)

        true_log_prob = float(log_probs[symbol])
        total_bits += -true_log_prob

        predictor.update(symbol)
        if max_context_length > 0:
            context_buffer[write_pos] = symbol
            write_pos = (write_pos + 1) % max_context_length
            if history_len < max_context_length:
                history_len += 1
        num_tokens += 1

    elapsed_seconds = perf_counter() - start
    bits_per_symbol = total_bits / num_tokens if num_tokens else float("nan")
    if elapsed_seconds > 0:
        tokens_per_second = num_tokens / elapsed_seconds
    else:
        tokens_per_second = float("inf") if num_tokens > 0 else 0.0

    return EvaluationResult(
        num_tokens=num_tokens,
        total_bits=total_bits,
        bits_per_symbol=bits_per_symbol,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
        timed_out=timed_out,
    )
