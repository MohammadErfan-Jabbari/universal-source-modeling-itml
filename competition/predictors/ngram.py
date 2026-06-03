"""Laplace-smoothed n-gram baseline predictor with suffix backoff.

Backoff behavior:
- Let `n` be the model order (e.g., `n=3` for trigrams).
- At prediction time the model uses at most the last `n-1` symbols from the
  harness-provided context.
- If that full context has not been seen during training/online updates, it
  backs off by dropping the oldest symbol (suffix backoff) until a seen context
  is found.
- If none is found, it uses the unigram/zero-order context `()`.

This implementation stores sparse next-symbol counts for each seen context to
avoid allocating a dense `alphabet_size` vector per context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor


@dataclass(slots=True)
class _SparseNextSymbolCounts:
    """Sparse counts for next-symbol statistics conditioned on one context."""

    total: int = 0
    counts: dict[int, int] = field(default_factory=dict)


class NGramPredictor(Predictor):
    """Laplace-smoothed n-gram predictor with optional online adaptation.

    The predictor can be fit on a training sequence and then evaluated
    sequentially. By default, `update()` also incorporates observed test tokens
    online (still without peeking) using only the harness-provided context from
    the immediately preceding `predict_next()` call.
    """

    def __init__(
        self,
        alphabet_size: int,
        *,
        n: int = 3,
        laplace: float = 1.0,
        max_context_length: int | None = None,
        adapt_online: bool = True,
        debug: bool = False,
        debug_max_messages: int = 20,
    ) -> None:
        if n < 1:
            raise ValueError("n must be at least 1.")
        if laplace <= 0:
            raise ValueError("laplace must be positive.")

        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        self.n = int(n)
        self.laplace = float(laplace)
        self.adapt_online = bool(adapt_online)
        self.debug = bool(debug)
        self.debug_max_messages = int(debug_max_messages)
        if self.debug_max_messages < 0:
            raise ValueError("debug_max_messages must be non-negative.")

        # One table per order k = 0..n-1. Keys are tuples of length k.
        self._counts_by_order: list[dict[tuple[int, ...], _SparseNextSymbolCounts]] = [
            {} for _ in range(self.n)
        ]
        self._pending_context_for_update: tuple[int, ...] | None = None
        self._log_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._is_fitted = False
        self._debug_messages_emitted = 0
        self._logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Reset per-run transient state but keep learned counts."""

        self._pending_context_for_update = None
        self._debug_messages_emitted = 0

    def fit(self, sequence: Iterable[int]) -> "NGramPredictor":
        """Fit n-gram counts from a training sequence.

        Args:
            sequence: Training symbols. Must be integers in [0, alphabet_size).

        Returns:
            self
        """

        self._reset_counts()
        history: deque[int] = deque(maxlen=max(self.n - 1, 0))

        for raw_symbol in sequence:
            symbol = int(raw_symbol)
            self._validate_symbol(symbol)
            context_tail = tuple(history)
            self._add_observation(context_tail=context_tail, symbol=symbol)
            if self.n > 1:
                history.append(symbol)

        self._is_fitted = True
        self.initialize()
        return self

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        """Return Laplace-smoothed log2 probabilities using suffix backoff."""

        context_tail = self._extract_tail_context(context)
        chosen_order, stats = self._lookup_backoff_stats(context_tail)

        denom = (self.alphabet_size * self.laplace) + (stats.total if stats is not None else 0)
        base_prob = self.laplace / denom
        base_log_prob = float(np.log2(base_prob))
        self._log_probs_buffer.fill(base_log_prob)

        if stats is not None and stats.counts:
            for symbol, count in stats.counts.items():
                prob = (count + self.laplace) / denom
                self._log_probs_buffer[symbol] = np.log2(prob)

        if self.adapt_online:
            self._pending_context_for_update = context_tail
        else:
            self._pending_context_for_update = None

        if self.debug and self._logger.isEnabledFor(logging.DEBUG):
            requested_order = min(len(context_tail), self.n - 1)
            backed_off = chosen_order < requested_order
            if backed_off and self._debug_messages_emitted < self.debug_max_messages:
                requested_ctx = context_tail[-requested_order:] if requested_order > 0 else ()
                chosen_ctx = context_tail[-chosen_order:] if chosen_order > 0 else ()
                self._logger.debug(
                    "NGramPredictor backoff: requested_order=%d chosen_order=%d "
                    "requested_context=%s chosen_context=%s total_count=%d",
                    requested_order,
                    chosen_order,
                    requested_ctx,
                    chosen_ctx,
                    0 if stats is None else stats.total,
                )
                self._debug_messages_emitted += 1

        return self._log_probs_buffer

    def update(self, observed_symbol: int) -> None:
        """Optionally adapt online with the observed symbol after prediction."""

        symbol = int(observed_symbol)
        self._validate_symbol(symbol)
        if not self.adapt_online:
            return

        if self._pending_context_for_update is None:
            raise RuntimeError(
                "update() called before predict_next() or predictor state was not initialized."
            )

        self._add_observation(context_tail=self._pending_context_for_update, symbol=symbol)
        self._pending_context_for_update = None
        self._is_fitted = True

    def _reset_counts(self) -> None:
        self._counts_by_order = [{} for _ in range(self.n)]
        self._pending_context_for_update = None
        self._is_fitted = False

    def _validate_symbol(self, symbol: int) -> None:
        if symbol < 0 or symbol >= self.alphabet_size:
            raise ValueError(
                f"Observed symbol {symbol} out of range [0, {self.alphabet_size})."
            )

    def _extract_tail_context(self, context: Sequence[int]) -> tuple[int, ...]:
        max_order = self.n - 1
        if max_order <= 0:
            return ()

        context_len = len(context)
        usable = min(context_len, max_order)
        if usable <= 0:
            return ()

        start = context_len - usable
        return tuple(int(context[start + idx]) for idx in range(usable))

    def _lookup_backoff_stats(
        self, context_tail: tuple[int, ...]
    ) -> tuple[int, _SparseNextSymbolCounts | None]:
        max_order = min(len(context_tail), self.n - 1)
        for order in range(max_order, -1, -1):
            key = context_tail[-order:] if order > 0 else ()
            stats = self._counts_by_order[order].get(key)
            if stats is not None:
                return order, stats
        return 0, None

    def _add_observation(self, *, context_tail: tuple[int, ...], symbol: int) -> None:
        max_order = min(len(context_tail), self.n - 1)
        for order in range(max_order + 1):
            key = context_tail[-order:] if order > 0 else ()
            table = self._counts_by_order[order]
            stats = table.get(key)
            if stats is None:
                stats = _SparseNextSymbolCounts()
                table[key] = stats
            stats.total += 1
            stats.counts[symbol] = stats.counts.get(symbol, 0) + 1
