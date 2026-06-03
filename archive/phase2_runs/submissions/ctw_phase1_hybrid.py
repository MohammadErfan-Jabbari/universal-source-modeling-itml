"""Approximate Context Tree Weighting component mixed with Run 004/Phase 1 hybrid.

The CTW-like component recursively blends a local KT estimate at each suffix
context with the shorter-context weighted distribution using canonical 1/2
mixing.  This differs from the min-count VOMM hard-depth selector and from PPM
escapes: all supported suffix depths contribute through a tree-style recursive
mixture.  The component is confidence-gated into the current best PPM+Phase1
hybrid so the new architecture can help only when it is sharp and supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from submissions.ppm_phase1_hybrid import build_predictor as build_base_predictor


DATA_DIR = Path("data/generator")
MAX_DEPTH = 5
KT_ALPHA = 0.5
CTW_MAX_WEIGHT = 0.05
COUNT_C = 100.0


class _Stats:
    __slots__ = ("total", "counts")

    def __init__(self) -> None:
        self.total = 0
        self.counts: dict[int, int] = {}


class ApproxCTWComponent(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int, train: NDArray[np.int64]) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        self.max_depth = int(min(MAX_DEPTH, max_context_length))
        self._tables: list[dict[tuple[int, ...], _Stats]] = [{} for _ in range(self.max_depth + 1)]
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._local = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._pending_context: tuple[int, ...] | None = None
        self.deepest_support = 0
        self.entropy_confidence = 0.0
        self._fit(train)

    def initialize(self) -> None:
        self._pending_context = None
        self.deepest_support = 0
        self.entropy_confidence = 0.0

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        context_tail = self._tail_context(context)
        max_depth = min(len(context_tail), self.max_depth)

        # Start with order-0 KT estimate, then recursively mix in each longer
        # suffix's local KT estimate with canonical 1/2 CTW-style weight.
        self._fill_kt(self._tables[0].get(()), self._probs)
        self.deepest_support = 0
        for depth in range(1, max_depth + 1):
            stats = self._tables[depth].get(context_tail[-depth:])
            if stats is None or stats.total <= 0:
                continue
            self.deepest_support = stats.total
            self._fill_kt(stats, self._local)
            self._probs *= 0.5
            self._probs += 0.5 * self._local

        self._probs /= float(np.sum(self._probs))
        entropy = -float(np.sum(self._probs * np.log2(np.maximum(self._probs, 1e-300))))
        self.entropy_confidence = max(0.0, 1.0 - entropy / np.log2(self.alphabet_size))
        np.log2(self._probs, out=self._log_probs)
        self._pending_context = context_tail
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        if self._pending_context is None:
            raise RuntimeError("update() called before predict_next().")
        self._add_observation(self._pending_context, int(observed_symbol))
        self._pending_context = None

    def _fill_kt(self, stats: _Stats | None, out: NDArray[np.float64]) -> None:
        total = 0 if stats is None else stats.total
        denom = total + self.alphabet_size * KT_ALPHA
        out.fill(KT_ALPHA / denom)
        if stats is not None:
            inv = 1.0 / denom
            for sym, count in stats.counts.items():
                out[sym] = (count + KT_ALPHA) * inv

    def _tail_context(self, context: Sequence[int]) -> tuple[int, ...]:
        n = min(len(context), self.max_depth)
        return tuple(int(x) for x in context[-n:]) if n > 0 else ()

    def _fit(self, train: NDArray[np.int64]) -> None:
        history: list[int] = []
        for raw in train:
            symbol = int(raw)
            self._add_observation(tuple(history[-self.max_depth:]), symbol)
            history.append(symbol)

    def _add_observation(self, context_tail: tuple[int, ...], symbol: int) -> None:
        max_depth = min(len(context_tail), self.max_depth)
        for depth in range(max_depth + 1):
            key = context_tail[-depth:] if depth > 0 else ()
            stats = self._tables[depth].get(key)
            if stats is None:
                stats = _Stats()
                self._tables[depth][key] = stats
            stats.total += 1
            stats.counts[symbol] = stats.counts.get(symbol, 0) + 1


class CTWHybridPredictor(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        train = np.asarray(np.load(DATA_DIR / "train.npy"), dtype=np.int64)
        self.base = build_base_predictor(alphabet_size, max_context_length)
        self.ctw = ApproxCTWComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.ctw.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        ctw_probs = np.exp2(np.asarray(self.ctw.predict_next(context), dtype=np.float64))
        count_conf = self.ctw.deepest_support / (self.ctw.deepest_support + COUNT_C)
        w = CTW_MAX_WEIGHT * count_conf * self.ctw.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * ctw_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.ctw.update(observed_symbol)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    return CTWHybridPredictor(alphabet_size, max_context_length)
