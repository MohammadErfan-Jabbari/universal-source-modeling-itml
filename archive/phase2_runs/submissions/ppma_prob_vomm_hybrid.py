"""PPM-A escape component mixed with current probabilistic-depth VOMM best.

This candidate adds a conservative, explicitly escaped PPM-A style model to the
current Run 014 best.  Unlike PPM-C (escape mass tied to number of distinct
continuations) and PPM-D-style variants, PPM-A uses one escape pseudo-count per
context: observed continuations get count/(N+1) and unseen mass gets 1/(N+1),
recursing to shorter contexts with exclusions.  The component is gated very
lightly to test whether a different escape law adds complementary signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from submissions.prob_depth_vomm_countmix_hybrid import build_predictor as build_base_predictor


DATA_DIR = Path("data/generator")
MAX_ORDER = 5
PPMA_MAX_WEIGHT = 0.02
GATE_COUNT_C = 120.0


class _Stats:
    __slots__ = ("total", "counts")

    def __init__(self) -> None:
        self.total = 0
        self.counts: dict[int, int] = {}


class PPMAComponent(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int, train: NDArray[np.int64]) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        self.max_order = int(min(MAX_ORDER, max_context_length))
        self._tables: list[dict[tuple[int, ...], _Stats]] = [{} for _ in range(self.max_order + 1)]
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._pending_context: tuple[int, ...] | None = None
        self.last_support = 0
        self.entropy_confidence = 0.0
        self._fit(train)

    def initialize(self) -> None:
        self._pending_context = None
        self.last_support = 0
        self.entropy_confidence = 0.0

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        context_tail = self._tail_context(context)
        excluded = [False] * self.alphabet_size
        self.last_support = 0
        self._probs[:] = self._distribution(context_tail, min(len(context_tail), self.max_order), excluded)
        total = float(np.sum(self._probs))
        if not np.isfinite(total) or total <= 0.0:
            self._probs.fill(1.0 / self.alphabet_size)
        else:
            self._probs /= total
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

    def _distribution(self, context_tail: tuple[int, ...], order: int, excluded: list[bool]) -> NDArray[np.float64]:
        out = np.zeros(self.alphabet_size, dtype=np.float64)
        if order < 0:
            avail = [i for i, is_excluded in enumerate(excluded) if not is_excluded]
            if avail:
                mass = 1.0 / len(avail)
                for sym in avail:
                    out[sym] = mass
            return out

        key = context_tail[-order:] if order > 0 else ()
        stats = self._tables[order].get(key)
        if stats is None or stats.total <= 0:
            return self._distribution(context_tail, order - 1, excluded)

        usable: list[tuple[int, int]] = []
        total_eff = 0
        for sym, count in stats.counts.items():
            if not excluded[sym]:
                usable.append((sym, count))
                total_eff += count
        if total_eff <= 0:
            return self._distribution(context_tail, order - 1, excluded)

        if order > 0 and total_eff > self.last_support:
            self.last_support = total_eff

        unseen_available = sum(1 for is_excluded in excluded if not is_excluded) - len(usable)
        if unseen_available <= 0:
            inv = 1.0 / total_eff
            for sym, count in usable:
                out[sym] = count * inv
            return out

        denom = float(total_eff + 1.0)
        inv = 1.0 / denom
        next_excluded = excluded.copy()
        for sym, count in usable:
            out[sym] = count * inv
            next_excluded[sym] = True
        lower = self._distribution(context_tail, order - 1, next_excluded)
        out += inv * lower
        return out

    def _tail_context(self, context: Sequence[int]) -> tuple[int, ...]:
        n = min(len(context), self.max_order)
        return tuple(int(x) for x in context[-n:]) if n > 0 else ()

    def _fit(self, train: NDArray[np.int64]) -> None:
        history: list[int] = []
        for raw in train:
            symbol = int(raw)
            self._add_observation(tuple(history[-self.max_order:]), symbol)
            history.append(symbol)

    def _add_observation(self, context_tail: tuple[int, ...], symbol: int) -> None:
        max_order = min(len(context_tail), self.max_order)
        for order in range(max_order + 1):
            key = context_tail[-order:] if order > 0 else ()
            stats = self._tables[order].get(key)
            if stats is None:
                stats = _Stats()
                self._tables[order][key] = stats
            stats.total += 1
            stats.counts[symbol] = stats.counts.get(symbol, 0) + 1


class PPMAProbVOMMHybridPredictor(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        train = np.asarray(np.load(DATA_DIR / "train.npy"), dtype=np.int64)
        self.base = build_base_predictor(alphabet_size, max_context_length)
        self.ppma = PPMAComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.ppma.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        ppma_probs = np.exp2(np.asarray(self.ppma.predict_next(context), dtype=np.float64))
        count_conf = self.ppma.last_support / (self.ppma.last_support + GATE_COUNT_C)
        w = PPMA_MAX_WEIGHT * count_conf * self.ppma.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * ppma_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.ppma.update(observed_symbol)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    return PPMAProbVOMMHybridPredictor(alphabet_size, max_context_length)
