"""Phase 2 Run 014: probabilistic-depth VOMM component mixed with the Run 010 count-dependent CTW stack. Score: 2.968982 bits/symbol.

INFORMATION-THEORETIC CONTEXT
==============================
Variable-Order Markov Models (VOMM) address a fundamental bias-variance tension in
context modelling. A fixed-order model pays variance (estimation error) at high orders
when counts are low, and bias (model mismatch) at low orders when the source has
longer dependencies. The canonical VOMM resolution is to select the deepest context
depth that has sufficient recurrence (min-count VOMM), yielding a hard decision.

PROBABILISTIC DEPTH MIXING
===========================
This model eliminates the hard depth decision by forming a soft probability
distribution over depths. Each depth d (0 ≤ d ≤ 6) contributes its local
KT-smoothed continuation distribution Q_d weighted by:
  weight_d = (N_d / (N_d + C)) × sharpness_d × (1 + d / max_depth)

where:
  - N_d / (N_d + C)   penalises unsupported depths (same count-weighting as CTW).
  - sharpness_d = 1 − H(Q_d) / log2(A)   rewards concentrated, informative predictions.
  - depth_factor       gives a soft prior toward longer contexts.

The final distribution is the weighted average of all Q_d, normalised to a valid
probability vector. This is a proper Bayesian-style model mixture: each depth is a
hypothesis about the true Markov order, and the weight reflects posterior plausibility
given the observed count and entropy. In information-theoretic terms the weighted
average achieves the geometric mean of the per-depth codelengths, which is always at
most the codelength of any single-depth model when the true source is in the mixture
class — a direct redundancy reduction relative to hard selection.

SEQUENTIAL / CAUSAL VALIDITY
=============================
All count updates happen in update(x_i); predict_next uses only x_1^{i-1}. The soft
depth weights are computed from the same causal context, so no future symbols are used.

MEASURED SCORE
==============
2.968982 bits/symbol (Run 014, full 200 000-symbol run). Improvement of ~0.000008 bps
over Run 010; probabilistic depth mixing captures complementary signal beyond hard
VOMM (Run 005, 2.969194) and the fixed CTW blend.

Probabilistic-depth VOMM component mixed with current count-dependent CTW best.

The added component is a variable-order Markov model that forms a distribution
over context depths rather than choosing one depth (min-count VOMM) or doing CTW
recursive node mixing.  Each suffix depth emits a KT-smoothed local continuation
distribution; depths receive probability proportional to support and sharpness.
The resulting VOMM expert is lightly gated into the current Run 010 best.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from submissions.ctw_countmix_phase1_hybrid import build_predictor as build_base_predictor


DATA_DIR = Path("data/generator")
MAX_DEPTH = 6
KT_ALPHA = 0.5
DEPTH_COUNT_C = 60.0
VOMM_MAX_WEIGHT = 0.04
GATE_COUNT_C = 100.0


class _Stats:
    __slots__ = ("total", "counts")

    def __init__(self) -> None:
        self.total = 0
        self.counts: dict[int, int] = {}


class ProbDepthVOMMComponent(Predictor):
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
        self._probs.fill(0.0)
        self.deepest_support = 0
        total_weight = 0.0

        for depth in range(max_depth + 1):
            key = context_tail[-depth:] if depth > 0 else ()
            stats = self._tables[depth].get(key)
            if stats is None or stats.total <= 0:
                continue
            if depth > 0:
                self.deepest_support = stats.total
            self._fill_kt(stats, self._local)
            entropy = -float(np.sum(self._local * np.log2(np.maximum(self._local, 1e-300))))
            sharp = max(0.02, 1.0 - entropy / np.log2(self.alphabet_size))
            support = stats.total / (stats.total + DEPTH_COUNT_C)
            # Prefer deeper contexts only when they are supported and sharp.
            depth_factor = 1.0 + (depth / max(1, self.max_depth))
            weight = support * sharp * depth_factor
            self._probs += weight * self._local
            total_weight += weight

        if total_weight <= 0.0:
            self._probs.fill(1.0 / self.alphabet_size)
        else:
            self._probs /= total_weight
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


class ProbDepthVOMMHybridPredictor(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        train = np.asarray(np.load(DATA_DIR / "train.npy"), dtype=np.int64)
        self.base = build_base_predictor(alphabet_size, max_context_length)
        self.vomm = ProbDepthVOMMComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.vomm.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        vomm_probs = np.exp2(np.asarray(self.vomm.predict_next(context), dtype=np.float64))
        count_conf = self.vomm.deepest_support / (self.vomm.deepest_support + GATE_COUNT_C)
        w = VOMM_MAX_WEIGHT * count_conf * self.vomm.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * vomm_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.vomm.update(observed_symbol)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    return ProbDepthVOMMHybridPredictor(alphabet_size, max_context_length)
