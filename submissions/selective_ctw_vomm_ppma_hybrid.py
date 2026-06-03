"""Selective-update CTW, VOMM, and PPM-A stack.

Run 022 confirmed that selective online updates help the probabilistic-depth
VOMM and PPM-A components, while Run 021 showed they do not help PPM-C.  This
variant keeps the PPM-C/Phase1 base unchanged, but applies the same selective
online update policy to the count-dependent CTW component: train fitting uses all
context depths, evaluation updates only depth 0 and the deepest available depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from submissions.ppm_phase1_hybrid import build_predictor as build_ppm_base_predictor
from submissions.ctw_countmix_phase1_hybrid import (
    CTW_MAX_WEIGHT,
    GATE_COUNT_C as CTW_GATE_COUNT_C,
    CountMixCTWComponent,
    _Stats as CTWStats,
)
from submissions.prob_depth_vomm_countmix_hybrid import (
    GATE_COUNT_C as VOMM_GATE_COUNT_C,
    VOMM_MAX_WEIGHT,
)
from submissions.selective_vomm_ppma_hybrid import SelectiveProbDepthVOMMComponent
from submissions.ppma_selective_update_hybrid import (
    GATE_COUNT_C as PPMA_GATE_COUNT_C,
    PPMA_MAX_WEIGHT,
    SelectivePPMAComponent,
)


DATA_DIR = Path("data/generator")


class SelectiveCountMixCTWComponent(CountMixCTWComponent):
    """Count-dependent CTW with selective online updates."""

    def update(self, observed_symbol: int) -> None:
        if self._pending_context is None:
            raise RuntimeError("update() called before predict_next().")
        self._add_online_observation(self._pending_context, int(observed_symbol))
        self._pending_context = None

    def _add_online_observation(self, context_tail: tuple[int, ...], symbol: int) -> None:
        self._add_at_depth(context_tail, symbol, 0)
        deepest = min(len(context_tail), self.max_depth)
        if deepest > 0:
            self._add_at_depth(context_tail, symbol, deepest)

    def _add_at_depth(self, context_tail: tuple[int, ...], symbol: int, depth: int) -> None:
        key = context_tail[-depth:] if depth > 0 else ()
        stats = self._tables[depth].get(key)
        if stats is None:
            stats = CTWStats()
            self._tables[depth][key] = stats
        stats.total += 1
        stats.counts[symbol] = stats.counts.get(symbol, 0) + 1


class SelectiveCTWBasePredictor(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int, train: NDArray[np.int64]) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        self.base = build_ppm_base_predictor(alphabet_size, max_context_length)
        self.ctw = SelectiveCountMixCTWComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.ctw.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        ctw_probs = np.exp2(np.asarray(self.ctw.predict_next(context), dtype=np.float64))
        count_conf = self.ctw.deepest_support / (self.ctw.deepest_support + CTW_GATE_COUNT_C)
        w = CTW_MAX_WEIGHT * count_conf * self.ctw.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * ctw_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.ctw.update(observed_symbol)


class SelectiveCTWVOMMBasePredictor(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int, train: NDArray[np.int64]) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        self.base = SelectiveCTWBasePredictor(alphabet_size, max_context_length, train)
        self.vomm = SelectiveProbDepthVOMMComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.vomm.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        vomm_probs = np.exp2(np.asarray(self.vomm.predict_next(context), dtype=np.float64))
        count_conf = self.vomm.deepest_support / (self.vomm.deepest_support + VOMM_GATE_COUNT_C)
        w = VOMM_MAX_WEIGHT * count_conf * self.vomm.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * vomm_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.vomm.update(observed_symbol)


class SelectiveCTWVOMMPPMAHybrid(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        train = np.asarray(np.load(DATA_DIR / "train.npy"), dtype=np.int64)
        self.base = SelectiveCTWVOMMBasePredictor(alphabet_size, max_context_length, train)
        self.ppma = SelectivePPMAComponent(alphabet_size, max_context_length, train)
        self._probs = np.empty(self.alphabet_size, dtype=np.float64)
        self._log_probs = np.empty(self.alphabet_size, dtype=np.float64)

    def initialize(self) -> None:
        self.base.initialize()
        self.ppma.initialize()

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        base_probs = np.exp2(np.asarray(self.base.predict_next(context), dtype=np.float64))
        ppma_probs = np.exp2(np.asarray(self.ppma.predict_next(context), dtype=np.float64))
        count_conf = self.ppma.last_support / (self.ppma.last_support + PPMA_GATE_COUNT_C)
        w = PPMA_MAX_WEIGHT * count_conf * self.ppma.entropy_confidence
        self._probs[:] = (1.0 - w) * base_probs + w * ppma_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.ppma.update(observed_symbol)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    return SelectiveCTWVOMMPPMAHybrid(alphabet_size, max_context_length)
