"""Stronger disagreement-boosted order-7 count-escape PPM-A stack.

Run 043 showed that penalizing PPM-A/base disagreement removes useful signal.
This variant tests the opposite: retain the Run 037 experts and give the PPM-A
expert a tiny boost when it disagrees with the base, interpreted as potential
conditional information not already encoded by the base stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from submissions.order7_count_escape_ppma_stack import SelectivePPMACountEscapeOrder7Component
from submissions.ppma_selective_update_hybrid import (
    GATE_COUNT_C as PPMA_GATE_COUNT_C,
    PPMA_MAX_WEIGHT,
)
from submissions.selective_ctw_vomm_ppma_hybrid import SelectiveCTWVOMMBasePredictor


DATA_DIR = Path("data/generator")
DISAGREEMENT_BOOST = 0.16
MAX_WEIGHT_MULTIPLIER = 1.20


class StrongDisagreementBoostPPMAStack(Predictor):
    def __init__(self, alphabet_size: int, max_context_length: int) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        train = np.asarray(np.load(DATA_DIR / "train.npy"), dtype=np.int64)
        self.base = SelectiveCTWVOMMBasePredictor(alphabet_size, max_context_length, train)
        self.ppma = SelectivePPMACountEscapeOrder7Component(alphabet_size, max_context_length, train)
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

        mix = 0.5 * (base_probs + ppma_probs)
        js = 0.5 * float(np.sum(base_probs * (np.log2(np.maximum(base_probs, 1e-300)) - np.log2(np.maximum(mix, 1e-300)))))
        js += 0.5 * float(np.sum(ppma_probs * (np.log2(np.maximum(ppma_probs, 1e-300)) - np.log2(np.maximum(mix, 1e-300)))))
        multiplier = 1.0 + DISAGREEMENT_BOOST * min(max(js, 0.0), 1.0)
        if multiplier > MAX_WEIGHT_MULTIPLIER:
            multiplier = MAX_WEIGHT_MULTIPLIER
        w *= multiplier

        self._probs[:] = (1.0 - w) * base_probs + w * ppma_probs
        self._probs /= float(np.sum(self._probs))
        np.log2(self._probs, out=self._log_probs)
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        self.base.update(observed_symbol)
        self.ppma.update(observed_symbol)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    return StrongDisagreementBoostPPMAStack(alphabet_size, max_context_length)
