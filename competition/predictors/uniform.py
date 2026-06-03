"""Uniform baseline predictor."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor


class UniformPredictor(Predictor):
    """Predicts a uniform distribution over the alphabet at every step."""

    def __init__(self, alphabet_size: int, max_context_length: int | None = None) -> None:
        super().__init__(alphabet_size=alphabet_size, max_context_length=max_context_length)
        log_prob = -np.log2(float(self.alphabet_size))
        self._log_probs = np.full(self.alphabet_size, log_prob, dtype=np.float64)

    def initialize(self) -> None:
        """No internal state to reset."""

    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        del context
        return self._log_probs

    def update(self, observed_symbol: int) -> None:
        del observed_symbol
