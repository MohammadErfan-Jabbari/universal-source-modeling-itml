"""Abstract predictor interface for sequential source modeling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class Predictor(ABC):
    """Base class for all sequential predictors used in the competition.

    Predictors must operate strictly online:
    - `predict_next` receives only past symbols (already observed).
    - `update` receives the newly revealed symbol after evaluation.
    - No lookahead is allowed.

    Notes:
    - `context` may be a lightweight, mutable read-only sequence view supplied by
      the evaluation harness for performance. Treat it as ephemeral and do not
      store it beyond the scope of `predict_next`.
    """

    def __init__(self, alphabet_size: int, max_context_length: int | None = None) -> None:
        if alphabet_size <= 0:
            raise ValueError("alphabet_size must be positive.")
        if max_context_length is not None and max_context_length < 0:
            raise ValueError("max_context_length must be non-negative or None.")
        self.alphabet_size = int(alphabet_size)
        self.max_context_length = max_context_length

    @abstractmethod
    def initialize(self) -> None:
        """Reset internal state before evaluating a new sequence."""

    @abstractmethod
    def predict_next(self, context: Sequence[int]) -> NDArray[np.float64]:
        """Return log2 probabilities for the next symbol.

        Args:
            context: Truncated past symbols in chronological order. This may be a
                lightweight sequence view rather than a tuple/list.

        Returns:
            A length-`alphabet_size` array of log2 probabilities.
        """

    @abstractmethod
    def update(self, observed_symbol: int) -> None:
        """Update internal state after the next symbol is revealed."""

    def reset(self) -> None:
        """Alias for `initialize` for convenience."""
        self.initialize()
