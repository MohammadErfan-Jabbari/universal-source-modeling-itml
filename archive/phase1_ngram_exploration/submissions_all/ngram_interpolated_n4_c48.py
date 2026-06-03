"""Count-dependent interpolated n=4 n-gram predictor with C=48.

This variant brackets the interpolation constant between the current best C=32
and the slightly worse C=64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from competition.predictors.ngram import NGramPredictor


DATA_DIR = Path("data/generator")

NGRAM_N = 4
LAPLACE = 1.0
INTERP_C = 48.0


class InterpolatedNGramPredictor(NGramPredictor):
    """Recursive Jelinek-Mercer-style interpolation over suffix orders."""

    def __init__(self, *args: Any, interp_c: float = 48.0, **kwargs: Any) -> None:
        if interp_c <= 0.0:
            raise ValueError("interp_c must be positive.")
        super().__init__(*args, **kwargs)
        self.interp_c = float(interp_c)
        self._mix_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._local_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)

    def predict_next(self, context) -> NDArray[np.float64]:  # type: ignore[override]
        context_tail = self._extract_tail_context(context)
        max_order = min(len(context_tail), self.n - 1)

        stats0 = self._counts_by_order[0].get(())
        self._fill_local_probs(stats0, self._mix_probs_buffer)

        for order in range(1, max_order + 1):
            key = context_tail[-order:]
            stats = self._counts_by_order[order].get(key)
            if stats is None:
                continue
            self._fill_local_probs(stats, self._local_probs_buffer)
            lam = stats.total / (stats.total + self.interp_c)
            self._mix_probs_buffer *= 1.0 - lam
            self._mix_probs_buffer += lam * self._local_probs_buffer

        np.log2(self._mix_probs_buffer, out=self._log_probs_buffer)

        if self.adapt_online:
            self._pending_context_for_update = context_tail
        else:
            self._pending_context_for_update = None

        return self._log_probs_buffer

    def _fill_local_probs(self, stats, out: NDArray[np.float64]) -> None:
        total = 0 if stats is None else stats.total
        denom = (self.alphabet_size * self.laplace) + total
        out.fill(self.laplace / denom)
        if stats is not None and stats.counts:
            inv_denom = 1.0 / denom
            for symbol, count in stats.counts.items():
                out[symbol] = (count + self.laplace) * inv_denom


def _load_train_sequence(data_dir: Path) -> np.ndarray:
    train_path = data_dir / "train.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. "
            "Update DATA_DIR in submissions/ngram_interpolated_n4_c48.py."
        )
    arr = np.load(train_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D train sequence at {train_path}, got {arr.shape}.")
    return np.asarray(arr, dtype=np.int64)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    train = _load_train_sequence(DATA_DIR)

    predictor = InterpolatedNGramPredictor(
        alphabet_size=alphabet_size,
        n=NGRAM_N,
        laplace=LAPLACE,
        max_context_length=max_context_length,
        adapt_online=True,
        interp_c=INTERP_C,
    )
    predictor.fit(train)
    return predictor
