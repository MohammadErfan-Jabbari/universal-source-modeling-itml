"""n=5 equal mixture of fixed-C and adaptive-C predictors with smoothing 0.3.

This keeps the best n=5 model-mixture structure and further sharpens local
context distributions relative to smoothing 0.4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from competition.predictors.ngram import NGramPredictor


DATA_DIR = Path("data/generator")

NGRAM_N = 5
LAPLACE = 0.3
FIXED_C = 44.0
UNIQUE_C_SCALE = 4.0
MIX_WEIGHT_FIXED = 0.5


class MixedInterpolatedNGramPredictor(NGramPredictor):
    """Mixture of fixed-count and continuation-diversity interpolation rules."""

    def __init__(
        self,
        *args: Any,
        fixed_c: float = 44.0,
        unique_c_scale: float = 4.0,
        mix_weight_fixed: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if fixed_c <= 0.0:
            raise ValueError("fixed_c must be positive.")
        if unique_c_scale <= 0.0:
            raise ValueError("unique_c_scale must be positive.")
        if not 0.0 <= mix_weight_fixed <= 1.0:
            raise ValueError("mix_weight_fixed must be in [0, 1].")
        super().__init__(*args, **kwargs)
        self.fixed_c = float(fixed_c)
        self.unique_c_scale = float(unique_c_scale)
        self.mix_weight_fixed = float(mix_weight_fixed)
        self._fixed_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._adaptive_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._local_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)

    def predict_next(self, context) -> NDArray[np.float64]:  # type: ignore[override]
        context_tail = self._extract_tail_context(context)
        max_order = min(len(context_tail), self.n - 1)

        stats0 = self._counts_by_order[0].get(())
        self._fill_local_probs(stats0, self._fixed_probs_buffer)
        self._adaptive_probs_buffer[:] = self._fixed_probs_buffer

        for order in range(1, max_order + 1):
            key = context_tail[-order:]
            stats = self._counts_by_order[order].get(key)
            if stats is None:
                continue

            self._fill_local_probs(stats, self._local_probs_buffer)

            lam_fixed = stats.total / (stats.total + self.fixed_c)
            self._fixed_probs_buffer *= 1.0 - lam_fixed
            self._fixed_probs_buffer += lam_fixed * self._local_probs_buffer

            distinct = max(1, len(stats.counts))
            c_eff = self.unique_c_scale * distinct
            lam_adaptive = stats.total / (stats.total + c_eff)
            self._adaptive_probs_buffer *= 1.0 - lam_adaptive
            self._adaptive_probs_buffer += lam_adaptive * self._local_probs_buffer

        self._fixed_probs_buffer *= self.mix_weight_fixed
        self._fixed_probs_buffer += (1.0 - self.mix_weight_fixed) * self._adaptive_probs_buffer
        np.log2(self._fixed_probs_buffer, out=self._log_probs_buffer)

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
            "Update DATA_DIR in submissions/ngram_interpolated_n5_laplace03_model_mix.py."
        )
    arr = np.load(train_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D train sequence at {train_path}, got {arr.shape}.")
    return np.asarray(arr, dtype=np.int64)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    train = _load_train_sequence(DATA_DIR)

    predictor = MixedInterpolatedNGramPredictor(
        alphabet_size=alphabet_size,
        n=NGRAM_N,
        laplace=LAPLACE,
        max_context_length=max_context_length,
        adapt_online=True,
        fixed_c=FIXED_C,
        unique_c_scale=UNIQUE_C_SCALE,
        mix_weight_fixed=MIX_WEIGHT_FIXED,
    )
    predictor.fit(train)
    return predictor
