"""n=5 main model plus n=3 recent-window online model at weight 0.14 and window 262144.

The main model is the current best train+online n=5 alpha=0.3 mixture. The
online-only sliding-window component uses n=3 to reduce sparse recent-context
variance while preserving sequential validity; max recent weight is 0.14 and window is 262144.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from competition.predictors.base import Predictor
from competition.predictors.ngram import NGramPredictor


DATA_DIR = Path("data/generator")

NGRAM_N = 5
RECENT_NGRAM_N = 3
LAPLACE = 0.3
FIXED_C = 44.0
UNIQUE_C_SCALE = 4.0
MIX_WEIGHT_FIXED = 0.5
RECENT_WINDOW = 262144
RECENT_MAX_WEIGHT = 0.14
RECENT_WARMUP = 10000


class _Counts:
    """Sparse counts for recent-window statistics.

    Deliberately not a dataclass: student predictors are loaded with
    importlib.util.module_from_spec, and local dataclasses can fail before the
    module is inserted into sys.modules in this environment.
    """

    __slots__ = ("total", "counts")

    def __init__(self) -> None:
        self.total = 0
        self.counts: dict[int, int] = {}


class RecentMixedInterpolatedNGramPredictor(NGramPredictor):
    """Fixed/adaptive interpolated n-gram with a small recent-window hedge."""

    def __init__(
        self,
        *args: Any,
        fixed_c: float = 44.0,
        unique_c_scale: float = 4.0,
        mix_weight_fixed: float = 0.5,
        recent_window: int = 8192,
        recent_max_weight: float = 0.14,
        recent_warmup: int = 10000,
        recent_n: int = 3,
        **kwargs: Any,
    ) -> None:
        if fixed_c <= 0.0:
            raise ValueError("fixed_c must be positive.")
        if unique_c_scale <= 0.0:
            raise ValueError("unique_c_scale must be positive.")
        if not 0.0 <= mix_weight_fixed <= 1.0:
            raise ValueError("mix_weight_fixed must be in [0, 1].")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive.")
        if not 0.0 <= recent_max_weight <= 1.0:
            raise ValueError("recent_max_weight must be in [0, 1].")
        if recent_warmup <= 0:
            raise ValueError("recent_warmup must be positive.")
        if recent_n < 1:
            raise ValueError("recent_n must be positive.")
        super().__init__(*args, **kwargs)
        if recent_n > self.n:
            raise ValueError("recent_n must be no larger than the main n.")
        self.fixed_c = float(fixed_c)
        self.unique_c_scale = float(unique_c_scale)
        self.mix_weight_fixed = float(mix_weight_fixed)
        self.recent_window = int(recent_window)
        self.recent_max_weight = float(recent_max_weight)
        self.recent_warmup = int(recent_warmup)
        self.recent_n = int(recent_n)

        self._main_fixed_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._main_adaptive_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._recent_fixed_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._recent_adaptive_buffer = np.empty(self.alphabet_size, dtype=np.float64)
        self._local_probs_buffer = np.empty(self.alphabet_size, dtype=np.float64)

        self._recent_counts_by_order: list[dict[tuple[int, ...], _Counts]] = []
        self._recent_observations: deque[tuple[tuple[int, ...], int]] = deque()
        self._tokens_seen = 0
        self._reset_recent_counts()

    def initialize(self) -> None:
        super().initialize()
        self._reset_recent_counts()
        self._tokens_seen = 0

    def predict_next(self, context) -> NDArray[np.float64]:  # type: ignore[override]
        context_tail = self._extract_tail_context(context)
        main_max_order = min(len(context_tail), self.n - 1)
        recent_max_order = min(len(context_tail), self.recent_n - 1)

        self._compute_component_probs(
            self._counts_by_order,
            context_tail,
            main_max_order,
            self._main_fixed_buffer,
            self._main_adaptive_buffer,
        )
        self._compute_component_probs(
            self._recent_counts_by_order,
            context_tail,
            recent_max_order,
            self._recent_fixed_buffer,
            self._recent_adaptive_buffer,
        )

        self._main_fixed_buffer *= self.mix_weight_fixed
        self._main_fixed_buffer += (1.0 - self.mix_weight_fixed) * self._main_adaptive_buffer
        self._recent_fixed_buffer *= self.mix_weight_fixed
        self._recent_fixed_buffer += (1.0 - self.mix_weight_fixed) * self._recent_adaptive_buffer

        recent_weight = self.recent_max_weight * min(
            1.0, self._tokens_seen / self.recent_warmup
        )
        self._main_fixed_buffer *= 1.0 - recent_weight
        self._main_fixed_buffer += recent_weight * self._recent_fixed_buffer
        np.log2(self._main_fixed_buffer, out=self._log_probs_buffer)

        if self.adapt_online:
            self._pending_context_for_update = context_tail
        else:
            self._pending_context_for_update = None

        return self._log_probs_buffer

    def update(self, observed_symbol: int) -> None:
        symbol = int(observed_symbol)
        self._validate_symbol(symbol)
        if not self.adapt_online:
            return

        if self._pending_context_for_update is None:
            raise RuntimeError(
                "update() called before predict_next() or predictor state was not initialized."
            )

        context_tail = self._pending_context_for_update
        self._add_observation(context_tail=context_tail, symbol=symbol)
        self._add_recent_observation(context_tail=context_tail, symbol=symbol)
        self._pending_context_for_update = None
        self._is_fitted = True
        self._tokens_seen += 1

    def _compute_component_probs(
        self,
        tables,
        context_tail: tuple[int, ...],
        max_order: int,
        fixed_out: NDArray[np.float64],
        adaptive_out: NDArray[np.float64],
    ) -> None:
        stats0 = tables[0].get(())
        self._fill_local_probs(stats0, fixed_out)
        adaptive_out[:] = fixed_out

        for order in range(1, max_order + 1):
            key = context_tail[-order:]
            stats = tables[order].get(key)
            if stats is None:
                continue

            self._fill_local_probs(stats, self._local_probs_buffer)

            lam_fixed = stats.total / (stats.total + self.fixed_c)
            fixed_out *= 1.0 - lam_fixed
            fixed_out += lam_fixed * self._local_probs_buffer

            distinct = max(1, len(stats.counts))
            c_eff = self.unique_c_scale * distinct
            lam_adaptive = stats.total / (stats.total + c_eff)
            adaptive_out *= 1.0 - lam_adaptive
            adaptive_out += lam_adaptive * self._local_probs_buffer

    def _fill_local_probs(self, stats, out: NDArray[np.float64]) -> None:
        total = 0 if stats is None else stats.total
        denom = (self.alphabet_size * self.laplace) + total
        out.fill(self.laplace / denom)
        if stats is not None and stats.counts:
            inv_denom = 1.0 / denom
            for symbol, count in stats.counts.items():
                out[symbol] = (count + self.laplace) * inv_denom

    def _reset_recent_counts(self) -> None:
        self._recent_counts_by_order = [{} for _ in range(self.recent_n)]
        self._recent_observations = deque()

    def _add_recent_observation(self, *, context_tail: tuple[int, ...], symbol: int) -> None:
        self._recent_observations.append((context_tail, symbol))
        self._adjust_recent_observation(context_tail=context_tail, symbol=symbol, delta=1)
        while len(self._recent_observations) > self.recent_window:
            old_context_tail, old_symbol = self._recent_observations.popleft()
            self._adjust_recent_observation(
                context_tail=old_context_tail, symbol=old_symbol, delta=-1
            )

    def _adjust_recent_observation(
        self, *, context_tail: tuple[int, ...], symbol: int, delta: int
    ) -> None:
        max_order = min(len(context_tail), self.recent_n - 1)
        for order in range(max_order + 1):
            key = context_tail[-order:] if order > 0 else ()
            table = self._recent_counts_by_order[order]
            stats = table.get(key)
            if stats is None:
                if delta < 0:
                    raise RuntimeError("recent count underflow: missing context")
                stats = _Counts()
                table[key] = stats

            stats.total += delta
            new_count = stats.counts.get(symbol, 0) + delta
            if new_count > 0:
                stats.counts[symbol] = new_count
            else:
                stats.counts.pop(symbol, None)

            if stats.total <= 0:
                if stats.total < 0 or stats.counts:
                    raise RuntimeError("recent count underflow")
                table.pop(key, None)


def _load_train_sequence(data_dir: Path) -> np.ndarray:
    train_path = data_dir / "train.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. "
            "Update DATA_DIR in submissions/ngram_interpolated_n5_laplace03_recent_n3_w14_win262144.py."
        )
    arr = np.load(train_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D train sequence at {train_path}, got {arr.shape}.")
    return np.asarray(arr, dtype=np.int64)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    train = _load_train_sequence(DATA_DIR)

    predictor = RecentMixedInterpolatedNGramPredictor(
        alphabet_size=alphabet_size,
        n=NGRAM_N,
        laplace=LAPLACE,
        max_context_length=max_context_length,
        adapt_online=True,
        fixed_c=FIXED_C,
        unique_c_scale=UNIQUE_C_SCALE,
        mix_weight_fixed=MIX_WEIGHT_FIXED,
        recent_window=RECENT_WINDOW,
        recent_max_weight=RECENT_MAX_WEIGHT,
        recent_warmup=RECENT_WARMUP,
        recent_n=RECENT_NGRAM_N,
    )
    predictor.fit(train)
    return predictor
