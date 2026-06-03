"""Student submission template for the Universal Source Modeling Challenge.

Competition-day contract:
- This file must define `build_predictor(alphabet_size, max_context_length)`.
- The returned object must be a `Predictor`.
- Prediction is strictly online/sequential: no lookahead.
- The harness enforces the context limit and runtime limit on competition day.

Template baseline:
- Loads `train.npy` from `DATA_DIR`
- Fits a hard-backoff Laplace-smoothed n-gram (`n=4`)
- Returns the fitted predictor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from competition.predictors.base import Predictor
from competition.predictors.ngram import NGramPredictor


# Change this path if your training data is stored somewhere else.
DATA_DIR = Path("data/generator")

# Simple starter baseline (instructor-provided reference style).
NGRAM_N = 4
LAPLACE = 1.0


def _load_train_sequence(data_dir: Path) -> np.ndarray:
    train_path = data_dir / "train.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. "
            "Update DATA_DIR in submissions/template_predictor.py."
        )
    arr = np.load(train_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D train sequence at {train_path}, got {arr.shape}.")
    return np.asarray(arr, dtype=np.int64)


def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    """Build and return a sequential predictor for live evaluation.

    Notes for students:
    - You do not get access to future test symbols.
    - The harness will pass only the allowed context window (max length enforced).
    - Competition-day runtime is capped, so keep model loading/inference efficient.
    """

    train = _load_train_sequence(DATA_DIR)

    predictor = NGramPredictor(
        alphabet_size=alphabet_size,
        n=NGRAM_N,
        laplace=LAPLACE,
        max_context_length=max_context_length,
        adapt_online=True,
    )
    predictor.fit(train)
    return predictor
