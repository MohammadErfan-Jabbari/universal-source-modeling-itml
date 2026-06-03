"""Official n-gram baseline (n=4, Laplace=1.0, hard backoff). Score: ≈2.9986 bits/symbol.

INFORMATION-THEORETIC CONTEXT
==============================
The task is online sequential prediction over an alphabet A = {0,...,15}. At each
step i the predictor emits a distribution q_i(· | x_1^{i-1}) and is penalised by
  −log2 q_i(x_i | x_1^{i-1})   bits  (the ideal codelength under q_i).
The empirical average across N symbols is the empirical cross-entropy:
  Ĥ_Q(x_1^N) = (1/N) Σ −log2 q_i(x_i | x_1^{i-1}).
By the chain rule of information, this equals the true source entropy H(P) plus the
KL/redundancy term D(P‖Q), so every improvement in modelling P reduces D(P‖Q) and
therefore reduces the average codelength.

WHAT THIS MODEL DOES
====================
Fits a fixed-order n=4 Markov model from train.npy using Laplace (add-α=1.0)
smoothing. At prediction time it looks up the (up to) 3-symbol context in the count
table; if unseen it backs off to shorter contexts (hard backoff). Online adaptation
(adapt_online=True) allows the count table to grow during evaluation, incrementally
updating q_i after each revealed x_i in a strictly causal manner.

SEQUENTIAL / CAUSAL VALIDITY
=============================
Causal validity is guaranteed because the harness passes only x_1^{i-1} to
predict_next and reveals x_i only through update(x_i). The model never conditions on
future symbols.

WHY IT FALLS SHORT
==================
A fixed-order Markov model of order 4 can only exploit recurrences within a 4-symbol
window. The source may have longer dependencies; this model pays a redundancy penalty
D(P‖Q) for every such dependency it ignores. The Laplace prior also misallocates mass
to unseen symbols.

MEASURED SCORE
==============
≈2.9986 bits/symbol (full 200 000-symbol run). Uniform baseline is 4.0 bps;
this model already exploits the n-gram structure of the synthetic source.

Student submission template for the Universal Source Modeling Challenge.

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
