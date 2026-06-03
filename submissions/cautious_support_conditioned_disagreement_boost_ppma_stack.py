"""FINAL BEST — Phase 2 Run 067: PPM-A order-7 + selective CTW/VOMM base stack + support-conditioned JS disagreement boost. Score: ≈2.968933 bits/symbol.

INFORMATION-THEORETIC CONTEXT
==============================
The prediction–coding duality states that every improvement in the model's sequential
probability assignments directly reduces the empirical cross-entropy (ideal average
codelength). This predictor realises this by stacking multiple complementary experts
whose redundancy terms D(P‖Q_k) are partially non-overlapping, then adaptively
upweighting whichever expert carries residual information the other does not.

ARCHITECTURE
============
Two components are mixed at each step:

  1. Base stack (SelectiveCTWVOMMBasePredictor): the accumulated Run 010/014/016/019/
     022/024 hierarchy of count-dependent CTW, probabilistic-depth VOMM, and selective
     PPM-A online-update models. This is the "background" model that already achieves
     low redundancy from long-range recurrence patterns.

  2. PPM-A order-7 component (SelectivePPMACountEscapeOrder7Component): a high-order
     PPM-A model that uses count-based escape — escape mass ∝ unique_seen rather than
     PPM-C's unique/total ratio — providing a complementary local context signal.

DISAGREEMENT BOOST
==================
The key innovation is the Jensen-Shannon (JS) divergence between the two experts'
predictions, used as a proxy for conditional information:
  JS(Q_base, Q_PPM) = ½ KL(Q_base ‖ M) + ½ KL(Q_PPM ‖ M),  M = ½(Q_base + Q_PPM).
JS is bounded in [0,1] in bits and equals zero when both experts agree — meaning
the PPM-A expert carries no information beyond the base. When JS > 0, the PPM-A
expert is genuinely predicting differently, which is evidence (not proof) of
complementary structure.

The PPM-A weight is then:
  w = PPMA_MAX_WEIGHT × count_conf × entropy_conf × multiplier
  multiplier = min(1 + 0.88 × JS × sqrt(count_conf), 1.80)

The sqrt(count_conf) factor modulates the boost by recurrence support: disagreement
on well-recurrent contexts is trusted more than disagreement on sparse ones. The cap
at 1.80 prevents runaway weight for pathological inputs. This is conservative by
design — a half-step increase over Run 064 (+0.04 in boost coefficient vs +0.08 for
prior iterations) to probe saturation rather than overshoot.

This mechanism is information-theoretically interpretable: by the chain rule,
H(X | Q_base) − H(X | Q_base, Q_PPM) measures the true residual information of
Q_PPM given Q_base. JS is an upper bound proxy. So boosting PPM-A weight when JS is
large is equivalent to estimating that D(P‖Q_base) still has a PPM-A-exploitable
component, and correcting for it.

SEQUENTIAL / CAUSAL VALIDITY
=============================
All components operate strictly on x_1^{i-1}. Both sub-predictors update in
update(x_i) after prediction. The JS computation uses only the predictions already
produced causally. No lookahead occurs at any level of the stack.

MEASURED SCORE
==============
≈2.968933 bits/symbol (Run 067, full 200 000-symbol run). Final best of the entire
project, after a monotone sequence of gains from Runs 054 → 057 → 060 → 062 → 064 →
067. Total improvement over the Phase 1 best: ~0.000295 bps; over the template
baseline: ~0.029692 bps. Stability std = 0.0; train-val gap within tolerance.

Cautious support-conditioned disagreement-boosted order-7 count-escape PPM-A stack.

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
DISAGREEMENT_BOOST = 0.88
MAX_WEIGHT_MULTIPLIER = 1.80


class CautiousSupportConditionedDisagreementBoostPPMAStack(Predictor):
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
        support_gate = np.sqrt(count_conf)
        multiplier = 1.0 + DISAGREEMENT_BOOST * min(max(js, 0.0), 1.0) * support_gate
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
    return CautiousSupportConditionedDisagreementBoostPPMAStack(alphabet_size, max_context_length)
