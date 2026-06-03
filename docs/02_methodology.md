# 2. Methodology

How the predictor was built, family by family, each motivated by the goal of section 1:
reduce redundancy `D(P‖Q)` online, within budget. The complete numerical record is in
[`EXPERIMENTS.md`](../EXPERIMENTS.md) and [`docs/03_results.md`](03_results.md); this
document explains the *ideas* and *why each step was taken*.

## 2.1 The ladder of models

### Baseline — fixed-order n-gram with Laplace smoothing
Estimate `q(x | context)` from counts of the last `n−1` symbols, with add-`α` smoothing so
no probability is ever 0. Simple, fast, and the official reference (n=4, Laplace=1 →
**2.998625** bps). Its weakness is structural: a single fixed order `n` cannot be both
data-rich (low `n`, stable estimates) and context-specific (high `n`, sharp predictions).

### Phase 1 — interpolated per-order mixture
Mix several orders together rather than committing to one:
`q = Σ_k λ_k · q^{(k)}`, with per-order weights and an online auxiliary low-order model fed
by a sliding window of recent symbols. This is a hand-rolled approximation to a universal
mixture: low orders provide a safe fallback (small redundancy when data is scarce), high
orders sharpen when the context has been seen often. **Phase 1 best (Run 117): 2.969228**
bps — a 25.8 % reduction from the uniform prior. 121 runs exhausted the n-gram design space.

### Phase 2 — structurally richer model classes
Phase 2 left n-gram tuning behind and brought in three classical universal-modeling tools,
each used as a *complementary correction* gated into the Phase 1 model:

- **PPM (Prediction by Partial Matching).** Predict from the longest matching context; when
  the next symbol is novel there, emit an **escape** probability and *back off* to a shorter
  context. Escape mass is the model's explicit budget for "things I haven't seen here yet" —
  a direct, online way to keep redundancy finite at high orders. Variants PPM-A/C/D differ in
  how escape mass is computed; **exclusions** remove already-predicted symbols from the
  backoff distribution. (Run 004: confidence-gated PPM-C → 2.969019.)

- **VOMM (Variable-Order Markov Model).** Instead of one fixed order, *choose* the context
  depth per position — deeper when it is well-supported, shallower when not. The
  probabilistic-depth variant mixes over depths weighted by count support and prediction
  sharpness rather than hard-selecting one. (Run 014: prob-depth VOMM hybrid → 2.968982.)

- **CTW (Context Tree Weighting).** The theoretically principled one: maintain a context
  tree and *weight together every bounded-depth tree source* via the Krichevsky–Trofimov
  estimator, with provable minimax-redundancy guarantees. The count-dependent mixing variant
  adapts the weighting to local support. (Run 010: count-dependent CTW → 2.968990.)

### The disagreement-boosted stack — the best model
The Phase 2 best (**Run 067, 2.968933**) layers these together and adds one new idea. The
final predictor is an order-7 count-escape PPM-A stacked on a selective CTW/VOMM base, all
gated into the Phase 1 champion. The novel component is a **support-conditioned
Jensen–Shannon disagreement boost**: the PPM-A expert's weight is multiplied by
`min(1 + c·JS·√(count-confidence), cap)`, i.e. boosted in proportion to (a) how strongly
the expert *disagrees* with the base model (the JS divergence between their distributions)
and (b) how *well-supported* the context is (the `√count-confidence` factor grows with the
amount of evidence), capped so it never runs away.

The information-theoretic reading: disagreement on a well-supported context is a credible
signal that a longer-context expert has captured specific structure the cheap base model
misses; trusting it more there extracts a little additional mutual information from the
past. Conditioning on support (rather than boosting on every disagreement) keeps the
predictor from amplifying noisy experts on thin evidence. The cap ensures the safe fallback
is never collapsed — a deliberate bias/variance trade between sharper predictions and
protection against overconfidence (which, recall, costs `+∞` if a zeroed symbol appears).

## 2.2 Why the gains shrink to nothing — and why that is the result

Each Phase 2 family is strictly more expressive than fixed-order n-grams, yet stacking them
bought only ~3×10⁻⁴ bps over Phase 1. This is the expected signature of having reached the
model class's **entropy floor** for this source: the redundancy `D(P‖Q)` has been squeezed
to a few ten-thousandths of a bit, below the resolution distinguishable from noise at
`N = 200,000`. Reporting this is the honest scientific outcome — the project's value is the
*demonstration* that successive universal models converge, not a heroic last decimal.

## 2.3 Validity and selection discipline

Two checks guard every kept result (see [`scripts/`](../scripts) and
[`validation_script.py`](../validation_script.py)):

1. **Stability.** Score the predictor multiple times on the same block; `stability_std`
   must be ≈0 (numerically, machine-epsilon `~4×10⁻¹⁶`). Non-zero variance means hidden
   randomness or mutable global state — a bug, and the run is discarded. This enforces that
   the model is a deterministic function of the past, as required.

2. **Train-derived validation.** Score on a held-out suffix of the training sequence
   (after a warmup), separate from the public practice file, to detect overfitting to the
   practice seed. The persistent ~0.21 bps gap between train-suffix (~2.76) and public
   (~2.97) is consistent with a slightly different marginal distribution, *not* practice-set
   overfitting.

All scores are produced only by the official `competition/run_live_eval.py`; the evaluator
is treated as read-only.

## 2.4 The autonomous research loop

Phase 2's 67 runs were driven by an **autonomous experiment loop** ("autoresearch", see
[`autoresearch.md`](../autoresearch.md), [`autoresearch.jsonl`](../autoresearch.jsonl)). For
each idea the loop: wrote an `explanations/run_###.md` motivation *before* running, executed
the official eval plus both validation checks, recorded the result, and kept the run only if
it beat the standing best *and* passed stability/overfitting gates. This made the search
reproducible and auditable — every number in the ledger traces to a logged run and a written
hypothesis. The loop itself is a demonstration of disciplined, information-theoretically
grounded experimentation, not just a hyperparameter sweep.

---

Next: [**3. Results**](03_results.md) — the full tables and figures · or
[**4. Skills & learnings**](04_skills_and_learnings.md).
