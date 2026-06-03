# 2. Methodology

The objective from section 1 is fixed: reduce redundancy `D(P‖Q)`, online, within budget. This
section explains the models we used to do it and the discipline that kept the results honest. The
numbers live in [`docs/03_results.md`](03_results.md) and the full ledger in
[`EXPERIMENTS.md`](../EXPERIMENTS.md); here we explain the ideas and why each step was taken.

## 2.1 A ladder of models

**Fixed-order n-gram (baseline).** Estimate `q(x | context)` from counts over the last `n−1`
symbols, with add-`α` (Laplace) smoothing so no probability is ever 0. Fast and standard, but
structurally limited: one fixed order cannot be both data-rich (low `n`, stable estimates) and
context-specific (high `n`, sharp predictions). The official reference (n=4, Laplace=1) scores
2.999 bits/symbol.

**Phase 1 — interpolated per-order mixture.** Rather than commit to one order, mix several:
`q = Σ_k λ_k q^{(k)}`, with per-order weights and an online auxiliary low-order model fed by a
sliding window of recent symbols. This is a hand-built approximation to a universal mixture — low
orders give a safe fallback when data is scarce, high orders sharpen once a context is common. The
Phase 1 best (Run 117) reaches **2.969228**, a 25.8 % reduction from the uniform prior. After 121
runs the n-gram design space was exhausted.

**Phase 2 — richer model classes.** Phase 2 dropped n-gram tuning and brought in three classical
universal-modeling tools, each used as a complementary correction gated into the Phase 1 model:

- **PPM** (Prediction by Partial Matching; Cleary & Witten 1984). Predict from the longest matching
  context; when the next symbol is novel there, emit an *escape* probability and back off to a
  shorter context. Escape mass is the model's explicit budget for "things not yet seen here" — an
  online way to keep redundancy finite at high orders. Variants A/C/D differ in how escape mass is
  computed; *exclusions* drop already-predicted symbols from the backoff. (Run 004 → 2.969019.)
- **VOMM** (Variable-Order Markov Model). Choose the context depth per position — deeper when
  well-supported, shallower otherwise. The probabilistic-depth variant mixes over depths weighted
  by support and prediction sharpness instead of hard-selecting one. (Run 014 → 2.968982.)
- **CTW** (Context Tree Weighting; Willems et al. 1995). The principled one: weight together every
  bounded-depth tree source via the Krichevsky–Trofimov estimator, with provable
  minimax-redundancy guarantees. The count-dependent mixing variant adapts the weighting to local
  support. (Run 010 → 2.968990.)

**The disagreement-boosted stack (best model, Run 067 → 2.968933).** The final predictor is an
order-7 count-escape PPM-A expert on a selective CTW/VOMM base, gated into the Phase 1 champion.
The one new idea is a *support-conditioned Jensen–Shannon disagreement boost*: the PPM-A expert's
weight is multiplied by `min(1 + c·JS·√(count-confidence), cap)`, i.e. boosted in proportion to
how strongly it *disagrees* with the base (the JS divergence between their distributions) and how
*well-supported* the context is. Disagreement on a well-supported context is a credible signal that
a longer-context expert has captured structure the cheap base misses; conditioning on support keeps
the boost from amplifying noisy experts on thin evidence, and the cap protects the safe fallback —
a deliberate trade between sharper predictions and the `+∞` cost of overconfidence.

## 2.2 Why the gains vanish — and why that is the result

Each Phase 2 family is strictly more expressive than fixed-order n-grams, yet the whole campaign
bought only ~3×10⁻⁴ bits over Phase 1. Section 3 shows why: the source's conditional entropy is
well-estimated only up to order ~3 given 3×10⁵ symbols, and our predictor already sits between the
order-2 and order-3 entropy. The remaining redundancy lives in high-order contexts that are
observed too rarely to estimate reliably *online* — exactly where more model expressiveness cannot
help without more data. Reporting this is the honest outcome; the contribution is the demonstration
that successive universal models converge to the source's finite-order entropy rate, not a heroic
last decimal.

## 2.3 Validity and selection discipline

Two checks guard every kept result (see [`scripts/`](../scripts) and
[`validation_script.py`](../validation_script.py)):

1. **Stability.** Score the predictor several times on the same block; `stability_std` must be ≈0
   (numerically `~4×10⁻¹⁶`, machine epsilon). Any real variance means hidden randomness or mutable
   state — a bug — and the run is discarded. This enforces that the predictor is a deterministic
   function of the past.
2. **Train-derived validation.** Score on a held-out tail of the training sequence, separate from
   the public-practice file, to watch for overfitting to the practice seed.

The train-tail score (~2.76) is consistently *below* the public-practice score (~2.97). It would be
easy to read this as a distribution shift, but it is not: the empirical entropy rate of the train
and public sequences is identical at every reliable order (section 3, e.g. order-2 entropy 3.0222
vs 3.0218). The gap is instead **in-sample vs out-of-sample** — the predictor seeds its counts from
`train.npy`, so scoring the train tail partly re-reads contexts already in its count tables, while
the public-practice sequence is genuinely held out. The honest performance number is the
out-of-sample one, 2.969.

All scores come only from the official `competition/run_live_eval.py`, treated as read-only.

## 2.4 The autonomous research loop

Phase 2's 67 runs were driven by an autonomous experiment loop ("autoresearch", see
[`autoresearch.md`](../autoresearch.md), [`autoresearch.jsonl`](../autoresearch.jsonl)). For each
idea the loop wrote an `explanations/run_###.md` motivation *before* running, executed the official
eval plus both validation checks, recorded the result, and kept the run only if it beat the standing
best and passed the stability and overfitting gates. Every number in the ledger therefore traces to
a logged run and a written hypothesis — the campaign is reproducible and auditable rather than a
blind sweep.

---

The numbers and the entropy-floor analysis that justify the claims here are in
[**3. Results**](03_results.md).

[README](../README.md) · [1. Problem](01_problem_and_information_theory.md) · [2. Methodology](02_methodology.md) · [3. Results](03_results.md) · [4. Skills](04_skills_and_learnings.md)

**References.** Cleary & Witten (1984), *Data Compression Using Adaptive Coding and Partial String
Matching*, IEEE Trans. Comm. Willems, Shtarkov & Tjalkens (1995), *The Context-Tree Weighting
Method*, IEEE Trans. IT. Krichevsky & Trofimov (1981), *The Performance of Universal Encoding*.
