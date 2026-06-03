# 4. What This Project Demonstrates

A short, honest accounting of the competencies behind the work — written to be checked against
the code and results, not taken on faith. We separate what was *applied* in the predictors from
what was *studied* in the course and informs the framing, because conflating the two would be the
easy way to oversell.

## 4.1 Research method (the part that matters most)

The project is small in surface area on purpose, and the discipline is the point:

- **Problem formalization.** The task is posed in its proper information-theoretic frame —
  log-loss as ideal code length, `H(P,Q) = H(P) + D(P‖Q)`, redundancy as model mismatch — and
  every design decision is justified against it (section 1).
- **Controlled, auditable experimentation.** 188 runs across two phases, each with a
  hypothesis written *before* it ran, a deterministic evaluator, and explicit gates: a stability
  check (`stability_std ≈ 0`) and a train-derived validation split to catch seed overfitting
  (section 2.3). The whole campaign is reproducible from logged runs.
- **Honest interpretation.** The headline result is a *negative* one — 67 models bought
  ~3×10⁻⁴ bits — and we treat it as a finding, estimating the source's entropy rate directly to
  show the predictor is at the achievable floor (section 3). Knowing when to stop, and proving
  why, is a research skill.
- **Research tooling.** An autonomous experiment loop that proposes, runs, validates, and logs
  experiments — infrastructure built to make a long search trustworthy rather than a blind sweep.

## 4.2 Information theory and coding — *applied*

Entropy, cross-entropy, and KL divergence as the optimization target; the prediction↔coding
duality used to justify scoring log-loss; arithmetic coding as the realizability link behind
Activity B's metrics; universal compression (Lempel–Ziv, minimax redundancy) as the motivation
for the model families, with **Context Tree Weighting** implemented as the principled
minimax-redundancy approach; and smoothing/escape understood as keeping `D(P‖Q)` finite.

## 4.3 Probabilistic sequence modeling — *applied*

Fixed-order n-grams with add-`α` smoothing and interpolated per-order mixtures; Variable-Order
Markov Models; Context Tree Weighting (Krichevsky–Trofimov estimator); PPM-A/C/D with escape and
exclusion; and online expert mixing with confidence gating and a Jensen–Shannon disagreement
signal for allocating trust between experts.

## 4.4 Engineering — *applied*

Online, strictly causal algorithm design under a 600 s / 256-context budget in NumPy only; cheap
incremental count updates rather than heavy per-step computation; determinism and reproducibility
discipline; GPU LLM inference across a model family (distilgpt2 → Qwen2.5-7B) with `transformers`,
normalizing token log-loss to bits/character; and `uv`-managed, reproducible pipelines with
committed figure-generation scripts.

## 4.5 Broader ML — *studied, not coded here*

Fully-observed and latent-variable models, EM and Gaussian mixtures, variational inference and
VAEs, the reparameterization trick, and information-theoretic generalization bounds appear in the
course and shaped the information-theoretic mindset of this work, but were not part of the
Activity A predictor or the Activity B pipeline. They are listed here as context, not as claims.

## 4.6 Three transferable lessons

1. **Pick the metric that *is* the goal.** Scoring in bits/symbol made "better model" mean
   exactly "lower KL divergence," which kept every experiment honest.
2. **Measure the floor before chasing it.** Estimating `H(P)` from the data turned vanishing
   gains from a frustration into the result.
3. **Make the search auditable.** Hypothesis-first write-ups, deterministic validation, and
   logged runs are what make a long campaign trustworthy.

---

Back to [**README**](../README.md) · [**1. Problem & theory**](01_problem_and_information_theory.md)
· [**2. Methodology**](02_methodology.md) · [**3. Results**](03_results.md)
