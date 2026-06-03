# 4. Skills and Learnings

What this project demonstrates, mapped to the *Information Theory for ML* curriculum. The
distinction below is deliberate and honest: some topics were **applied** in the predictor;
others were **studied** in the course and inform the framing but were not directly coded.

## 4.1 Information theory & coding — *applied*

- **Entropy, cross-entropy, relative entropy (KL).** The entire objective is empirical
  cross-entropy; the optimization target is the KL/redundancy term `D(P‖Q)` in
  `H(P,Q) = H(P) + D(P‖Q)`. Reasoned about quantitatively, not just defined.
- **Source coding & the prediction↔coding duality.** Used the fact that `−log₂ q(x)` is the
  ideal code length to justify scoring log-loss instead of a compressed file size.
- **Arithmetic coding.** The realizability link behind Activity B's "ideal code length"
  metric and the LLMZip rank-stream scheme.
- **Universal compression — Lempel–Ziv, minimax redundancy.** The motivation for the model
  families: achieve near-`H(P)` performance without knowing `P`. **Context Tree Weighting**
  was implemented as the principled minimax-redundancy approach.
- **Smoothing / escape as redundancy control.** Laplace smoothing and PPM escape mechanisms
  understood as keeping `D(P‖Q)` finite (never assign probability 0).

## 4.2 Probabilistic sequence modeling — *applied*

- Fixed-order n-grams; add-`α` (Laplace) smoothing; interpolated per-order mixtures.
- **Variable-Order Markov Models (VOMM)** — adaptive context-depth selection.
- **Context Tree Weighting (CTW)** — Krichevsky–Trofimov estimator, weighting over all
  bounded-depth tree sources.
- **PPM-A/C/D** — escape and exclusion mechanisms, count-based backoff.
- **Expert mixing & gating** — confidence gates, support-conditioned weighting, and a
  **Jensen–Shannon disagreement** signal to allocate trust between experts online.

## 4.3 Engineering & scientific method — *applied*

- **Online/causal algorithm design** under a hard 600 s / 256-context budget using NumPy
  only; cheap incremental count updates instead of per-step heavy computation.
- **Determinism & reproducibility** — stability testing (`stability_std ≈ 0`), train-derived
  validation to detect seed overfitting, data integrity via SHA-256.
- **Experiment discipline** — hypothesis-before-run write-ups, a 188-run ledger across two
  phases, JSONL run logs, and an **autonomous research loop** that proposed, ran, validated,
  and recorded experiments. Knowing when to *stop* (recognizing the entropy floor) is part
  of the skill.
- **LLM inference at scale (Activity B)** — running and vectorizing token-level scoring
  across a model family (distilgpt2 → Qwen2.5-7B) on GPU with `transformers`, normalizing
  token log-loss to bits/character to compare against published results.
- **Tooling** — `uv` environment management, reproducible CLI pipelines, figure generation.

## 4.4 Broader ML — *studied in the course* (context, not coded here)

Fully-observed probabilistic models; latent-variable models; EM and Gaussian mixtures;
variational inference and VAEs; the reparameterization trick; information-theoretic
generalization bounds (sub-Gaussian concentration). These appear in the lecture material and
shaped the information-theoretic mindset of the project, but were not part of the Activity A
predictor or Activity B pipeline.

## 4.5 The three transferable lessons

1. **Pick the metric that *is* the goal.** Scoring in bits/symbol made "build a better
   model" mean exactly "reduce KL divergence," which kept every experiment honest.
2. **Know the floor.** Recognizing `H(P)` as a hard limit turned vanishing gains from a
   frustration into the actual finding — the model class had converged.
3. **Make search auditable.** Hypothesis-first write-ups + deterministic validation +
   logged runs make a long experimental campaign trustworthy and reproducible.

---

Back to [**README**](../README.md) · [**1. Problem & theory**](01_problem_and_information_theory.md)
· [**2. Methodology**](02_methodology.md) · [**3. Results**](03_results.md)
