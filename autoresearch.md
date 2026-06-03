# Autoresearch: ITML Activity A Sequential Predictor — Phase 2

## Objective

Improve the Activity A predictor for the Universal Source Modeling Challenge.

Phase 1 exhaustively explored fixed-order n-grams, mixtures, and online adaptation (121 runs).
Phase 1 best: **2.969228 bits/symbol** (Run 117, n=5 interpolated per-order mixture + n=3 online auxiliary).

Phase 2 mandate: **Explore structurally different model families**: Variable-Order Markov Models, Context Tree Weighting, and PPM-style escape mechanisms.
Do not run more n-gram hyperparameter sweeps — that space is exhausted.

The official objective is empirical average log-loss on a sequential source over alphabet size 16:

```text
Hhat_Q(x_1^N) = (1/N) * sum_i -log2 q_i(x_i | x_1^{i-1})
```

The primary optimization target is lower `bits_per_symbol` on the official public-practice evaluator, while preserving strict sequential validity, runtime safety, and a presentation-quality explanation.

This is not a generic code-golf task. The final result must be defensible in the Information Theory for ML course: prediction as coding, cross-entropy as ideal codelength, and redundancy as model mismatch.

## Phase 2 Baseline

```text
Phase 1 best predictor: submissions/best_predictor_phase1.py
FINAL_SCORE bits_per_symbol=2.9692281375 elapsed_seconds=9.722408 timed_out=False evaluated_tokens=200000
```

## Metrics

- **Primary**: `bits_per_symbol` (bits/symbol, lower is better)
- **Secondary**:
  - `elapsed_seconds` (must stay well below 600s)
  - `evaluated_tokens` (must equal 200000 for full runs)
  - `timed_out` (must be 0)
  - smoke-test validity with probability validation
  - explanation quality and theoretical defensibility

## Tiered Validation (New in Phase 2)

Every full evaluation now includes two extra checks to reduce hallucination and overfitting:

### 1. Stability Check (always runs, ~9s)
- Evaluates the predictor **3 times** on the **same 5k block** from `train.npy` (after 50k warmup).
- Purpose: **Detect non-determinism.** If the 3 scores differ, the model has a bug (uninitialized state, randomness, or side effects).
- `stability_std` should be `0.0` for a correct model.
- Example output: `VALIDATION stability_std=0.0 ...`

### 2. Train-Derived Validation (always runs, ~11s)
- Evaluates the predictor on the **last 50k symbols** of `train.npy` (after 250k warmup).
- Purpose: **Detect overfitting to the public-practice seed.** If a model does great on `test.npy` but badly on `train.npy`, it may be overfitting seed 123.
- Compare `train_val_score` to `bits_per_symbol`. A large gap (>0.05) is a red flag.
- Example output: `VALIDATION ... train_val_score=2.8444 ...`

### Why both?

| Check | Catches | Example failure |
|-------|---------|-----------------|
| Stability | Non-deterministic code | Random init, mutable global state |
| Train-val | Seed overfitting | Model memorizes test.npy patterns |

Both run automatically after every full eval. Total overhead: ~20s per run.

## Rules for Phase 2

1. Each run must try a genuinely different architecture (VOMM, CTW, or PPM).
2. Do NOT run more n-gram Laplace sweeps or n-gram hyperparameter tuning.
3. Hybrid ideas (new model + Phase 1 best) are allowed, but the new component must be substantive.
4. Beat 2.969228 to be kept, unless the run teaches a critical architectural lesson.
5. Write an `explanations/run_###.md` for every run with information-theoretic motivation.
6. **Check stability_std in the validation output.** If it is not 0.0, the model is buggy. Do not keep it.
7. **Check train_val_score.** If it is much worse than the public-practice score, the model may be overfitting.

## Official template predictor (archived baseline)

```text
submissions/template_predictor.py
```

Verified full public-practice baseline:
```text
FINAL_SCORE bits_per_symbol=2.9986251144 elapsed_seconds=1.858381 timed_out=False evaluated_tokens=200000
```

## Evaluator

Use the official evaluator:

```bash
PREDICTOR_PATH=submissions/your_predictor.py ./autoresearch.sh
```

## Checks

Run `./autoresearch.checks.sh` before committing.

## Phase 1 Archive

Phase 1 artifacts (121 runs, 56 submissions, 56 explanations) are preserved in:
```text
archive/phase1_ngram_exploration/
```

## Operational notes

The autoresearch loop is configured for up to 500 iterations.
The loop should start from the Phase 2 baseline and explore new architectures.
Do not let the agent revert to n-gram micro-tuning.
If a run is a thin variant of an n-gram, reject it and redirect to VOMM/CTW/PPM.
