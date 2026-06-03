# Activity A — Submissions

This directory contains the **six curated milestone predictors** that trace the full
development arc from the instructor baseline to the final best submission (table below),
plus the **four supporting components** the best predictor imports (see *Supporting
components* below). The full long-tail of Phase 2 variant runs lives in
`../archive/phase2_runs/submissions/`; the 121 Phase 1 runs in
`../archive/phase1_ngram_exploration/`; and every run is recorded in `../EXPERIMENTS.md`.

---

## Predictor Interface

Every predictor file exports one function:

```python
def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    ...
```

`Predictor` is the abstract base class defined in `competition/predictors/base.py`.
Implementations must override three methods:

| Method | Signature | Contract |
|---|---|---|
| `initialize()` | `() -> None` | Reset all internal state before a new sequence begins. |
| `predict_next(context)` | `(Sequence[int]) -> NDArray[float64]` | Return **log2 probabilities** for the next symbol. |
| `update(observed_symbol)` | `(int) -> None` | Update internal state after the true symbol is revealed. |

### Output contract for `predict_next`

- **Shape**: `(alphabet_size,)` — one value per symbol in `{0, …, alphabet_size−1}`.
- **Values**: log2 probabilities — finite, non-positive (i.e. in `(−∞, 0]`).
- **Probability sum**: `sum(2**log_probs) == 1.0` in probability space (up to float64 tolerance).
- **Lookahead forbidden**: `context` contains only `x_1^{i-1}`. Do not store or inspect it beyond the scope of the call.

The evaluation harness computes empirical cross-entropy as:

```
Hhat_Q(x_1^N) = (1/N) * sum_{i=1}^N  -log2_prob[i][x_i]
```

Lower is better. By `H(P,Q) = H(P) + D(P‖Q)`, every reduction in `Hhat` relative to
a baseline represents a direct reduction in the KL divergence / redundancy D(P‖Q).

---

## Milestone Progression

| # | File | Algorithm idea | bits/symbol | Key lesson |
|---|---|---|---|---|
| 1 | `template_predictor.py` | n=4 Laplace (α=1.0) hard-backoff n-gram with online adaptation | **≈2.9986** | Instructor baseline; establishes the coding duality and the online-adaptation interface. |
| 2 | `best_predictor_phase1.py` | n=5 per-order interpolated mixture (fixed-λ + adaptive-λ blend) + n=3 sliding-window recent auxiliary (window=262 144, warmup=15 000) | **2.969228** | Per-order count-dependent interpolation reduces order-selection bias; a sliding-window hedge reduces redundancy from non-stationarity. Phase 1 best of 121 runs. |
| 3 | `ppm_phase1_hybrid.py` | PPM-C order-5 with symbol exclusions, confidence-gated by `count_conf × entropy_conf` (max 8% weight) into milestone 2 | **2.969019** | Pure PPM-C overfits training recurrences; a count-and-entropy gate extracts the complementary local signal without letting model mismatch dominate. |
| 4 | `ctw_countmix_phase1_hybrid.py` | Count-dependent CTW suffix tree (λ=N/(N+C), KT smoothing α=0.5 per symbol), gated into milestone 3 | **2.968990** | Count-adaptive blending beats the canonical fixed-50/50 CTW; high-evidence nodes carry more weight, recovering the Bayesian interpolation optimality of CTW. |
| 5 | `prob_depth_vomm_countmix_hybrid.py` | Probabilistic-depth VOMM: soft mixture over context depths 0–6, each weighted by count support × entropy sharpness × depth factor, gated into milestone 4 | **2.968982** | Soft depth selection dominates hard min-count VOMM; the depth weights act as a posterior over Markov order and reduce the order-selection redundancy penalty. |
| 6 | `cautious_support_conditioned_disagreement_boost_ppma_stack.py` | PPM-A order-7 count-escape component + selective CTW/VOMM base stack; PPM-A weight boosted by `min(1 + 0.88 × JS × sqrt(count_conf), 1.80)` where JS is the Jensen-Shannon divergence between the two experts | **≈2.968933** | JS divergence proxies the residual conditional information in the PPM-A expert given the base; boosting its weight when JS > 0 on well-supported contexts closes the remaining redundancy gap. Final best. |

Total improvement over the template baseline: **≈0.02969 bits/symbol**.

### Supporting components

The final predictor (milestone 6) is the top of a small tower of intermediate experts it
imports. These files are kept here so the best predictor runs out of the box; they are
components, not headline milestones:

- `order7_count_escape_ppma_stack.py` — the order-7 count-escape PPM-A expert.
- `selective_ctw_vomm_ppma_hybrid.py` — the selective CTW/VOMM base stack.
- `selective_vomm_ppma_hybrid.py` — selective probabilistic-depth VOMM component.
- `ppma_selective_update_hybrid.py` — selective-update PPM-A hybrid base.

---

## Running a Predictor

```bash
# Smoke test (5000 symbols, fast sanity check)
uv run python -m competition.run_live_eval \
  --test-path data/public_practice/test.npy \
  --predictor-path submissions/template_predictor.py --smoke-test

# Full official run (200 000-symbol prefix)
uv run python -m competition.run_live_eval \
  --test-path data/public_practice/test.npy \
  --predictor-path submissions/template_predictor.py

# Stability + train-val validation
uv run python scripts/validate_predictor.py \
  --predictor-path submissions/your_predictor.py
```

The canonical result line is:
```
FINAL_SCORE bits_per_symbol=... elapsed_seconds=... timed_out=... evaluated_tokens=...
```

See [`../docs/02_methodology.md`](../docs/02_methodology.md) for the full evaluation and
experiment-hygiene protocol, and [`../COMPETITION_RULES.md`](../COMPETITION_RULES.md) for
the official rules.

---

## Bonus: Model-Matched Arithmetic Coding Validation (Practice Set)

This is optional and does **not** affect leaderboard ranking. It validates that the
sequential probability assignments `q_i(· | x_1^{i-1})` define a valid arithmetic
code by running a sequential arithmetic coder matched to those probabilities.

Recommended setup:
- Dataset: `data/public_practice/test.npy` only.
- Short fixed prefix: `N_AC = 30000` symbols.

Report (base-2, bits/symbol):
- `Hhat_Q` — empirical log-loss from the evaluator.
- `bps_AC = compressed_bits / N_AC` — from the arithmetic coder.
- `Delta_AC = bps_AC - Hhat_Q` — gap to the ideal codelength.
- `DECODE_OK = True/False` — lossless decode is required.

The arithmetic coder integration with the sequential PMFs must be your own work.
Do not run this on the live-day secret test.
