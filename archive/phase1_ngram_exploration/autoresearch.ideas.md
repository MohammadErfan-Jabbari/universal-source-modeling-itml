# Autoresearch Ideas — ITML USM Challenge

This file seeds the autoresearch loop. Treat it as a menu, not a mandate. The agent should choose one coherent idea per run, write `explanations/run_###.md`, then benchmark through `./autoresearch.sh`.

Primary target: lower `bits_per_symbol` than the official `n=4`, `Laplace=1.0` hard-backoff baseline (`2.9986251144` on public practice) while staying valid, fast, and explainable.

## Highest-priority first experiments

### 1. Hyperparameter sweep around the official n-gram baseline

**Idea:** The provided template uses `NGRAM_N = 4`, `LAPLACE = 1.0`, and online adaptation. Try nearby orders and smoothing values before inventing a new model.

Candidate settings:

- `n`: 3, 4, 5, 6, 7, 8
- `laplace`: 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0
- `adapt_online`: True vs False

Why it may work:

- Laplace=1.0 is conservative for alphabet size 16 and may over-smooth frequent contexts.
- Higher context order may capture more source memory if counts are sufficient.
- Lower smoothing may help frequent contexts but can hurt rare contexts.

Risk:

- Higher `n` makes sparse contexts and memory/time worse.
- Public practice may favor a setting that is not robust live.

Implementation tip:

- Copy `submissions/template_predictor.py` to a new file, e.g. `submissions/ngram_tuned.py`.
- Do not edit the official template directly unless there is a clear reason.

### 2. Thresholded n-gram backoff

**Idea:** Use a high-order context only if it has enough observations; otherwise back off to shorter contexts. The rules mention an optional stronger reference: `ngram_threshold` with `n=5`, `min_count=8`.

Candidate settings:

- `n`: 5, 6, 7
- `min_count`: 2, 4, 8, 16, 32
- smoothing: 0.05, 0.1, 0.2, 0.5

Why it may work:

- High-order contexts can be excellent when well-supported.
- Rare high-order contexts overfit; thresholding reduces variance.
- This is directly aligned with the bias-variance tradeoff discussed in the slides.

Risk:

- If threshold too high, model collapses to low-order baseline.
- If threshold too low, it overfits sparse contexts.

### 3. Mixture of context orders instead of hard backoff

**Idea:** Instead of choosing one backoff order, combine probability estimates from several suffix lengths.

Example:

```text
q = w0 q_order0 + w1 q_order1 + ... + wk q_orderk
```

Weights can be fixed, count-dependent, or recursively updated.

Candidate variants:

- fixed weights favoring higher orders, e.g. `[0.05, 0.10, 0.15, 0.25, 0.45]`
- count-dependent weights, where high-order weight increases with context count
- exponential confidence: `weight_order_k ∝ count_k / (count_k + c)`

Why it may work:

- Hard backoff discards shorter-context evidence once a longer context exists.
- Mixtures are often more robust: rare contexts still borrow low-order statistics.
- Theoretical story: universal coding/model averaging reduces redundancy from wrong model-order choice.

Risk:

- Slightly more expensive per prediction.
- Need careful normalization and no invalid probabilities.

### 4. Online adaptation weighting

**Idea:** The public and live test distributions likely come from the same generator family but a different seed. Online adaptation may help, but blindly adding test counts with the same weight as train counts may be suboptimal.

Candidate variants:

- train counts + online counts with multiplier `alpha_online < 1` or `alpha_online > 1`
- delayed adaptation: adapt after warmup only
- recency window adaptation: maintain separate recent counts for last `W` symbols
- mixture of static train model and online model

Why it may work:

- The test realization may have local state/path-specific regularities.
- Online adaptation is allowed and still sequential.

Risk:

- Too much adaptation can chase noise.
- Maintaining recent counts can slow down runtime or complicate code.

### 5. Validation split from `train.npy` before trusting public practice

**Idea:** Build a script that splits `data/generator/train.npy` into fit/eval parts and evaluates candidate predictors on held-out train suffixes.

Why it may work:

- Public practice is one seed; live day is another seed.
- Train-derived validation helps avoid overfitting public practice.

Candidate split:

- fit first 200k, validate next 100k
- rolling splits: fit first 100k/150k/200k, validate next 50k

Risk:

- The official evaluator currently loads predictors that themselves load full `train.npy`, so validation infrastructure may require auxiliary scripts.
- Keep official score separate from auxiliary validation.

## Strong classical model directions

### 6. PPM-style predictor

**Idea:** Implement Prediction by Partial Matching style context modeling over alphabet size 16.

Possible components:

- max order 5-8
- escape probability based on number of unique seen symbols and total counts
- recursive backoff distribution
- optional exclusion: avoid double-counting symbols already assigned at higher order

Why it may work:

- PPM is a strong universal compression family for symbolic sequences.
- It naturally matches the course theme: sequential probability assignment → coding length.

Risk:

- More implementation complexity.
- Need careful probability mass accounting to ensure distribution sums to 1.

### 7. Context-tree weighting inspired model

**Idea:** Use a context tree where each node mixes its local KT/Lidstone estimate with children/suffix estimates.

Simplified version:

- Build suffix-count trie up to depth `D`.
- Each context probability is a mixture of local smoothed distribution and shorter-context distribution.
- Use count-dependent mixture weights.

Why it may work:

- CTW is theoretically tied to universal coding and model-order uncertainty.
- Good presentation story if implemented cleanly.

Risk:

- Full CTW for alphabet 16 is nontrivial.
- Approximate CTW is fine, but must be honest in presentation.

### 8. Hidden Markov intuition without training a full HMM

The data appears synthetic and may come from a hidden-state generator. Full HMM inference/training could help, but is more complex.

Possible lightweight approximations:

- cluster contexts by next-symbol distribution
- maintain state-like recent feature signatures
- mixture over short-term and long-term n-gram models

Why it may work:

- If source is HMM-like, finite context statistics approximate posterior over hidden state.

Risk:

- Full Baum-Welch/EM may be slow and hard to integrate into strict sequential predictor.
- Do not jump here before exhausting n-gram/PPM/mixture baselines.

## Runtime and implementation ideas

### 9. Dense arrays for low-order counts, sparse dictionaries for high-order counts

Current baseline uses dictionaries for all contexts. That is simple and fast enough, but if higher orders or mixtures get expensive, consider hybrid storage.

Ideas:

- encode contexts as integer rolling hashes/base-16 numbers for depth <= `D`
- dense `np.ndarray` for counts at low orders
- sparse dict for rare/high-order contexts
- cache computed log-prob vectors for contexts that repeat often, invalidating carefully after online update

Risk:

- Premature optimization can introduce bugs.
- Any cache must remain valid under online adaptation.

### 10. Separate smoke vs full benchmark behavior

During early development, use smoke tests to catch invalid probabilities and syntax errors. But do not judge improvements on 5000 tokens; it is too noisy.

Rule of thumb:

- use smoke for validity only
- use full 200k public practice for primary metric
- use train-derived validation for robustness

## Presentation-worthy theory hooks

Use these to explain successful experiments:

- **Cross-entropy:** score is average ideal codelength under model probabilities.
- **Redundancy:** `H(P,Q)=H(P)+D(P||Q)`; improvements reduce model mismatch.
- **Bias-variance tradeoff:** longer contexts capture more structure but overfit sparse histories.
- **Universal coding:** mixtures/backoff/PPM reduce redundancy when true source order is unknown.
- **Online adaptation:** allowed because it uses only past symbols; it estimates realization-specific structure.
- **Runtime as constraint:** a lower-entropy model is useless if it times out.

## Ideas to avoid initially

- Editing `competition/run_live_eval.py` or `competition/evaluation/harness.py` for score changes.
- Using the future/public-practice test sequence inside training code.
- Huge neural models before classical models are exhausted.
- External API calls at evaluation time.
- A predictor that is too complex to explain in the presentation.
- Optimizing only for public practice without train-derived validation.

## Suggested first 10 runs

1. Baseline official template (already recorded as run 001).
2. `n=5`, Laplace=1.0, adapt online.
3. `n=5`, Laplace=0.1, adapt online.
4. `n=6`, Laplace=0.1, adapt online.
5. `n=5`, Laplace=0.05, adapt online.
6. `n=5`, Laplace=0.2, adapt online.
7. `n=5`, `min_count=8` threshold backoff, Laplace=0.1.
8. `n=6`, `min_count=8` threshold backoff, Laplace=0.1.
9. Mixture of orders 0..5 with count-dependent weights.
10. Static-vs-online mixture variant.

After run 10, pause and update this file based on actual results.

## Appended 2025-04-30 (after 91 runs, best at 2.9704 bps)

### PPM-style exclusion
**Idea:** When interpolating across orders, exclude symbols seen at higher-order contexts from lower-order borrowing. This prevents probability dilution: a symbol with strong higher-order evidence shouldn't get re-distributed lower-order mass. Implementation would modify `_compute_component_probs` to zero out already-seen symbols in lower-order local distributions before mixing. Complex but theoretically well-motivated.

### Adaptive main model smoothing per order
**Idea:** Use different Laplace pseudocounts for each context order in the main model. Lower orders (dense) could use sharper smoothing like the auxiliary model, while higher orders keep alpha=0.3 protection.

### Validation split from train.npy
**Idea:** Build a held-out validation from the training sequence to confirm robustness. The public practice sequence may favor settings that don't generalize.

### Context-tree weighting (CTW) simplified
**Idea:** Replace interpolation with CTW-style recursive mixing along context suffix tree. More principled than interpolation but similar in spirit.

### Re-run best for noise floor
**Idea:** Re-run the current best predictor to measure run-to-run variance. The improvement over baseline is ~0.94% but individual gains are now <0.0001 bits/symbol.
