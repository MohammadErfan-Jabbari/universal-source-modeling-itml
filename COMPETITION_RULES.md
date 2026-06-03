# Universal Source Modeling Challenge — Competition Rules

## A) Objective (Information Theory Framing)

Let the evaluated realization be `x_1^N` over alphabet `\mathcal{X}`, and let your sequential predictor emit `q_i(\cdot \mid x_1^{i-1})`.

The score is empirical average log-loss (empirical cross-entropy) in bits/symbol (all logs base 2):

`\widehat{H}_Q(x_1^N) = (1/N) * sum_{i=1}^N -log2 q_i(x_i | x_1^{i-1})`

- Lower is better.
- This is the leaderboard score (directly scored log-loss, not an achieved compressed file size).
- Each term has an ideal codelength interpretation under model probabilities.
- Relative to the source distribution, excess cross-entropy is the KL / redundancy term in the identity `H(P,Q)=H(P)+D(P||Q)` (distributional statement; finite-sample scores are empirical).

## B) Data & Evaluation Protocol

- The hidden test set is released live on competition day.
- Evaluation uses only a fixed prefix `x_1^N` of the released test set.
- Default evaluated prefix length: `N = 200000` symbols.
- Alphabet size: `A = |\mathcal{X}| = 16`.
- Prediction is strictly sequential (online): no lookahead.

## C) Constraints

- `max_context_length = 256` (hard-enforced by the harness)
- `time_limit_seconds = 600` (10 minutes)
- Submissions must run in a Google Colab Pro+ style environment (resource-constrained)
- Fixed prefix length `N` is required for fairness and runtime predictability

## D) Ranking Rules

Timeout policy (Choice A):
- Runs with `timed_out=True` are **DISQUALIFIED**.

Among non-timeout valid runs, rank by:
1. `bits_per_symbol` (ascending; lower is better)
2. `elapsed_seconds` (ascending) as tiebreaker

Validity rule:
- `evaluated_tokens` must equal the required prefix length (default `200000`) unless the instructor explicitly allows otherwise.
- Mismatched `evaluated_tokens` are treated as invalid for official ranking.

## E) Allowed / Disallowed

Allowed:
- Any model architecture
- Any training at home
- Any preprocessing done before live evaluation

Disallowed during live evaluation:
- External API calls
- Test-set peeking or manual inspection beyond normal evaluation usage
- Lookahead (using future test symbols)
- Modifying the provided evaluation script / protocol

Submission requirement:
- Students must provide a Python predictor file implementing:
  `build_predictor(alphabet_size, max_context_length)`

## F) Provided Baselines

- Uniform baseline (`4.0` bps for alphabet size `16`)
- Official simple classical baseline: hard-backoff n-gram with `n=4`, `Laplace=1.0`
- Optional stronger classical reference: `ngram_threshold` with `n=5`, `min_count=8`

## G) Bonus (Optional)

- The compression sanity check (`zlib` / `lzma` / `bz2`) is optional.
- It is for demonstrating the prediction ↔ coding connection.
- LLMZip-style arithmetic-coding realizability is educational context only; the leaderboard does not run arithmetic coding.
- Optional arithmetic-coding validation on the **public practice set** may be submitted as a course bonus item, but it is not part of leaderboard scoring and is not run on live day.
- It is **not** part of the leaderboard ranking.

## Student Commands

Smoke test (5000 symbols):

```bash
python -m competition.run_live_eval \
  --test-path data/live_release/test.npy \
  --predictor-path submissions/template_predictor.py \
  --smoke-test
```

Full default run (200000-symbol prefix):

```bash
python -m competition.run_live_eval \
  --test-path data/live_release/test.npy \
  --predictor-path submissions/template_predictor.py
```

## Canonical Output Line

Students must submit the final line produced by the evaluator:

`FINAL_SCORE bits_per_symbol=... elapsed_seconds=... timed_out=... evaluated_tokens=...`
