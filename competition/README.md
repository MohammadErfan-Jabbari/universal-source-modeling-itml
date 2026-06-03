# Live Competition Protocol

Instructor/maintainer protocol reference. Students should use `README.md` and `submissions/README.md` first.

This package contains the live competition release/evaluation tools for the Universal Source Modeling Challenge.

Score (canonical): The evaluator computes empirical average log-loss (base-2) in bits/symbol:
\widehat{H}_Q(x_1^N) = \frac{1}{N}\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1}).
See COMPETITION_RULES.md for the official definition and ranking validity rules.

## Live Workflow

Instructor:
1. Run `python -m competition.release_test --test-length 300000 --seed <SECRET_SEED>`
2. Share the released `test.npy` (and optionally `sha256.txt` for integrity verification)

Students:
1. Create `predictor.py` that defines:
   `build_predictor(alphabet_size: int, max_context_length: int) -> Predictor`
2. Run `python -m competition.run_live_eval --test-path path/to/test.npy --predictor-path path/to/predictor.py`
3. Submit the printed `FINAL_SCORE ...` line

## Evaluation Rules (Enforced)

- Only the first `N=200000` symbols are evaluated by default (`--num-tokens`, default 200000)
- `--smoke-test` overrides this to `5000` symbols
- Runtime limit enforced via `--time-limit-seconds` (default `600`)
- Context length enforced via `--max-context-length` (default `256`)
- Sequential prediction only, no lookahead (`q_i` may depend only on past symbols)

The evaluator score is the empirical average log-loss in base 2 (bits/symbol); it scores log-loss directly and does not perform arithmetic coding.

## Canonical Output Line

The evaluator prints exactly one final line for copy/paste into ranking tools:

`FINAL_SCORE bits_per_symbol=... elapsed_seconds=... timed_out=... evaluated_tokens=...`

## Baseline Benchmark Table (Instructor Utility)

You can generate a reference benchmark table (empirical bits/symbol on the chosen test set, speed, timeout status):

```bash
python -m competition.benchmark_baselines \
  --test-path data/public_practice/test.npy \
  --train-path data/generator/train.npy \
  --num-tokens 200000
```
