# Experiment Ledger — ITML USM Challenge

Record durable, human-readable results here. Autoresearch also writes machine logs to `autoresearch.jsonl`.

| Run | Predictor | Main idea | bits/symbol | seconds | tokens | status | Notes |
|---:|---|---|---:|---:|---:|---|---|
| 001 | `submissions/template_predictor.py` | Official n=4 Laplace hard-backoff n-gram baseline | 2.9986251144 | 1.858381 | 200000 | baseline | Verified before autoresearch setup. |
| 009 | `submissions/ngram_interpolated_n4_c16.py` | Count-dependent interpolation over n-gram orders 0..3 (`C=16`) | 2.9817296585 | 3.591695 | 200000 | kept | First improvement; see `explanations/run_009.md`. Confidence low, needs confirmation/validation. |
| 010 | `submissions/ngram_interpolated_n4_c32.py` | More conservative count-dependent interpolation (`C=32`) | 2.9794649970 | 3.608527 | 200000 | kept | Former best; see `explanations/run_010.md`. Confidence low, needs confirmation/validation. |
| 012 | `submissions/ngram_interpolated_n4_c48.py` | Intermediate count-dependent interpolation (`C=48`) | 2.9792935749 | 3.197505 | 200000 | kept | Former best; see `explanations/run_012.md`. Tiny gain, needs confirmation/validation. |
| 013 | `submissions/ngram_interpolated_n4_c40.py` | Local interpolation tuning (`C=40`) | 2.9792409029 | 3.047526 | 200000 | kept | Current best; see `explanations/run_013.md`. Very tiny gain, needs confirmation/validation. |

## Rules

- Add meaningful kept runs and important dead ends.
- Include links to `explanations/run_###.md` when possible.
- Do not treat public-practice score as the only truth; note robustness checks and train-derived validation when available.
