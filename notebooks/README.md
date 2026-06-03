# Notebooks

Score (canonical): The evaluator computes empirical average log-loss (base-2) in bits/symbol:
\widehat{H}_Q(x_1^N) = \frac{1}{N}\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1}).
See COMPETITION_RULES.md for the official definition and ranking validity rules.

## Colab Starter

Use `notebooks/colab_starter.ipynb` as the baseline-focused starter notebook for practice runs.

### How to use (Colab)

1. Open the notebook in Google Colab (after cloning/mounting this repository).
2. Set the working directory to the repository root.
3. Run cells top-to-bottom:
   - environment/version checks
   - data availability check (`data/generator/train.npy`, `data/generator/test.npy`)
   - smoke test (`--smoke-test`, 5000 symbols)
   - full practice run (default fixed prefix `N=200000`)

### Adapting your predictor

- Edit `submissions/template_predictor.py` or copy it to your own file.
- You must keep the required function:
  `build_predictor(alphabet_size: int, max_context_length: int) -> Predictor`

### Competition day

When the instructor releases `data/live_release/test.npy`, run:

```bash
python -m competition.run_live_eval \
  --test-path data/live_release/test.npy \
  --predictor-path your_predictor.py
```

Submit the printed `FINAL_SCORE ...` line exactly as produced.

Reminder: the score is direct log-loss (`bits/symbol`) from sequential prediction, not an actual compressed file size.
