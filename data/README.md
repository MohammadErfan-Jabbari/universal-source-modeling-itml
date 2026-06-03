# Data

Synthetic data for Activity A (the Universal Source Modeling Challenge). The source is a
synthetic generator over an alphabet of size 16; the true source distribution `P` is not
disclosed — modeling it under that uncertainty is the point of the task.

## Contents

| Path | Shape / dtype | Description |
|---|---|---|
| `generator/train.npy` | `(300000,)` `int64`, values 0–15 | Training sequence, available before evaluation. Used for building counts and for the train-derived validation split. |
| `public_practice/test.npy` | `(300000,)` `int64`, values 0–15 | Public practice sequence. The official score uses the first `N = 200,000` symbols. |
| `public_practice/metadata.json` | — | `{ test_length: 300000, alphabet_size: 16, seed: 123, run_id: "practice_a16_t300000_seed123" }` |
| `public_practice/sha256.txt` | — | Integrity checksum for `test.npy`. |

## Integrity check

```bash
(cd data/public_practice && sha256sum -c sha256.txt)   # expect: test.npy: OK
```

## Provenance and scope

- These arrays are the **competition-provided** practice/training data (seed 123) and are
  included here for reproducibility.
- The **secret live test set** (`data/live_release/`) is *never* part of this repository and
  is git-ignored. Reported scores are on the public practice file only.
- Model selection should prefer the **train-derived validation split** over the public
  practice file to avoid overfitting to seed 123; see [`../docs/02_methodology.md`](../docs/02_methodology.md).
