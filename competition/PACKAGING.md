# Instructor Packaging Guide

Instructor-only operational document (private repo / maintainer use). Not intended as the student entrypoint.

This document describes a clean workflow for preparing and distributing the student starter bundle and a public practice set.

## Workflow

### Step 1: Prepare / refresh training data (`data/generator/train.npy`)

If needed, generate or refresh the training set using the instructor-side generator workflow.

Example:

```bash
python -m src.data.synthetic_source \
  --train-length 300000 \
  --test-length 300000 \
  --seed 0 \
  --output-dir data/generator
```

Students only need the training split for the starter baseline template.

### Step 2: Generate a public practice test set

Use the public practice release script (same locked HMM family as live day, but public seed):

```bash
python -m competition.release_practice \
  --test-length 300000 \
  --seed 4242 \
  --output-dir data/public_practice
```

Optional integrity check:

```bash
python -m competition.verify_hashes --dir data/public_practice
```

### Step 3: Share the student starter bundle

Recommended minimal bundle:

- `submissions/template_predictor.py`
- `submissions/README.md`
- `notebooks/colab_starter.ipynb`
- `notebooks/README.md`
- `competition/run_live_eval.py`
- `COMPETITION_RULES.md`
- `data/generator/train.npy`
- `data/public_practice/test.npy`
- `data/public_practice/metadata.json`
- `data/public_practice/sha256.txt`

### Build a student bundle (zip-ready folder)

You can assemble the recommended bundle into a clean output directory with:

```bash
python -m competition.make_student_bundle --output-dir /tmp/student_bundle
```

Then zip `/tmp/student_bundle` and distribute it to students.

## Example Colab commands for students

Smoke test (5000 tokens):

```bash
python -m competition.run_live_eval \
  --test-path data/public_practice/test.npy \
  --predictor-path submissions/template_predictor.py \
  --smoke-test
```

Full practice run (default 200000-token prefix):

```bash
python -m competition.run_live_eval \
  --test-path data/public_practice/test.npy \
  --predictor-path submissions/template_predictor.py
```

## Important note

- The **practice seed is public** and may be shared freely.
- The **live-day seed remains secret** and is only released at competition time via `competition.release_test`.
