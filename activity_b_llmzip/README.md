# Activity B — LLMZip-Style English Compression

This folder sets up Part B of the ITML project on Centcom.

Professor's Activity B asks us to explore natural-language prediction/compression with pretrained LLMs, inspired by LLMZip:

- work with `text8` if possible,
- use the largest local LLM we can run,
- compute empirical entropy / cross-entropy estimates,
- compare against classical compressors,
- optionally implement a lossless arithmetic-coding demonstration.

## Folder layout

```text
activity_b_llmzip/
  AGENTS.md
  README.md
  requirements.txt
  data/                  # ignored: text8/text samples
  results/               # ignored: JSON metrics, rank files
  reports/
    experiment_log.md    # durable human-readable run log
  scripts/
    download_text8.py
    compress_baselines.py
    llm_bpc_eval.py
```

## One-time environment setup

From repo root:

```bash
cd /path/to/universal-source-modeling-itml
uv pip install -r activity_b_llmzip/requirements.txt
```

If PyTorch/CUDA is already installed globally, `uv pip install` may reuse or install a compatible build. If it tries to download a very large CUDA stack, stop and install PyTorch using the cluster's recommended command instead.

Recommended cache location:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
```

## Step 1: get text8

```bash
uv run python activity_b_llmzip/scripts/download_text8.py \
  --out activity_b_llmzip/data/text8.txt
```

## Step 2: classical compression baselines

Start small:

```bash
uv run python activity_b_llmzip/scripts/compress_baselines.py \
  --text-path activity_b_llmzip/data/text8.txt \
  --max-chars 100000 \
  --out activity_b_llmzip/results/baselines_100k.json
```

Then 1MB:

```bash
uv run python activity_b_llmzip/scripts/compress_baselines.py \
  --text-path activity_b_llmzip/data/text8.txt \
  --max-chars 1000000 \
  --out activity_b_llmzip/results/baselines_1mb.json
```

## Step 3: LLM ideal codelength and rank-zlib

Smoke test on one L40S GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python activity_b_llmzip/scripts/llm_bpc_eval.py \
  --text-path activity_b_llmzip/data/text8.txt \
  --model gpt2 \
  --max-chars 20000 \
  --stride 512 \
  --save-ranks \
  --out activity_b_llmzip/results/gpt2_20k.json
```

Larger run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python activity_b_llmzip/scripts/llm_bpc_eval.py \
  --text-path activity_b_llmzip/data/text8.txt \
  --model gpt2 \
  --max-chars 1000000 \
  --stride 512 \
  --save-ranks \
  --out activity_b_llmzip/results/gpt2_1mb.json
```

For larger models, replace `--model gpt2` with the chosen Hugging Face model id and consider:

```bash
--dtype bfloat16 --device cuda
```

## Metrics to report

- `bits_per_token`: LLM token-level cross-entropy.
- `bits_per_character`: total LLM token bits divided by original character count.
- `rank_zero_fraction`: fraction of positions where the true next token was top-ranked.
- `rank_zlib_bits_per_character`: LLMZip-style rank stream compressed with zlib, normalized per original character.
- classical compressor bpc: zlib/bz2/lzma on raw text.

## Paper reference numbers from professor slides

For 1MB text8, the slides report roughly:

| Method | bits/character |
|---|---:|
| Entropy upper bound | 0.7093 |
| LLaMA + arithmetic coding | 0.7101 |
| LLaMA + token-by-token | 0.845 |
| LLaMA + zlib ranks | 1.0896 |
| ZPAQ | ~1.4 |

Our first goal is not to match LLaMA-7B immediately; it is to reproduce the measurement pipeline and explain the gap.
