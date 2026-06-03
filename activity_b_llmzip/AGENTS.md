# Activity B — LLMZip-Style Exploration

This folder is for **Activity B / Part B** of the ITML Universal Source Modeling Challenge.

Activity A remains the constrained synthetic-source leaderboard task in the repo root. Activity B is separate: natural-language prediction/compression inspired by the LLMZip paper and evaluated through presentation quality.

## Goal

Use a pretrained causal language model as a sequential predictor for English text, then measure:

- empirical cross-entropy / ideal codelength,
- bits per character on `text8` or a clearly stated text corpus,
- rank-sequence compressibility (`LLM + zlib` style),
- optional arithmetic-coding realizability on a small sample.

The main conceptual bridge is the same as Activity A:

```text
better next-token probabilities -> shorter ideal code lengths -> lower empirical cross-entropy
```

## Safety and reproducibility rules

- Keep raw datasets, model caches, and generated outputs out of git.
- Prefer `text8` first, because the professor slides and LLMZip paper use it.
- Start with small `--max-chars` smoke runs before large GPU jobs.
- Record every meaningful run in `reports/experiment_log.md`.
- Always report the model, tokenizer, number of characters, number of scored tokens, context/stride, and exact normalization.
- Normalize LLM token log-loss to **bits per original character** when comparing to the paper.
- External APIs are not needed; use local Hugging Face models on Centcom GPUs.

## Suggested model ladder

1. `gpt2` or `distilgpt2` for smoke tests.
2. A larger local/Hugging Face causal LM that fits on one L40S.
3. If time permits, compare multiple model sizes and show the entropy-estimate trend.

## Expected presentation story

1. Activity A used a hand-built predictor for a synthetic source.
2. Activity B uses a pretrained LLM for English.
3. The objective is still cross-entropy: `-log2 q(x_i | x_<i)`.
4. Divide total token codelength by original characters to get bits/character.
5. Ranks and arithmetic coding are implementation paths from probabilities to actual compression.
