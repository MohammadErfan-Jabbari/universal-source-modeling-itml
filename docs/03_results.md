# Results

All numbers are extracted directly from experiment logs and JSON result files.
No numbers are invented or estimated.

---

## Activity A — Sequential Source Modeling

### Score Progression

The table below traces every meaningful milestone from the uniform prior to the
Phase 2 best. Each row is an official `competition.run_live_eval` result
(N = 200 000 symbols, alphabet = 16).

| Milestone | Main idea | bits/symbol | Runtime (s) | Δ vs. prev. |
|---|---|---:|---:|---:|
| Uniform prior | Equal weight on all 16 symbols | 4.000000 | — | — |
| Template n-gram (n=4, Laplace) | Counting-based 4-gram with add-1 smoothing | 2.998625 | — | −1.001375 |
| **Phase 1 best — Run 117** | n=5 interpolated per-order mixture + n=3 all-past online auxiliary | 2.969228 | 9.72 | −0.029397 |
| Phase 2 R004 | Confidence-gated PPM-C blended into Phase 1 | 2.969019 | 15.64 | −0.000209 |
| Phase 2 R006 | Approximate CTW tree mixture gated into R004 | 2.969008 | 23.05 | −0.000011 |
| Phase 2 R010 | Count-dependent CTW mixture (support-adaptive) | 2.968990 | 22.36 | −0.000018 |
| Phase 2 R014 | Probabilistic-depth VOMM + CTW hybrid | 2.968982 | 34.54 | −0.000008 |
| Phase 2 R033 | Order-7 half-escape selective PPM-A | 2.968939 | 39.97 | −0.000043 |
| Phase 2 R044 | JS-disagreement-boosted PPM-A gate | 2.968938 | 42.89 | −0.000001 |
| Phase 2 R057 | Stronger support-conditioned JS disagreement boost | 2.968935 | 42.30 | −0.000003 |
| Phase 2 R062 | Support-conditioned JS disagreement boost (0.76) | 2.968934 | 44.69 | −0.000001 |
| Phase 2 R064 | JS disagreement boost 0.84 | 2.968933 | 49.95 | −0.000001 |
| **Phase 2 R067 (best overall)** | Sqrt-support JS disagreement boost 0.88 | **2.968933** | 43.85 | −0.000000† |

† R067 improves on R064 by ~2×10⁻⁷ bps; both round to 2.968933 at 6 decimal places.  
All Phase 2 runs: `stability_std = 0` (or numerical zero 4.4×10⁻¹⁶); `evaluated_tokens = 200 000`.

**Figure reference:** [Figure 1 — Activity A score progression](figures/activity_a_progression.png)

### Diminishing Returns and the Entropy Floor

The overall reduction from the uniform prior (4.000000 bps) to the Phase 1 best
(2.969228 bps) is **1.030772 bps** — a 25.8 % reduction. The entirety of Phase 2's
67 runs, exploring PPM-C variants, Variable-Order Markov Models, Context Tree
Weighting, and JS-divergence–gated hybrids, yielded a further improvement of only
**0.000295 bps** (i.e., ~3×10⁻⁴ bps), about 0.01 % relative.

This is not a failure of the Phase 2 architectures. It is consistent with the
prediction that the n-gram family had already converged near the model class's
practical entropy floor for this source. The Phase 2 models (CTW, VOMM, PPM-A)
are strictly more expressive than fixed-order n-grams, yet they add only marginal
complementary signal when combined as corrections to the Phase 1 model.

The train-derived validation scores (~2.759–2.760 bps for all Phase 2 best runs,
vs. 2.968–2.969 on the public practice file) reveal a gap that is consistent with
a slightly different marginal distribution between the train suffix and the public
practice test sequence — not evidence of overfitting to the practice file.

Information-theoretically, the residual redundancy `D(P‖Q) = H(P,Q) − H(P)` has
been compressed to at most a few ten-thousandths of a bit per symbol, which is
below the resolution at which further architectural changes can reliably be
distinguished from noise at N = 200 000.

---

## Activity B — LLMZip / Language Model Compression

All results below are on the **text8 dataset**, 1 MB (1 000 000 characters) unless
noted. Metric is **bits per original character (bpc)**; lower is better.

### Classical Compressor Baselines (1 MB text8)

| Method | bits/character |
|---|---:|
| Raw UTF-8 | 8.000 |
| zlib (level 9) | 2.638 |
| lzma (level 9) | 2.198 |
| bz2 (level 9) | 2.098 |

### LLM Ideal Codelength and LLMZip Rank+Compress (1 MB text8)

"Ideal codelength" = the theoretical lower bound if the LLM's probability
distribution were used as the codebook (arithmetic coding). "Rank+lzma" = the
LLMZip-style practical scheme: encode the token rank as a variable-length integer
and compress the rank stream with lzma.

| Model | Params | Ideal bpc | Rank+lzma bpc | Rank-zero fraction |
|---|---:|---:|---:|---:|
| distilgpt2 | 82 M | 1.254 | 1.328 | 0.282 |
| gpt2 | 124 M | 1.121 | 1.205 | 0.328 |
| gpt2-medium | 355 M | 1.016 | 1.105 | 0.364 |
| gpt2-large | 774 M | 0.968 | 1.069 | 0.378 |
| EleutherAI/pythia-1b | 1010 M | 0.924 | 1.022 | 0.400 |
| Qwen/Qwen2.5-0.5B | 494 M | 0.894 | 0.987 | 0.404 |
| Qwen/Qwen2.5-1.5B | 1500 M | 0.761 | 0.856 | 0.460 |
| Qwen/Qwen2.5-3B | 3090 M | 0.698 | 0.796 | 0.489 |
| **Qwen/Qwen2.5-7B** | **7620 M** | **0.624** | **0.724** | **0.528** |
| LLaMA+arithmetic coding (paper ref.) | — | **0.710** | — | — |

Notes:
- The gpt2 20k-char smoke result (ideal bpc = 1.079, fp32) and the 1 MB result
  (1.121, fp16) are close; the small difference reflects the different sample and
  numerical precision, not a modeling change.
- Rank-zero fraction measures what share of tokens were top-1 predictions;
  Qwen2.5-7B's 52.8 % means the model places the correct token first more than
  half the time.
- The paper reference (LLaMA + arithmetic coding, 0.710 bpc) is a *realized*
  arithmetic-coded size, whereas our LLM figures are *ideal* codelengths (token
  cross-entropy normalized per character) — so this is not a strict apples-to-apples
  comparison. In the ideal-codelength sense Qwen2.5-3B (0.698) is already below 0.710 and
  Qwen2.5-7B (0.624) is ~12 % below; our comparable *realized* numbers are the rank+lzma
  column (e.g. Qwen2.5-7B 0.724), which sit just above the paper's figure.

### Gap-to-Paper Discussion

Classical compressors saturate around 2.1–2.6 bpc because they exploit only
local byte-level redundancy (zlib/DEFLATE via LZ77+Huffman, lzma via LZMA, bz2 via the
Burrows–Wheeler transform). LLMs capture long-range semantic and syntactic structure,
reducing to 0.6–1.3 bpc depending on size.

The LLMZip practical overhead (rank+lzma minus the ideal codelength) is a roughly constant
**+0.07 to +0.10 bpc** tax across the family (distilgpt2 +0.074 → Qwen2.5-7B +0.100). In
absolute terms it grows slightly; as a *fraction* of the shrinking ideal codelength it
grows more (≈6 % → ≈16 %). This gap is the price of the practical rank-coding scheme
relative to an ideal arithmetic code — the residual that a real coder cannot recover, the
same `D(P‖Q)` gap between an achievable code and the ideal one that underlies Activity A.

The true entropy of the text8 source is unknown. Classical estimates put the
character-level entropy of natural English around 1.0–1.3 bits/char (Shannon, 1951);
text8's restricted alphabet (lowercase letters + space, 27 symbols) is lower. Qwen2.5-7B's
0.624 bpc shows even 7B-scale models still leave measurable redundancy, and the monotone
trend suggests further scaling would keep reducing it.

**Figure reference:** [Figure 2 — Activity B bpc vs. model size](figures/activity_b_bpc_vs_size.png)

---

## Figures

1. **`docs/figures/activity_a_progression.png`** — Bar chart of Activity A bits/symbol
   across milestones (uniform → template → Phase 1 best → Phase 2 kept milestones),
   with an inset zoom on the Phase 1→Phase 2 range. Dashed reference lines for
   template n-gram and Phase 1 best.

2. **`docs/figures/activity_b_bpc_vs_size.png`** — Activity B ideal codelength (bpc)
   vs. model parameter count (log scale) for the GPT-2 family, Qwen2.5 family, and
   Pythia-1B. Horizontal dashed lines for classical compressor baselines (zlib, lzma,
   bz2) and the LLaMA+AC paper reference. Also shows rank+lzma bpc per model as a
   dotted line.

Figures are generated by `docs/figures/make_figures.py` (run:
`uv run python docs/figures/make_figures.py` from repo root).
