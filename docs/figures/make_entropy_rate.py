"""Estimate the empirical entropy rate of the Activity A source and plot it
against the predictor's achieved log-loss.

For each Markov order k we compute the plug-in conditional empirical entropy
H(X_i | X_{i-k}^{i-1}) in bits, on both the training and public-practice
sequences. Low orders are well sampled and reliable; past order ~3 the number
of distinct contexts (16^k) overwhelms the 3e5 symbols available, so the
plug-in estimate is dominated by sampling noise and biases sharply downward —
those points are artifacts, not structure.

Run from repo root:  uv run python docs/figures/make_entropy_rate.py
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "activity_a_entropy_rate.png"

# Achieved official log-loss (bits/symbol), from EXPERIMENTS.md / autoresearch.jsonl
BEST_TEST = 2.968933        # Run 067, public-practice test
TRAIN_VAL = 2.759518        # Run 067, train-derived validation tail
TEMPLATE = 2.998625         # official n-gram baseline
RELIABLE_MIN_SAMPLES = 50   # avg samples/context below which Hk is unreliable
KMAX = 6


def cond_entropy_rate(seq, k):
    if k == 0:
        c = np.bincount(seq, minlength=16).astype(float)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log2(p)).sum()), float(len(seq))
    ctx = defaultdict(lambda: np.zeros(16))
    for i in range(k, len(seq)):
        ctx[tuple(seq[i - k:i])][seq[i]] += 1
    n = len(seq) - k
    H = 0.0
    for cnt in ctx.values():
        N = cnt.sum()
        p = cnt[cnt > 0] / N
        H += (N / n) * (-(p * np.log2(p)).sum())
    return float(H), n / len(ctx)


def curve(path):
    s = np.load(path).astype(int)
    Hs, spc = [], []
    for k in range(KMAX + 1):
        h, c = cond_entropy_rate(s, k)
        Hs.append(h)
        spc.append(c)
    return np.array(Hs), np.array(spc)


def main():
    ks = np.arange(KMAX + 1)
    H_train, spc = curve(ROOT / "data" / "generator" / "train.npy")
    H_test, _ = curve(ROOT / "data" / "public_practice" / "test.npy")
    reliable = spc >= RELIABLE_MIN_SAMPLES

    fig, ax = plt.subplots(figsize=(8, 5))
    # shade the undersampled region
    first_bad = int(np.argmax(~reliable)) if (~reliable).any() else KMAX + 1
    if first_bad <= KMAX:
        ax.axvspan(first_bad - 0.5, KMAX + 0.5, color="0.92", zorder=0)
        ax.text(first_bad + 0.05, 3.4, "undersampled\n(plug-in bias ↓)",
                fontsize=9, color="0.4", va="top")

    ax.plot(ks, H_test, "o-", color="#1f77b4", label=r"$\hat H_k$ public-practice")
    ax.plot(ks, H_train, "s--", color="#7fbf7f", label=r"$\hat H_k$ train", alpha=0.8)
    ax.axhline(BEST_TEST, color="#2ca02c", lw=1.6,
               label=f"predictor (Run 067): {BEST_TEST:.3f}")
    ax.axhline(TEMPLATE, color="#ff7f0e", lw=1.0, ls=":",
               label=f"n-gram baseline: {TEMPLATE:.3f}")
    ax.axhline(4.0, color="0.6", lw=0.8, ls=":", label="uniform: 4.000")

    ax.set_xlabel("Markov order $k$ (context length)")
    ax.set_ylabel("conditional empirical entropy  $\\hat H_k$  (bits/symbol)")
    ax.set_title("Activity A: empirical entropy rate vs. achieved log-loss")
    ax.set_xticks(ks)
    ax.set_ylim(0.5, 4.15)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)
    print("reliable orders (>=%d samples/ctx): k <= %d" % (RELIABLE_MIN_SAMPLES, first_bad - 1))
    for k in ks:
        print(f"  k={k}  H_test={H_test[k]:.4f}  H_train={H_train[k]:.4f}  samples/ctx={spc[k]:.1f}")


if __name__ == "__main__":
    main()
