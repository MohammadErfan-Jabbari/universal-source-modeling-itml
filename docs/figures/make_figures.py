"""
make_figures.py — generate Activity A and Activity B result figures.

Self-contained: reads directly from the data sources in the repo.
Run from repo root:
    uv run python docs/figures/make_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1 — Activity A: score progression across milestones
# ---------------------------------------------------------------------------

def fig_activity_a_progression():
    milestones = [
        ("Uniform\n(baseline)",         4.000000,  None),
        ("Template n-gram\n(n=4 Laplace)", 2.998625, None),
        ("Phase 1 best\n(Run 117)\nn=5 interp. + n=3 aux", 2.969228, 9.72),
        ("Phase 2: R004\nConf-gated PPM+P1",  2.969019, 15.64),
        ("Phase 2: R010\nCount-dep. CTW+P1",  2.968990, 22.36),
        ("Phase 2: R014\nProb-depth VOMM+P1", 2.968982, 34.54),
        ("Phase 2: R033\nOrder-7 half-esc. PPM-A", 2.968939, 39.97),
        ("Phase 2: R044\nJS disagree-boost PPM-A", 2.968938, 42.89),
        ("Phase 2: R067\nSqrt-support disagr. (best)", 2.968933, 43.85),
    ]

    labels = [m[0] for m in milestones]
    scores = [m[1] for m in milestones]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 5.5))

    colors = ["#c0392b"] + ["#e67e22"] + ["#2980b9"] + ["#27ae60"] * (len(milestones) - 3)
    bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.8, width=0.65)

    # Annotate bar tops
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{score:.6f}",
            ha="center", va="bottom", fontsize=7.5, rotation=0, color="#2c3e50"
        )

    # Highlight the Phase 1 baseline as a dashed line
    ax.axhline(2.969228, color="#2980b9", linestyle="--", linewidth=1.2, alpha=0.7, label="Phase 1 best (2.969228)")
    ax.axhline(2.998625, color="#e67e22", linestyle=":", linewidth=1.2, alpha=0.7, label="Template n-gram (2.998625)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Bits per symbol (lower is better)", fontsize=11)
    ax.set_title("Activity A — Score Progression: Uniform → Template → Phase 1 → Phase 2", fontsize=12, fontweight="bold")

    # Zoom y-axis to show resolution
    ax.set_ylim(2.960, 4.08)

    # Inset zoom panel on Phase 1/2 region
    axins = ax.inset_axes([0.55, 0.45, 0.43, 0.50])
    zoom_idx = list(range(2, len(milestones)))  # Phase1 best onward
    xz = np.arange(len(zoom_idx))
    zscores = [scores[i] for i in zoom_idx]
    zcolors = [colors[i] for i in zoom_idx]
    axins.bar(xz, zscores, color=zcolors, edgecolor="white", linewidth=0.6, width=0.65)
    axins.set_xticks(xz)
    axins.set_xticklabels([labels[i].split("\n")[0] for i in zoom_idx], fontsize=6.5, rotation=30, ha="right")
    axins.set_ylim(2.9688, 2.9695)
    axins.axhline(2.969228, color="#2980b9", linestyle="--", linewidth=1.0, alpha=0.8)
    axins.set_title("Phase 1→2 zoom", fontsize=8)
    axins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
    axins.tick_params(axis='y', labelsize=7)

    legend_patches = [
        mpatches.Patch(color="#c0392b", label="Uniform baseline"),
        mpatches.Patch(color="#e67e22", label="Template n-gram baseline"),
        mpatches.Patch(color="#2980b9", label="Phase 1 best"),
        mpatches.Patch(color="#27ae60", label="Phase 2 kept milestones"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "activity_a_progression.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Activity B: bits/char vs model size (params)
# ---------------------------------------------------------------------------

# Approximate parameter counts (public knowledge)
MODEL_PARAMS = {
    "distilgpt2":          82e6,
    "gpt2":               124e6,
    "gpt2-medium":        355e6,
    "gpt2-large":         774e6,
    "Qwen/Qwen2.5-0.5B":  494e6,
    "Qwen/Qwen2.5-1.5B": 1500e6,
    "Qwen/Qwen2.5-3B":   3090e6,
    "Qwen/Qwen2.5-7B":   7620e6,
    "EleutherAI/pythia-1b": 1010e6,
}

def load_llm_results():
    results_dir = os.path.join(REPO_ROOT, "activity_b_llmzip", "results")
    llm = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            d = json.load(f)
        if "model" not in d:
            continue
        # Prefer 1MB results over 20k smoke
        if d.get("chars", 0) < 500000:
            continue
        llm.append(d)
    return llm


def fig_activity_b_bpc_vs_size():
    llm_data = load_llm_results()

    gpt2_family = []
    qwen_family = []
    pythia_family = []

    for d in llm_data:
        model = d["model"]
        bpc = d["bits_per_character"]
        params = MODEL_PARAMS.get(model)
        if params is None:
            continue
        rank_bpc = d.get("rank_varint_lzma_bits_per_character")
        entry = {"model": model, "params": params, "ideal_bpc": bpc, "rank_lzma_bpc": rank_bpc}
        if "gpt2" in model.lower() or "distilgpt2" in model.lower():
            gpt2_family.append(entry)
        elif "qwen" in model.lower():
            qwen_family.append(entry)
        elif "pythia" in model.lower():
            pythia_family.append(entry)

    # Sort by params
    for lst in [gpt2_family, qwen_family, pythia_family]:
        lst.sort(key=lambda x: x["params"])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Classical baselines
    baselines = {
        "zlib_9 (2.638)":   2.638112,
        "lzma_9 (2.198)":   2.197696,
        "bz2_9 (2.098)":    2.097872,
    }
    baseline_styles = [
        ("#e74c3c", "-.",  "zlib_9 (2.638 bpc)"),
        ("#e67e22", "--",  "lzma_9 (2.198 bpc)"),
        ("#c0392b", ":",   "bz2_9 (2.098 bpc)"),
    ]
    for (color, ls, label), val in zip(baseline_styles, baselines.values()):
        ax.axhline(val, color=color, linestyle=ls, linewidth=1.5, alpha=0.8, label=label)

    # Paper reference
    ax.axhline(0.710, color="#8e44ad", linestyle="--", linewidth=1.8, alpha=0.9, label="LLaMA+AC paper ref (0.710 bpc)")

    # Plot families
    def plot_family(family, color, marker, label, plot_rank=True):
        if not family:
            return
        xs = [e["params"] / 1e9 for e in family]
        ys_ideal = [e["ideal_bpc"] for e in family]
        ax.plot(xs, ys_ideal, color=color, marker=marker, markersize=8,
                linewidth=2, label=f"{label} — ideal bpc", zorder=5)
        for x, y, e in zip(xs, ys_ideal, family):
            ax.annotate(e["model"].split("/")[-1], (x, y),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=color)
        if plot_rank:
            ys_rank = [e["rank_lzma_bpc"] for e in family if e["rank_lzma_bpc"]]
            xs_rank = [e["params"] / 1e9 for e in family if e["rank_lzma_bpc"]]
            if ys_rank:
                ax.plot(xs_rank, ys_rank, color=color, marker=marker, markersize=6,
                        linewidth=1.2, linestyle=":", alpha=0.6, label=f"{label} — rank+lzma bpc")

    plot_family(gpt2_family,   "#2980b9", "o", "GPT-2 family")
    plot_family(qwen_family,   "#27ae60", "s", "Qwen2.5 family")
    plot_family(pythia_family, "#9b59b6", "^", "Pythia")

    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (billions, log scale)", fontsize=11)
    ax.set_ylabel("Bits per character (lower is better)", fontsize=11)
    ax.set_title("Activity B — LLM Ideal Codelength vs. Model Size on text8 (1MB)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 2.9)
    ax.set_xlim(0.05, 15)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2g}B"))
    ax.legend(fontsize=8.5, loc="upper right", ncol=1)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "activity_b_bpc_vs_size.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    f1 = fig_activity_a_progression()
    f2 = fig_activity_b_bpc_vs_size()
    print("Done.")
