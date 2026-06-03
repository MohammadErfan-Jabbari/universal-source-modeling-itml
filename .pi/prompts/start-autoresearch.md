---
description: Start or resume ITML USM autoresearch on Centcom — Phase 2
---
You are in `/home/centcom/data/Masters/Information_Theory_for_ML/USM_Challenge`.

This is **Phase 2** of the autoresearch loop. Phase 1 (121 runs) exhaustively explored fixed-order n-grams and mixtures. The best result was **Run 117 at 2.969228 bits/symbol**.

**Your mandate for Phase 2:**
1. Read `AGENTS.md`, `autoresearch.md`, `autoresearch.ideas.md`, and `EXPERIMENTS.md`.
2. The baseline to beat is **2.969228 bits/symbol** (submissions/best_predictor_phase1.py).
3. **Do NOT run more n-gram Laplace sweeps or n-gram hyperparameter tuning.** That space is exhausted.
4. Explore **structurally different model families**:
   - **Variable-Order Markov Models (VOMM)** — e.g., min-count depth selection, confidence-based depth, PPM-style tries
   - **Context Tree Weighting (CTW)** — e.g., KT estimator, recursive tree mixing, approximate CTW
   - **Prediction by Partial Matching (PPM)** — e.g., PPM-C/PPM-D with escape mechanisms, exclusions, bounded PPM
5. Hybrids with the Phase 1 best predictor are allowed (e.g., 0.7 new model + 0.3 phase1).
6. Before each run, write `explanations/run_###.md` using the template in `explanations/README.md`.
7. Use `./autoresearch.sh` to benchmark. The evaluator runs the official 200k-token public-practice test.
8. If a run improves on 2.969228, keep it. Otherwise discard unless it teaches a critical lesson.
9. The loop is configured for up to 500 iterations total (Phase 1 used 121).

Enter autoresearch mode with `/autoresearch` and begin exploring the first idea from `autoresearch.ideas.md`.
