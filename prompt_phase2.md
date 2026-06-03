# Phase 2 Autoresearch Prompt — Copy & Paste into pi

You are in the repository root.

## Context

This is **Phase 2** of the ITML Universal Source Modeling Challenge autoresearch loop.

**Phase 1 is complete and archived.** It ran 121 runs exploring fixed-order n-grams, mixtures, and online adaptation. The best result was **Run 117 at 2.969228 bits/symbol** (n=5 interpolated per-order mixture + n=3 online auxiliary). Phase 1 artifacts are in `archive/phase1_ngram_exploration/`.

## Your Mandate for Phase 2

**Read these files first:**
- `AGENTS.md` — operating guide
- `autoresearch.md` — loop rules and baseline
- `autoresearch.ideas.md` — seeded ideas for Phase 2
- `explanations/README.md` — template for run explanations

**Baseline to beat:** 2.969228 bits/symbol (`submissions/best_predictor_phase1.py`)

**CRITICAL RULES:**
1. **Do NOT run more n-gram hyperparameter sweeps.** The n-gram space is exhausted. No more Laplace tweaks, no more mixture weight brackets, no more per-order constant tuning.
2. **Each run must try a genuinely different architecture:** Variable-Order Markov Model, Context Tree Weighting, or PPM-style escape mechanism.
3. Hybrids with the Phase 1 best predictor are allowed (e.g., 0.7 new model + 0.3 phase1), but the new component must be substantive.
4. Before every run, write `explanations/run_###.md` using the template.
5. Use `./autoresearch.sh` to benchmark. It runs the official 200k-token evaluator.
6. Beat 2.969228 to keep a run. Discard anything worse unless it teaches a critical architectural lesson.

## Suggested First Runs (from ideas file)

1. **PPM-C, max_order=5, escape method C, no exclusion.** Simplest strong variable-order model.
2. **PPM-C, max_order=6.** Test depth sensitivity.
3. **Min-count VOMM, max_depth=6, min_count=4.** Simplest variable-order without full PPM escapes.
4. **CTW approximate, max_depth=4, KT estimator, canonical 0.5 mixing.** Simplified context tree.
5. **PPM + Phase 1 best mixture.** Hedge risk while testing PPM.

## Evaluation Discipline

Before trusting any score:
1. Run a **smoke test** (5000 tokens) to catch crashes and invalid probabilities.
2. Run the **full 200k evaluation** for the official score.
3. If a run looks like a big improvement, **re-run it once** to verify stability.
4. After every 5-10 runs, check `autoresearch.jsonl` to ensure scores are consistent.

## Start Now

Enter autoresearch mode and begin with the first idea from `autoresearch.ideas.md`.

```
/autoresearch
```
