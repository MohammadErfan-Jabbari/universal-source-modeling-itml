# Autoresearch Ideas — ITML USM Challenge (Phase 2)

Phase 1 (121 runs) exhaustively explored fixed-order n-grams, mixtures, and online adaptation.
Best score: **2.969228 bits/symbol** (Run 117, n=5 interpolated per-order mixture + n=3 online auxiliary).

**Phase 2 mandate:** Explore structurally different model families. Do NOT run more n-gram hyperparameter sweeps.
The baseline for Phase 2 is 2.969228 bits/symbol. Any new model must beat this to be kept.

---

## Direction 1: Variable-Order Markov Models (VOMM)

**Core idea:** Instead of fixing context order to n=4/5, let the model choose context depth dynamically per history.

**Why it may work:**
- Some contexts need long memory (e.g., periodic patterns); others need only 1-2 symbols.
- Fixed n=5 wastes variance on sparse contexts and underfits where longer memory exists.
- VOMM directly addresses the bias-variance tradeoff by adapting model complexity to context support.

**Implementation paths:**

### 1a. PPM-A style: depth-limited trie with escape probability
- Build a suffix trie up to max depth D (e.g., 6-8).
- At each prediction, walk the trie using the recent suffix.
- Use escape probability when a context has not seen the next symbol.
- Escape can be: Method A (count of seen symbols / total count), Method B, or Method C.
- Back off to shorter context recursively until a prediction is possible.

### 1b. Min-count depth selection
- Use the longest context with at least `min_count` observations.
- If none, back off to shorter context.
- This is simpler than full PPM but still variable-order.

### 1c. Probabilistic depth selection
- Choose depth proportional to context confidence (e.g., entropy of next-symbol distribution).
- High-confidence contexts -> deeper; uncertain contexts -> shallower.

**Key parameters to explore:**
- `max_depth`: 5, 6, 7, 8
- Escape method: A, B, C (or custom)
- `min_count` threshold for trusting a node
- Smoothing at leaf nodes: Laplace, Lidstone, or KT estimator

**Risk:**
- Trie can grow large; must stay within memory/time constraints.
- Escape probability accounting must be exact (probabilities must sum to 1).
- Online updates to the trie must be fast.

---

## Direction 2: Context Tree Weighting (CTW)

**Core idea:** Maintain a context tree where each node mixes its local estimate with the weighted estimate of its children/suffixes.

**Why it may work:**
- CTW is theoretically grounded in universal coding; it competes with the best tree source.
- It naturally handles uncertainty about which context length is best by mixing over the tree.
- Unlike fixed n-grams, CTW can represent irregular context dependencies.

**Implementation paths:**

### 2a. Simplified CTW with KT estimator
- Build a suffix tree (or context tree) up to depth D.
- Each node maintains a Krichevsky-Trofimov (KT) estimate of next-symbol probabilities.
- The estimate at each node is a weighted mixture of the local KT estimate and the weighted estimate of all child contexts.
- Weight is typically 1/2 for local vs children mixture (the original CTW weighting).

### 2b. Count-dependent CTW (approximate)
- Instead of full binary tree mixing, use a sparse tree where nodes exist only when counts are nonzero.
- Mix local estimate with shorter-context estimate using count-dependent weights.
- This bridges CTW and the Phase 1 interpolation approach.

### 2c. CTW + n-gram hybrid
- Use CTW for low-order contexts (0..3) and n-gram for higher orders.
- Or use CTW as the main model and the Phase 1 best predictor as a fixed auxiliary.

**Key parameters to explore:**
- `max_depth`: 4, 5, 6
- Local estimator: KT (Krichevsky-Trofimov) vs Laplace vs Lidstone
- Mixing weight: 0.5 (canonical CTW) vs count-dependent
- Sparse vs full tree

**Risk:**
- Full CTW for alphabet 16 is more complex than binary CTW.
- Must ensure probability normalization and sequential validity.
- Runtime: tree traversal per symbol may be slower than n-gram dict lookup.

---

## Direction 3: PPM (Prediction by Partial Matching) with Explicit Escapes

**Core idea:** PPM is the canonical variable-order compression method. It maintains a trie of contexts and uses explicit escape probabilities to handle unseen symbols within a context.

**Why it may work:**
- PPM is historically one of the strongest classical compressors for text/symbol sequences.
- It explicitly models "this context has never seen symbol x" via escapes, rather than smoothing blindly.
- For alphabet size 16, PPM should be both fast and powerful.

**Implementation paths:**

### 3a. PPM-C (escape based on number of unique continuations)
- For a context, if symbol x has been seen, probability proportional to count(x).
- If not seen, escape probability proportional to (number of unique seen symbols in this context) / (total count + unique count).
- Distribute escaped mass to shorter context recursively.

### 3b. PPM-D (zero-frequency problem handling)
- More aggressive escape for contexts with low diversity.
- Blend escape with a uniform fallback at order -1.

### 3c. PPM + exclusions
- When backing off, exclude symbols that already had probability mass at higher orders.
- Prevents double-counting and sharpens distribution.

### 3d. Bounded PPM
- Cap max order at 5-7 to control runtime.
- Use sparse trie (only contexts that actually appeared).
- Optional: mix PPM with the Phase 1 best predictor as a safety net.

**Key parameters to explore:**
- `max_order`: 4, 5, 6, 7
- Escape method: A, B, C, D, X
- Exclusion: on/off
- Smoothing at order -1: uniform vs train-unigram
- Update rule: update all contexts or only matched context

**Risk:**
- Escape probability accounting is tricky; must sum to exactly 1.
- Exclusions add complexity and can backfire if not calibrated.
- Must handle the case where even order-0 context has not seen the symbol.

---

## Cross-cutting ideas (apply to any direction)

### Hybrid: new model + Phase 1 best predictor
- Any new model can be mixed with `best_predictor_phase1.py`.
- Fixed weight (e.g., 0.7 new + 0.3 phase1) or count-dependent.
- This hedges risk: if the new model fails, the mixture still has the known-good component.

### Online adaptation for tree models
- Update the trie/CTW/PPM counts as test symbols arrive.
- But be careful: too aggressive online updates can hurt.
- Try: update all contexts vs update only the contexts used for the prediction.

### Train-derived validation before public practice
- Before benchmarking on public practice, evaluate on held-out suffix of `data/generator/train.npy`.
- This catches overfitting early and saves public-practice runs for final validation.

---

## Phase 2 rules

1. **Do NOT run more n-gram Laplace sweeps.** That space is exhausted.
2. **Do NOT tweak Phase 1 hyperparameters.** The n-gram mixture is archived.
3. **Each run must try a genuinely different architecture** (VOMM, CTW, or PPM).
4. **Hybrid ideas are allowed** (e.g., PPM + Phase 1 predictor), but the new component must be substantive.
5. **Beat 2.969228 to be kept.** Anything worse is a discard unless it teaches a critical lesson.
6. **Explain the information-theoretic motivation** in `explanations/run_###.md`.

---

## Suggested first 10 Phase 2 runs

1. **PPM-C, max_order=5, escape method C, no exclusion.** Simplest strong PPM.
2. **PPM-C, max_order=6, same settings.** Test depth sensitivity.
3. **PPM with exclusions, max_order=5.** Classic refinement.
4. **Min-count VOMM, max_depth=6, min_count=4.** Simplest variable-order.
5. **CTW approximate, max_depth=4, KT estimator, canonical 0.5 mixing.** Simplified CTW.
6. **CTW approximate, max_depth=5, same.** Depth sensitivity.
7. **PPM + Phase 1 best mixture (0.7 PPM / 0.3 phase1).** Hedge new model risk.
8. **PPM-D, max_order=5, more aggressive escape.** Compare escape methods.
9. **CTW + Phase 1 best mixture (0.7 CTW / 0.3 phase1).** Hybrid tree model.
10. **Probabilistic depth VOMM, max_depth=6, confidence-based depth.** More adaptive depth.

After run 10, update this file based on actual results.
