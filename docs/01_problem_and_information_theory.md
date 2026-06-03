# 1. The Problem and Its Information-Theoretic Frame

A predictor and a compressor are the same machine. If a model says the next symbol has
probability `q`, the cheapest code that exists spends `−log₂ q` bits to write it down; betting
well and compressing well are therefore the same skill. The score in this project is just *how
many bits your beliefs cost*. This section sets up the problem and that equivalence, because
everything else follows from it.

## 1.1 The task

A source emits a sequence `x₁, …, x_N` over a finite alphabet (size `A`). A sequential
predictor must, at each position `i`, output a distribution over the next symbol using only the
past, `q_i(· | x₁^{i-1})`. After the true symbol is revealed it may update its state (online
adaptation), then predict the next position. It never looks ahead — prediction is strictly
causal.

For Activity A: `A = 16`, `N = 200,000`, context is capped at 256 past symbols, and the run
must finish within 600 seconds.

## 1.2 The score is empirical cross-entropy

The predictor is scored by its average log-loss in bits (all logs base 2):

```
Ĥ_Q(x₁ᴺ) = (1/N) · Σ_{i=1}^{N}  −log₂ q_i(x_i | x₁^{i-1})
```

Lower is better, and the quantity has two readings that we lean on throughout:

- **As code length.** A code assigning symbol `x` a length of `−log₂ q(x)` bits is optimal for a
  source distributed as `q` (Shannon 1948). So `−log₂ q_i(x_i | ·)` is the ideal number of bits
  to encode the symbol that actually occurred, given the model's belief at step `i`. The score is
  the average ideal code length per symbol. This is why we can score log-loss directly instead of
  building a file: an arithmetic coder achieves it up to vanishing overhead (Activity B checks
  this empirically).
- **As cross-entropy.** Averaging `−log₂ q` over the realized symbols is the empirical estimate of
  the cross-entropy `H(P, Q)` between the source `P` and the model `Q`.

## 1.3 Redundancy is model mismatch

The decomposition we optimize against is

```
H(P, Q) = H(P) + D(P ‖ Q),
```

where `H(P)` is the source entropy — the irreducible cost no causal predictor can beat on
average — and `D(P ‖ Q) ≥ 0` is the Kullback–Leibler divergence, equivalently the redundancy:
the extra bits per symbol we pay *only because the model is wrong*, zero iff `Q = P`. Since
`H(P)` is fixed by the source, minimizing the score means minimizing `D(P ‖ Q)`. "Build a better
predictor" becomes the precise objective: move `Q` closer to `P` in KL, online, within budget.

One corollary recurs in the analysis. Once two models differ by only a few ten-thousandths of a
bit per symbol, the difference is below the resolution of `D(P‖Q)` at this `N`; we have reached
the practical entropy floor for the model class, and further changes are indistinguishable from
noise. Section 3 makes this floor quantitative by estimating `H(P)` directly from the data.

## 1.4 Why "universal"

`P` is unknown. A *universal* source model aims to perform almost as well as if it were known,
for any `P` in a broad class; the excess over `H(P)` is the minimax redundancy, which for
well-behaved classes vanishes like `(k/2)·log N / N` per symbol (`k` = free parameters). The
model families used here — variable-order Markov, Context Tree Weighting, PPM — are the classical
tools for achieving small redundancy without knowing `P` in advance. Context Tree Weighting in
particular carries provable redundancy guarantees against the class of bounded-depth tree sources
(Willems, Shtarkov & Tjalkens 1995).

## 1.5 Constraints shape the design

The information-theoretic ideal — use the richest model, mix over all experts — collides with
hard limits, and those limits motivate the design:

| Constraint | Consequence for modeling |
|---|---|
| Strictly causal, online | No batch training on the test stream; state updates symbol-by-symbol. |
| Context ≤ 256 | Bounds the Markov order; favors *adaptive* order selection over a single fixed order. |
| ≤ 600 s for 200k symbols | Rules out heavy per-step computation; favors count-based models with cheap updates. |
| Finite, non-positive log-probs summing to 1 | Forces principled smoothing/escape — no symbol may get probability 0. |

The last row is itself information-theoretic: assigning probability 0 claims a symbol is
impossible, and if it occurs the log-loss is `+∞`. Laplace smoothing and PPM escape mechanisms
exist precisely to keep `D(P‖Q)` finite.

---

With the objective fixed, [**2. Methodology**](02_methodology.md) turns it into concrete models
and the autonomous loop that searched them.

[README](../README.md) · [1. Problem](01_problem_and_information_theory.md) · [2. Methodology](02_methodology.md) · [3. Results](03_results.md) · [4. Skills](04_skills_and_learnings.md)

**References.** Shannon (1948), *A Mathematical Theory of Communication*. Cover & Thomas,
*Elements of Information Theory*. Willems, Shtarkov & Tjalkens (1995), *The Context-Tree
Weighting Method*, IEEE Trans. IT.
