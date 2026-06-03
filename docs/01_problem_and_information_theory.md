# 1. The Problem and Its Information-Theoretic Frame

This document derives, from first principles, *why* the score used in this project is the
right one and *what* it measures. Everything else in the repository follows from it.

## 1.1 The task

A source emits a sequence `x₁, x₂, …, x_N` over a finite alphabet `𝒳` (size `A`). A
**sequential predictor** must, at each position `i`, emit a probability distribution over
the next symbol using **only the past**:

```
q_i( · | x₁, …, x_{i-1} )      a distribution on 𝒳, depends on the past only
```

After the true symbol `x_i` is revealed, the predictor may update its state (online
adaptation). It then predicts position `i+1`, and so on. No lookahead, no peeking at the
future — the predictor is strictly **causal**.

For Activity A: `A = 16`, `N = 200,000`, context length is capped at 256 past symbols, and
the whole run must finish within 600 seconds.

## 1.2 The score is empirical cross-entropy

The predictor is scored by the **average log-loss** in bits (all logs base 2):

```
Ĥ_Q(x₁ᴺ) = (1/N) · Σ_{i=1}^{N}  −log₂ q_i(x_i | x₁^{i-1})
```

Lower is better. Two readings of this quantity, both essential:

1. **As a code length.** A code that assigns symbol `x` a length of `−log₂ q(x)` bits is
   optimal for a source with distribution `q` (Shannon). So `−log₂ q_i(x_i | …)` is the
   *ideal number of bits* to encode the actual symbol `x_i` under the model's belief at
   step `i`. The score is therefore the **average ideal code length per symbol** achieved
   by the model — prediction and compression are the same problem.

2. **As cross-entropy.** Averaging `−log₂ q` of the realized symbols is the empirical
   estimate of the cross-entropy `H(P, Q)` between the true source `P` and the model `Q`.

This is why the leaderboard scores log-loss directly rather than a compressed file size:
the two are equivalent up to the (vanishing) overhead of a real arithmetic coder.

## 1.3 Redundancy is model mismatch

The central identity of the whole course, and of this project:

```
H(P, Q) = H(P) + D(P ‖ Q)
```

- `H(P)` — the **source entropy**, the irreducible cost. No causal predictor can beat it
  on average; it is the floor.
- `D(P ‖ Q) ≥ 0` — the **Kullback–Leibler divergence**, equivalently the **redundancy**:
  the extra bits per symbol we pay *purely because our model `Q` is wrong*. It is zero iff
  `Q = P`.

So minimizing the score = minimizing cross-entropy = minimizing `D(P‖Q)` (since `H(P)` is
fixed by the source). **Every gain in this repository is a reduction in model mismatch.**
This reframes "build a better predictor" as the precise, measurable objective "make `Q`
closer to `P` in KL divergence, online, within budget."

A corollary used repeatedly in the analysis: once successive models differ by only a few
ten-thousandths of a bit per symbol, we are within `D(P‖Q)`'s resolution at this `N` — we
have hit the practical entropy floor for the model class, and further architecture changes
are indistinguishable from noise.

## 1.4 Why "universal"

The true source `P` is unknown. A **universal** source model aims to do almost as well as
if `P` were known, for *any* `P` in a broad class — the excess over `H(P)` is the
**minimax redundancy**, which for nice classes grows like `(k/2) log N / N → 0` per symbol
(here `k` = number of free parameters). The model families explored here — variable-order
Markov, Context Tree Weighting, PPM — are exactly the classical tools for achieving small
redundancy without knowing `P` in advance. CTW in particular has provable redundancy
guarantees against the class of bounded-depth tree sources.

## 1.5 Constraints shape the design

The information-theoretic ideal (use the richest possible model, mix over all experts) is
bounded by hard engineering limits, and these directly motivate the design choices:

| Constraint | Consequence for modeling |
|---|---|
| Strictly causal, online | No batch training on the test sequence; counts and state are updated symbol-by-symbol after each reveal. |
| Context ≤ 256 | Bounds the maximum Markov order; favors *adaptive* order selection over a single fixed high order. |
| ≤ 600 s for 200k symbols | Rules out heavy per-step computation (e.g. large neural nets); favors count-based models with cheap updates. |
| Probabilities must be finite, non-positive log-probs summing to 1 | Forces principled smoothing/escape so no symbol ever gets probability 0 (an infinite penalty if it then occurs). |

The last row is itself an information-theoretic point: assigning a symbol probability 0 is
claiming it is *impossible*; if it occurs, the log-loss is `+∞`. Smoothing (Laplace) and
escape mechanisms (PPM) exist precisely to keep `D(P‖Q)` finite.

---

Next: [**2. Methodology**](02_methodology.md) — the concrete model families used to push
`D(P‖Q)` down, and the autonomous loop that searched them.
