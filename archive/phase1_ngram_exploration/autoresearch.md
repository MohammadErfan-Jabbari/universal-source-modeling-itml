# Autoresearch: ITML Activity A Sequential Predictor

## Objective

Improve the Activity A predictor for the Universal Source Modeling Challenge.

The official objective is empirical average log-loss on a sequential source over alphabet size 16:

```text
Hhat_Q(x_1^N) = (1/N) * sum_i -log2 q_i(x_i | x_1^{i-1})
```

The primary optimization target is lower `bits_per_symbol` on the official public-practice evaluator, while preserving strict sequential validity, runtime safety, and a presentation-quality explanation.

This is not a generic code-golf task. The final result must be defensible in the Information Theory for ML course: prediction as coding, cross-entropy as ideal codelength, and redundancy as model mismatch.

## Metrics

- **Primary**: `bits_per_symbol` (bits/symbol, lower is better)
- **Secondary**:
  - `elapsed_seconds` (lower is better as a tie-breaker and safety signal)
  - `evaluated_tokens` (must equal `200000` for full runs)
  - `timed_out` (must be `0`)
  - smoke-test validity with probability validation
  - explanation quality and theoretical defensibility

## Baseline

Official template predictor:

```text
submissions/template_predictor.py
```

Verified full public-practice baseline:

```text
FINAL_SCORE bits_per_symbol=2.9986251144 elapsed_seconds=1.858381 timed_out=False evaluated_tokens=200000
```

Uniform baseline is `4.0` bits/symbol. Beating about `3.0` is the first meaningful target.

## How to Run

Run the benchmark script:

```bash
./autoresearch.sh
```

By default this evaluates:

```text
PREDICTOR_PATH=submissions/template_predictor.py
TEST_PATH=data/public_practice/test.npy
```

To evaluate a different predictor:

```bash
PREDICTOR_PATH=submissions/my_predictor.py ./autoresearch.sh
```

The script emits parseable lines:

```text
METRIC bits_per_symbol=...
METRIC elapsed_seconds=...
METRIC timed_out=0
METRIC evaluated_tokens=200000
```

## Required pre-run explanation

Before every benchmark run, write an explanation file:

```text
explanations/run_###.md
```

`autoresearch.sh` infers the expected run number from `autoresearch.jsonl` and refuses to run if the matching explanation file is missing or lacks required headings. This is deliberate: every score should have a reasoning trail.

Each explanation must include:

- `## Proposed change`
- `## Source and evidence`
- `## Course-material connection`
- `## Hypothesis`
- `## Risks`
- `## Validation plan`

Read `explanations/README.md` for the template.

## Required reading before decisions

Before deciding on an experiment, the agent must review the relevant local material:

1. `AGENTS.md` for project rules and repo map.
2. `COMPETITION_RULES.md` and `submissions/README.md` for official rules.
3. `docs/USM_challenge_slides.pdf` if presentation/grading context matters.
4. Course notes in `lectures/`, especially:
   - Lectures 6-9: source coding and prefix/Huffman-related ideas.
   - Lectures 10-12: arithmetic coding and prediction-to-code-length connection.
   - Lectures 13-15: universal compression, Lempel-Ziv, minimax redundancy.
5. The last five files in `explanations/run_*.md`, which are also surfaced by `autoresearch.hooks/before.sh`.

Do not mechanically read every lecture on every iteration if the run is narrow. Instead, read the relevant notes and cite them in the explanation.

## Search policy

The agent may use its own knowledge and available search tools. Good sources include:

- local course notes in `lectures/`
- official project files in this repo
- web search for PPM, CTW, variable-order Markov models, Bayesian smoothing, universal coding
- research-paper search for context-tree weighting, prediction by partial matching, redundancy of Markov sources
- code search for implementation patterns, but final code must be self-contained and understood

Every explanation file must identify the source class for the idea:

```text
source: course material | agent prior knowledge | web search | paper | code search | experiment observation
```

If external material influenced an experiment, cite it with title/URL or paper name in the explanation.

## Files in scope

Agents may modify/add:

- `submissions/*.py`
- new project modules under `submissions/` or clearly named helper packages
- `competition/predictors/*.py` only when adding new predictor classes; do not alter official baseline semantics silently
- `scripts/*.sh` and `scripts/*.py`
- `EXPERIMENTS.md`
- `README.md`
- `autoresearch.md`
- `autoresearch.sh`
- `autoresearch.checks.sh`
- `autoresearch.hooks/*`
- `explanations/*.md`

## Off limits

Do not edit for score improvement:

- `competition/run_live_eval.py`
- `competition/evaluation/harness.py`
- `data/public_practice/test.npy`
- `data/generator/train.npy`
- `data/live_release/`

Do not use future symbols, manually inspect live test content, or tune by reading hidden/live data.

## Constraints

- Predictor must expose `build_predictor(alphabet_size, max_context_length)`.
- The returned object must be a `competition.predictors.base.Predictor`.
- `predict_next(context)` returns log2 probabilities, not raw probabilities or logits.
- Probabilities must be finite, non-positive in log2 form, and sum to 1 after exponentiation.
- Must be strictly sequential: no lookahead.
- Must respect context length 256.
- Must avoid external API calls at live evaluation time.
- Must finish well under 600 seconds on the official 200000-token prefix.
- Prefer robust improvements over practice-set memorization.

## Initial search space

Start classical and interpretable:

- n-gram order tuning
- smoothing variants: Laplace, Lidstone, Witten-Bell-like, Kneser-Ney-like ideas if justified
- hard vs soft backoff
- mixtures of n-gram orders
- variable-order Markov predictors
- PPM-style exclusion/backoff variants
- context trees / CTW-inspired mixtures
- online adaptation schedules and priors
- count table layout and runtime optimization

Avoid as first moves:

- huge neural models
- API-dependent methods
- methods that only memorize public practice `test.npy`
- changes that make the predictor hard to explain in a 10-minute presentation

## Experiment loop expectations

For each run:

1. Read this file, `AGENTS.md`, and relevant source/course material.
2. Read the last five explanation files.
3. Choose one coherent hypothesis.
4. Write `explanations/run_###.md` before running.
5. Change only the needed files.
6. Run `./autoresearch.sh`.
7. If the benchmark passes, run/observe `autoresearch.checks.sh` through autoresearch backpressure.
8. Log the result with precise ASI fields: hypothesis, source, changed_files, learned, next_focus.
9. Update `autoresearch.md` and/or `EXPERIMENTS.md` periodically so context survives resets.

## What's been tried

- Run 001: baseline official n=4 hard-backoff Laplace n-gram from `submissions/template_predictor.py`: `2.9986251144` bits/symbol; checks passed.
- Run 002: n=5 hard-backoff with Laplace `1.0` and online adaptation worsened to `3.1231208254` bits/symbol; likely sparse high-order contexts and/or over-smoothing. Do not retry plain n=5/Laplace=1.0.
- Run 003: n=5 hard-backoff with lower Laplace `0.1` worsened further to `3.2288198643` bits/symbol. The plain n=5 hard-backoff issue is not just over-smoothing; sparse length-4 contexts and hard selection are likely harmful. Prefer n=4 smoothing sweeps or thresholded/mixture backoff before more high-order hard-backoff runs.
- Run 004: n=4 hard-backoff with lower Laplace `0.1` worsened to `3.0358045164` bits/symbol. Very low pseudocounts underweight surprising symbols even at baseline order. Avoid very low smoothing for hard-backoff n-grams; try gentler values like `0.5`, online-adaptation variants, or non-hard backoff.
- Run 005: n=4 hard-backoff with Laplace `0.5` scored `3.0001517271`, only slightly worse than baseline. For plain n=4 hard-backoff, Laplace `1.0` remains best among tested `0.1`, `0.5`, and `1.0`.
- Run 006: n=4 train-only/static (`adapt_online=False`) worsened to `3.0109261870` while running faster. Online adaptation improves log-loss enough to keep enabled; avoid static variants unless runtime becomes the bottleneck.
- Run 007: thresholded n=5 with min_count `8` applied to all nonzero orders worsened to `3.0626460725`. This likely backed off too aggressively at lower orders and/or high-order contexts remain unhelpful with Laplace `1.0`. If revisiting thresholding, threshold only the new length-4 context so baseline n=4 behavior is preserved otherwise.
- Run 008: thresholding only the new length-4 context also worsened to `3.0655748167`. Stop n=5 hard/thresholded context experiments for now; length-4 hard-context estimates appear harmful even when supported by at least 8 observations.
- Run 009: count-dependent interpolated n=4 over orders 0..3 with `C=16` improved to `2.9817296585` bits/symbol; checks passed. This is the first successful direction. Runtime increased to `3.59s`, still safe. Confidence was low (`0.5x` noise floor), so tune/confirm interpolation constants and later validate on train-derived splits.
- Run 010: interpolated n=4 with more conservative `C=32` improved further to `2.9794649970` bits/symbol; checks passed. This suggests lower-order borrowing is beneficial and the optimum may use even weaker high-order confidence. Runtime `3.61s` remains safe. Confidence still low, so confirm and validate later.
- Run 011: interpolated n=4 with `C=64` scored `2.9798657316`, slightly worse than C=32 but still better than baseline. The optimum is likely near C=32 rather than much larger; try intermediate values such as C=48 or C=24, or rerun C=32 for confirmation.
- Run 012: interpolated n=4 with `C=48` improved slightly to `2.9792935749`, current best; checks passed. Effect size over C=32 is tiny and confidence remains below noise floor, so confirmation/validation is required before trusting this as robust.
- Run 013: interpolated n=4 with `C=40` improved very slightly to `2.9792409029`, current best; checks passed. Difference from C=48 is only about `0.000053` bits/symbol and confidence remains below noise floor, so treat C=40 as provisional until repeated/validated.
- Run 014: interpolated n=4 with `C=36` scored `2.9793094026`, worse than C=40 and close to C=48. The local optimum remains around C=40, but differences are tiny and confirmation is more valuable than excessive public-practice tuning.
- Run 015: interpolated n=4 with `C=44` scored `2.9792403269`, technically current best but only `0.000000576` bits/symbol better than C=40. The n=4 interpolation constant optimum is flat/noise-dominated around C=40-44; stop fine constant tuning until repeated/validated.
- Run 016: interpolated n=5 with `C=44` scored `2.9848362778`, worse than n=4 C=44 while still above baseline. Length-4 contexts hurt even under soft interpolation and add runtime; avoid more n=5/order-4 context experiments unless there is a substantially different mechanism.
- Run 017: interpolated n=4 C=44 with online update weight `2` scored `2.9801459465`, worse than unit online updates. Online adaptation helps versus static hard-backoff, but simple overweighting chases noise; avoid update weights above 1 as a simple multiplier.
- Run 018: interpolated n=4 C=44 with additive smoothing `0.5` scored `2.9773145652`, a real improvement over add-one smoothing. Interpolation makes sharper KT/Jeffreys-style half-count local estimates useful, unlike hard-backoff where Laplace 0.5 was slightly worse.
- Run 019: interpolated n=4 C=44 with additive smoothing `0.25` scored `2.9780370195`, worse than 0.5 but still better than add-one. The smoothing optimum is likely near the KT/Jeffreys half-count rather than much lower; too-sharp local estimates underweight surprises.
- Run 020: interpolated n=4 C=44 with additive smoothing `0.75` scored `2.9779357716`, worse than 0.5 but better than add-one. Together with 0.25, this brackets the local smoothing optimum around 0.5; next retune interpolation `C` around the half-count model rather than more smoothing brackets.
- Run 021: interpolated n=4 with `Laplace=0.5`, `C=56` scored `2.9775503820`, worse than C=44 but close. The half-count model does not benefit from much more conservative high-order weighting; if retuning C, try C=40 or C=36, then prioritize validation/confirmation.
- Run 022: interpolated n=4 with `Laplace=0.5`, `C=40` scored `2.9773386624`, an extremely close miss versus C=44. The half-count interpolation optimum remains around C=44; stop narrow C/smoothing tuning and move to validation or a qualitatively different sequential mixture.
- Run 023: n=4 half-count interpolation with adaptive `C_eff = 4 * T(context)` scored `2.9771815925`, improving slightly over fixed C=44. Continuation diversity is a useful uncertainty signal beyond total count alone, but the gain is small; bracket the scale once or validate before more public-practice tuning.
- Run 024: adaptive unique-count scale `3` scored `2.9774736781`, worse than scale 4. The run 023 model was not too cautious; smaller diversity penalty overtrusts context-specific counts. Try one upper bracket such as scale 5/6, then validate rather than continue fine public-practice tuning.
- Run 025: adaptive unique-count scale `5` scored `2.9774876940`, also worse than scale 4. Scale 4 is bracketed by worse lower and upper settings; stop adaptive scale fine-tuning and move to validation/confirmation or a qualitatively different PPM-style escape variant.
- Run 026: PPM-style adaptive escape with unsmoothed MLE higher-order local distributions scored `2.9832630018`, much worse than smoothed adaptive Run 023. Half-count smoothing inside higher-order local distributions remains important; escape/backoff alone is not robust enough.
- Run 027: equal 50/50 mixture of fixed-C44 and adaptive `C_eff=4*T` half-count predictors scored `2.9771224184`, a small improvement over adaptive alone at the cost of runtime rising to about 5s. Fixed-count and diversity-adaptive schedules have complementary calibration errors; validate/repeat before fine tuning, or at most try a coarse weight such as 25% fixed / 75% adaptive.
- Run 028: 25% fixed / 75% adaptive model mixture scored `2.9771224575`, effectively tied but fractionally worse than the equal mixture. Equal weighting remains best within numerical/noise-level resolution; stop public-practice mixture-weight tuning.
- Run 029: extending the fixed/adaptive half-count model mixture to `n=5` scored `2.9766819653`, improving over n=4. Earlier n=5 failures were due to insufficiently robust confidence/smoothing, not necessarily useless length-4 context information; cautiously bracket with n=6 or validate n=5 vs n=4.
- Run 030: extending the same robust model mixture to `n=6` scored `2.9965394663`, much worse than n=5 and only slightly better than baseline. Length-5 contexts are too sparse/noisy even with robust confidence and half-count smoothing; do not increase order beyond n=5 in this family.
- Run 031: n=5 model mixture with additive smoothing `0.4` scored `2.9754042414`, a sizable improvement over `0.5`. The n=5 robust mixture benefits from sharper local distributions than the n=4 family; bracket smoothing around 0.4 but validate soon.
- Run 032: n=5 model mixture with additive smoothing `0.3` scored `2.9749371262`, improving again. The robust n=5 mixture continues to benefit from sharper local distributions, but lower smoothing raises surprise-symbol risk; try one lower bracket such as 0.2/0.25, then validate.
- Run 033: n=5 model mixture with additive smoothing `0.2` scored `2.9759198844`, worse than `0.3` and `0.4`. The n=5 smoothing optimum is bracketed, and too little smoothing underweights surprises in sparse high-order contexts; prefer adaptive smoothing or validation over more narrow scalar smoothing sweeps.
- Run 034: n=5 diversity-dependent smoothing scored `2.9752131102`, better than global `0.2` but worse than global `0.3`. Continuation diversity alone did not choose smoothing well enough; avoid this formula and prefer either the global `0.3` model or conservative model averaging/validation.
- Run 035: 50/50 smoothing ensemble over global `0.3` and `0.4` scored `2.9750329918`, slightly worse than `0.3` alone and much slower. The conservative component diluted the best model more than it protected against surprises; avoid more smoothing ensembles unless validation motivates them.
- Run 036: order-dependent smoothing (`0.5` for orders 0..3, `0.3` for order 4) scored `2.9755466270`, worse than global `0.3`. The n=5 gain is not isolated to order-4 smoothing; avoid further smoothing tweaks and prefer qualitatively different online adaptation/recency or validation.
- Run 037: recent-window online mixture crashed before evaluation because a local `@dataclass(slots=True)` count class failed under the official importlib loader. The modeling idea was not tested; retry without dataclasses before drawing conclusions about recency.
- Run 038: n=5 `alpha=0.3` model plus a 5% ramped online-only recent-window (`8192` symbols) mixture scored `2.9740425033`, improving meaningfully over the prior best. A small separate recent model captures useful local structure that cumulative train+online counts miss; bracket weight/window coarsely and validate soon.
- Run 039: increasing the recent-window max weight to `0.08` scored `2.9742842890`, worse than `0.05` but still better than pre-recency. Recency signal is useful, but too much recent mass overfits local noise; try lower weight or window-size brackets, not larger weights.
- Run 040: lowering the recent-window max weight to `0.03` scored `2.9741829201`, worse than `0.05` but better than `0.08` and pre-recency. The useful recent weight is bracketed around `0.05`; stop weight tuning and vary window size or validate.
- Run 041: doubling the recent window to `16384` at weight `0.05` scored `2.9739917554`, a tiny improvement over `8192`. A broader recent window slightly reduces variance or better matches local regimes; try one opposite short-window bracket such as `4096`, then validate/repeat rather than fine-tune windows.
- Run 042: shortening the recent window to `4096` scored `2.9742436210`, worse than `8192` and `16384`. The recent component needs more samples than 4096 symbols for reliable n=5 context estimates; prefer `16384`, consider one upper bracket such as `32768`, then validate.
- Run 043: broadening the recent window to `32768` scored `2.9740190050`, worse than `16384` but better than `8192`/`4096`. The window optimum is bracketed around `16384`; stop public-practice window tuning and validate or try a qualitatively different recency schedule/order.
- Run 044: keeping the main model at `n=5` but lowering only the recent-window component to `n=4` scored `2.9737102858`, a meaningful improvement over n=5 recent. The sliding recent table is variance-limited; lower recent order gives more reliable local adaptation while the cumulative main model preserves longer-context signal.
- Run 045: increasing only the n=4 recent component smoothing to `0.5` scored `2.9738223892`, worse than recent smoothing `0.3`. Even with lower recent order, the recent local distributions prefer the sharper `0.3` calibration; keep recent `alpha=0.3`, `window=16384`.
- Run 046: lowering only the recent-window component to `n=3` scored `2.9734028461`, improving over recent `n=4` and running faster. The sliding recent table is strongly variance-limited; medium-short local adaptation is more reliable than sparse long recent contexts.
- Run 047: lowering the recent-window component further to `n=2` scored `2.9739475853`, worse than recent `n=3` despite faster runtime. The useful recent-window signal needs at least two-symbol context; recent-order optimum is bracketed around `n=3`. Keep recent `n=3`, `alpha=0.3`, `window=16384`; if tuning recency further, use coarse weight/window brackets or validation.
- Run 048: increasing the `n=3` recent-window max mixture weight from `0.05` to `0.08` scored `2.9730956697`, improving over Run 046. Once the recent component is lowered to `n=3`, it is reliable enough to carry more mass than sparse `n=5` recent could; coarsely bracket the `n=3` recent weight upward/downward but avoid excessive public-practice tuning without validation.
- Run 049: increasing the `n=3` recent-window max mixture weight further to `0.11` scored `2.9731718557`, worse than `0.08` but still better than `0.05`. The `n=3` recent-weight optimum is bracketed below `0.11`; recent evidence is useful but starts to dilute main-model evidence when too heavy.
- Run 050: setting the `n=3` recent-window max mixture weight to `0.095` scored `2.9730885237`, only `0.000007` bits/symbol better than `0.08`. The weight optimum is very flat around `0.08-0.095` and likely public-practice/noise sensitive; stop narrow recent-weight tuning unless validation motivates it.
- Run 051: shortening the `n=3`, weight `0.095` recent window from `16384` to `8192` scored `2.9737838917`, much worse. Even with lower recent order, the recent component needs the broader `16384`-symbol window; shorter windows overfit local noise. Avoid shorter recent windows.
- Run 052: broadening the `n=3`, weight `0.095` recent window to `32768` scored `2.9726531097`, a sizable improvement over `16384`. For n=3 recent contexts, the recent source is still variance-limited; broader history stabilizes medium-short local probabilities better than shorter windows.
- Run 053: broadening the `n=3`, weight `0.095` recent window to `65536` scored `2.9724090913`, improving another `0.000244` bits/symbol over `32768`. The recent component remains sample-limited at `32768`; broader recent history still adds stable medium-short-context adaptation, though the window is approaching the full evaluated prefix scale.
- Run 054: broadening the `n=3`, weight `0.095` recent window to `131072` scored `2.9722786904`, improving another `0.000130` bits/symbol over `65536`. Broad test-prefix history still helps the online-only low-order component, but marginal gains are shrinking and the window is near the official `200000`-token prefix.
- Run 055: setting the `n=3`, weight `0.095` online-only component window above the official prefix (`262144`) scored `2.9722665972`, only `0.000012` bits/symbol better than `131072`. Broad-window tuning is saturated; the component is effectively a test-prefix-only low-order online model rather than a recency model.
- Run 056: increasing the all-past `n=3` online component max weight to `0.12` scored `2.9720334599`, improving by about `0.000233` bits/symbol over weight `0.095`. Once the auxiliary component uses essentially all past test-prefix symbols, it is stable enough to carry more mixture mass than shorter recency-window variants.
- Run 057: increasing the all-past `n=3` online component max weight to `0.15` scored `2.9719738362`, improving only about `0.000060` bits/symbol over `0.12`. The stable low-order online component can carry more mass, but gains are shrinking; bracket the upper side once then validate/repeat.
- Run 058: increasing the all-past `n=3` online component max weight to `0.18` scored `2.9721402744`, worse than `0.15`. The all-past low-order online component is useful but over-weighted by `0.18`; keep weight `0.15` as the current best and avoid larger n=3 online weights unless validation says otherwise.
- Run 059: switching the all-past auxiliary component from `n=3` to `n=4` at weight `0.15` scored `2.9722374796`, worse than the current best `n=3` all-past component. Even full-prefix online counts do not make auxiliary order 4 reliable enough at this weight; keep auxiliary max order `n=3` unless a much smaller exploratory weight is justified.
- Run 060: lowering the all-past `n=3` online component max weight from `0.15` to `0.14` scored `2.9719679315`, a tiny improvement of about `0.0000059` bits/symbol. The auxiliary-weight optimum is extremely flat near `0.14-0.15`; avoid strong claims without validation/repetition and prefer qualitative changes over more narrow public-practice weight tuning.
- Run 061: keeping all-past `n=3` weight `0.14` but slowing the auxiliary warmup from `10000` to `20000` tokens scored `2.9719344575`, improving about `0.0000335` bits/symbol. Early online-only estimates are noisy enough that delaying full auxiliary weight helps; bracket warmup coarsely before returning to narrow weight tuning.
- Run 062: increasing the same warmup further to `40000` tokens scored `2.9719714025`, worse than `20000`. A too-slow ramp underuses useful low-order online evidence; the warmup optimum is bracketed between `10000` and `40000`, with `20000` currently best.
- Run 063: trying an upper-mid `25000` token warmup scored `2.9719350195`, essentially tied but slightly worse than `20000` by about `0.0000006` bits/symbol. Treat warmup tuning as flat/noise-level around `20000-25000`; keep `20000` and move to qualitative changes such as separate auxiliary smoothing.
- Run 064: separating auxiliary smoothing and lowering only the all-past `n=3` online component to `alpha=0.2` scored `2.9718899997`, improving about `0.0000445` bits/symbol over Run 061. The dense all-past low-order component benefits from sharper smoothing even though global `n=5` `alpha=0.2` was too sharp; bracket auxiliary smoothing while keeping main `alpha=0.3`, weight `0.14`, and warmup `20000`.
- Run 065: lowering only the all-past `n=3` auxiliary smoothing further to `alpha=0.1` scored `2.9718452410`, improving another `0.0000448` bits/symbol over `alpha=0.2`. The dense low-order online component still benefits from sharper smoothing; bracket the lower side once more while watching for rare-symbol surprise penalties.
- Run 066: lowering only the all-past `n=3` auxiliary smoothing to `alpha=0.05` scored `2.9718226807`, improving another `0.0000226` bits/symbol over `alpha=0.1`. Lower-side smoothing gains are shrinking but not yet reversed; bracket cautiously below `0.05` while keeping main `alpha=0.3`, weight `0.14`, and warmup `20000`.
- Run 067: lowering only the all-past `n=3` auxiliary smoothing to `alpha=0.02` scored `2.9718090910`, improving another `0.0000136` bits/symbol over `alpha=0.05`. Gains continue but are shrinking; rare-symbol penalties have not reversed yet, but any further lower-side smoothing should be very cautious and validated.
- Run 068: lowering only the all-past `n=3` auxiliary smoothing to `alpha=0.01` scored `2.9718045659`, improving only about `0.0000045` bits/symbol over `alpha=0.02`. The lower-side smoothing optimum is now flat/noise-level around `0.01-0.02`; stop aggressive lower smoothing and prefer retuning weight/warmup or validating before strong claims.
- Run 069: with auxiliary `alpha=0.01`, increasing the all-past `n=3` max mixture weight from `0.14` to `0.16` scored `2.9717929495`, improving about `0.0000116` bits/symbol. Sharpening the auxiliary model shifted the best hedge upward slightly, but gains are small; bracket the upper weight cautiously and stop if over-weighting returns.
- Run 070: increasing the same sharp all-past `n=3` auxiliary max weight to `0.18` scored `2.9718751228`, worse by about `0.000082` bits/symbol. Weight `0.18` over-weights the low-order online component even after sharpening; keep `0.16` as the best bracketed weight unless a different ramp schedule changes the tradeoff.
- Run 071: keeping auxiliary `alpha=0.01` and max weight `0.16` but slowing warmup from `20000` to `30000` scored `2.9717926651`, a negligible `0.0000003` bits/symbol improvement. Treat warmup `20000-30000` as flat/noise-level; avoid strong claims and prefer qualitatively different schedules or model-mixture calibration over more linear warmup micro-tuning.
- Run 072: quadratic warmup to `30000` scored `2.9718579119`, worse by about `0.000065` bits/symbol. The sharp all-past auxiliary evidence is useful early enough that quadratic delay loses more than it protects; avoid slower-than-linear early ramps for this auxiliary model.
- Run 073: giving the all-past `n=3` auxiliary model its own internal `recent_mix_weight_fixed=0.75` (favoring fixed-C over adaptive-C) scored `2.9717829205`, improving about `0.000010` bits/symbol. Dense low-order online contexts benefit from simpler fixed-C44 calibration more than the sparse main model does.
- Run 074: the opposite bracket `recent_mix_weight_fixed=0.25` scored `2.9718037280`, worse than both `0.75` and the shared `0.5`. Dense low-order contexts clearly prefer fixed-C44; adaptive-C diversity weighting is unnecessary for the auxiliary.
- Run 075: `recent_mix_weight_fixed=0.9` improved to `2.9717777131`, about `0.000005` better than `0.75`. The dense low-order auxiliary strongly prefers fixed-C44.
- Run 076: `recent_mix_weight_fixed=0.95` improved to `2.9717760846`, marginal ~`0.0000016` gain.
- Run 077: pure fixed-C `1.0` for auxiliary improved to `2.9717745101`, about `0.0000016` better than `0.95`. Adaptive-C never helped dense low-order all-past contexts.

Update this section as experiments accumulate. Include dead ends so the agent does not rediscover them.
