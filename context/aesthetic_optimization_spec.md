# Spec: Metric-Guided Aesthetic Image Optimization

## 1. Goal

Generate abstract images that score well on quantitative proxies for human aesthetic preference, treat the weighting of those proxies as an explicit unknown, and ultimately replace population-level targets with weights fitted to an individual observer via pairwise comparisons.

The problem is posed as: find generator parameters θ minimizing a scalarized loss

  L_w(θ) = Σᵢ wᵢ · ((φᵢ(g(θ)) − tᵢ) / sᵢ)² + barrier(φ(g(θ)))

where g is a parametric image generator, φ is a feature map of aesthetic metrics, t are literature-derived targets, s are normalization scales, and w is a weight vector on the probability simplex.

## 2. Generator g(θ), θ ∈ ℝ¹⁸

Procedural, differentiable-free, seeded (same θ reproduces the same image; noise seeds fixed per role so structure is stable across render resolutions).

- Base field: spectral synthesis noise with power spectrum ∝ 1/f^β_gen (θ₀), mixed with a squared second field (θ₁).
- Domain warp: bilinear `map_coordinates` displacement by two smooth noise fields; strength θ₂, smoothness θ₃. (Bilinear sampling matters: nearest-neighbor octave sampling in v0 aliased the spectrum and collapsed measured β to ~1.)
- Sparsify: sigmoid ridge mixed in (amount θ₅, sharpness θ₆) — intended, imperfect lever for gradient sparsity.
- Tone gamma θ₄; vignette θ₁₇.
- Color: cosine palette a + b·cos(2π(c·t + d)) — 10 params (θ₇–θ₁₆). In grayscale runs the output is collapsed to luminance before measurement and the palette degenerates to a free nonmonotonic tone curve.

All parameters live in fixed ranges via a sigmoid map (see §5).

## 3. Feature map φ: image → ℝ¹⁰

Luminance L = 0.2126R + 0.7152G + 0.0722B. Seven metrics are luminance-only; three use chromaticity.

| # | metric | definition (sketch) | target t | scale s | rationale |
|---|--------|--------------------|----------|---------|-----------|
| 0 | β spectral slope | fit of log radial power vs log f, mid-band, Hann-windowed | 2.0 | 0.4 | natural-scene statistics; preference near 1/f² power |
| 1 | fractal D | box counting on thresholded gradient-edge map | 1.4 | 0.15 | preferred fractal dimension ≈ 1.3–1.5 (Taylor/Spehar) |
| 2 | colorfulness | Hasler–Süsstrunk M on opponent channels R−G, ½(R+G)−B | 55 | 20 | moderate-high colorfulness preferred |
| 3 | hue dispersion | saturation-weighted circular std of hue | 0.4 | 0.3 | harmonic/analogous palettes over uniform hue scatter |
| 4 | entropy | Shannon entropy of 256-bin luminance histogram (bits) | 5.0 | 1.2 | Berlyne inverted-U: mid complexity |
| 5 | edge density | fraction of pixels with |∇L| > 0.04 (absolute threshold) | 0.08 | 0.05 | visual clutter penalty |
| 6 | gradient Gini | Gini coefficient of |∇L| distribution | 0.75 | 0.12 | sparse coding / processing fluency; natural images are gradient-sparse |
| 7 | symmetry | max of LR/UD mirror correlation of L | 0.30 | 0.30 | mild symmetry preferred over none or perfect |
| 8 | RMS contrast | std of L | 0.20 | 0.07 | anti-washout |
| 9 | mean saturation | mean HSV S | 0.45 | 0.18 | moderate saturation |

Color-degenerate note: on grayscale input, metrics 2 and 9 go to 0 and metric 3 collapses to 0 as a code artifact (hue defaults to 0 where saturation vanishes). Grayscale runs therefore restrict the loss to the 7 structural metrics {0,1,4,5,6,7,8} and renormalize w.

Measurement is two-scale — φ averaged over the render and its 2× downsample — to penalize scale-fragile solutions. Known residual issue: colorfulness, entropy, and edge density still transfer poorly from optimization resolution (256²) to presentation resolution (768²); β and D transfer well.

Targets are population-level values from the empirical-aesthetics literature; individual variance around them is large, which motivates §7.

## 4. Barriers (weight-independent constraints)

Added after observing reward hacking (a near-flat image gaming the fractal-D box counter): quadratic penalties for RMS contrast < 0.10, entropy < 3.0 bits, and (color runs only) colorfulness > 95. These apply regardless of w.

## 5. Optimization: low-rank evolution strategy

- Parameterization: θ = range_map(σ(A z)), with A ∈ ℝ¹⁸ˣ⁸ random Gaussian fixed per run; search happens in the rank-8 subspace z ∈ ℝ⁸ (regularizes and speeds convergence; "low-rank" option per spec discussion).
- Algorithm: elitist (1+12)-ES, 14 generations, step size σ annealed 1.0 → ~0.06 (×0.82/gen). ~170 evaluations per weighting, ~8–14 s at 256² on CPU.
- Winners re-rendered at 768² and re-measured; reported features are always full-resolution.

## 6. Weight sampling protocol

w ~ Dirichlet(α·1), annealing α to grow variance of the weighting:
- α → ∞: equal-weight baseline (1 run)
- α = 8: mild variance (2 runs)
- α = 2: moderate (3 runs)
- α = 0.5: near-corner / near-single-metric (3 runs)

Because scalar losses under different w are incomparable, every winner is re-scored under the common equal-weight loss L_eq (L_eq7 for grayscale) as the shared yardstick. Interpretation: random scalarization samples points on the (convex reachable part of the) Pareto front of the metrics; the equal-weight point has no privileged perceptual status.

## 7. Personalization via pairwise comparisons (planned)

Population targets are means; the object of interest is an individual's w. Protocol:

1. Present the observer pairs of generated images (drawn from runs with diverse w and hence diverse feature vectors).
2. Model choice probability with Bradley–Terry: P(i ≻ j) = σ(u_i − u_j), with utility linear in per-metric losses, u = −wᵀ·loss_vec(φ). (Equivalently logistic regression on loss-vector differences.)
3. Fit w by maximum likelihood (10 parameters; a few dozen comparisons suffice; L2-regularize toward uniform).
4. Re-run the ES under the fitted w; optionally iterate (active sampling: choose next pairs to maximally reduce posterior uncertainty in w).

This replaces the Dirichlet lottery with an estimated observer model and simultaneously validates the metric set: metrics receiving near-zero fitted weight are perceptually irrelevant for this observer; systematic misfit of Bradley–Terry indicates the feature map is missing dimensions.

## 8. Empirical findings so far

1. Reward hacking is real at this scale: under lopsided w, unweighted metrics get sacrificed in perceptually salient ways (flat washed-out images satisfying fractal D). Barriers in §4 are the mitigation.
2. Equal-weight baselines underperformed lopsided draws under the common yardstick in both the color sweep (worst of 9) and the grayscale ablation (3rd of 4). Working hypothesis: with a short ES budget, pressure on many objectives selects fragile finely-balanced optima that transfer poorly across resolution, while lopsided weights land in coarser, sturdier basins. Needs replication over seeds before being trusted.
3. Generator couplings bind: β, fractal D, and gradient Gini are entangled through the single spectral backbone (D stuck ≈ 1.5–1.7 whenever β ≥ 2; Gini stuck ≈ 0.55–0.65 vs target 0.75). Symmetry is an epiphenomenon of low-frequency dominance, not an independent lever. Removing the color metrics made the residual worse, not better — evidence the bottleneck is generator expressivity, not the loss.
4. Metric hygiene matters: relative edge thresholds over-count grain in smooth images (fixed to absolute); non-band-limited warping corrupts β (fixed to bilinear).

## 9. Next steps (in rough priority order)

- Add an independent fine-scale sparsity mechanism to the generator (e.g., oriented ridge/curvelet layer with its own density parameter) to decouple D and Gini from β.
- Run the Bradley–Terry loop (§7) with the existing image pool as the seed set.
- Replicate finding #2 across seeds; if robust, prefer corner-ish scalarizations for exploration.
- Optional: replace hand-set scales s with per-metric losses calibrated so one unit ≈ one just-noticeable difference.

## 10. Artifacts

- `generate.py`, `metrics.py` — v1 images and metric suite
- `optimize.py` — generator, feature map, targets, low-rank ES (color, random w)
- `optimize_v2.py` — Dirichlet annealing, barriers, two-scale φ, common yardstick
- `optimize_bw.py` — grayscale ablation (7 structural metrics)
- `metrics.csv`, `opt_results.csv`, `v2_results.csv`, `bw_results.csv` — feature vectors per image
