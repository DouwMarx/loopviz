# Spec: Procedural Image Generator

Companion to `aesthetic_optimization_spec.md`. That document covers the metrics and optimization loop; this one covers only the generator g(θ) — how an 18-number parameter vector plus a handful of fixed random seeds becomes an image, and why each stage exists.

## 0. Design principle

The generator is a prior, not just a dimension reducer. Its job is to make *every* point in its range look like something coherent, so that when a metric pushes on it, the result stays on the "image-like" manifold instead of drifting into adversarial static (which is what happens when pixels are optimized directly — verified empirically in `compare_param.py`: pixel optimization scored 44.8, this generator 0.61, on the same loss and budget).

Each knob was chosen to give the optimizer a handle on a specific metric family: one knob for the power spectrum, knobs for sparsity/edges, knobs for tone and color, one for composition. The generator can only be steered along directions someone thought to build in — that is both its strength (robustness) and its ceiling (texture, never composition).

## 1. The noise primitive: spectral synthesis

Everything is built from one primitive, a random field with a prescribed power spectrum:

1. Create a grid of frequencies (2-D FFT layout).
2. Assign every frequency an independent uniform-random phase in [0, 2π).
3. Assign amplitude |f|^(−β/2), i.e. power falls as 1/f^β. Zero out the DC term.
4. Inverse FFT, take the real part, min–max normalize to [0, 1].

Why this primitive: the exponent β is a single number that directly controls the metric we care most about (spectral slope), and it interpolates the whole texture family — β ≈ 0 is white noise, β ≈ 2 is the cloud/terrain statistics of natural scenes, β ≈ 3+ is very smooth blobs. Big smooth structures come from the dominant low frequencies; the power-law tail decorates them with proportionally scaled detail at every finer scale. That scale-freeness is what reads as "cloud."

Limitation to be aware of: any *linear* combination of these fields is a Gaussian random field, and for Gaussian fields all higher-order statistics are locked to the spectrum. In particular gradient sparsity cannot be changed by linear means. Stages 2 and 4 below exist specifically to break Gaussianity.

## 2. Pipeline stages (in order)

Let X_β(seed) denote the primitive above. All fields are generated at the render resolution n×n.

**Stage 1 — base field.**
T = (1 − m)·X_β(s₁) + m·X_β(s₂)²
Knobs: β (θ₀ ∈ [1.2, 3.2]), mix m (θ₁ ∈ [0, 1]). The squared term is the first non-Gaussian ingredient: squaring turns extremes of either sign into bright ridges.

**Stage 2 — domain warp (marbling).**
T(x) ← T( (x + a·D(x)) mod n ), where D is a pair of displacement fields X_{β_w}(s₃), X_{β_w}(s₄), centered.
Knobs: warp strength a (θ₂ ∈ [0, 220] px at n = 256, scaled ∝ n), warp smoothness β_w (θ₃ ∈ [2.0, 3.6]).
Implementation constraint that matters: the lookup MUST be bilinear (`scipy.ndimage.map_coordinates`, order 1). Nearest-neighbor sampling aliases the spectrum and silently destroys the measured β (observed in v0: β collapsed from a designed 2.0 to a measured 0.95).

**Stage 3 — ridge / sparsify.**
T ← (1 − s)·T + s·sigmoid(k·(T − 0.55))
Knobs: amount s (θ₅ ∈ [0, 1]), sharpness k (θ₆ ∈ [2, 12]).
Purpose: manufacture "smooth ground + occasional crisp boundary," the heavy-tailed gradient distribution that the Gini/sparsity metric wants and that stages 1–2 cannot produce. Second non-Gaussian ingredient. Known weakness: in practice this lever is not strong enough to reach Gini ≈ 0.75 while β stays near 2 — the documented β–D–Gini coupling.

**Stage 4 — tone.**
T ← T^γ, knob γ (θ₄ ∈ [0.4, 1.8]). Standard gamma: <1 lifts midtones, >1 deepens them.

**Stage 5 — color (cosine palette).**
RGB(x) = a + b ⊙ cos(2π·(c·T(x) + d)), then clip to [0, 1].
Knobs: offsets a ∈ [0.2, 0.8]³ (θ₇₋₉), amplitudes b ∈ [0.05, 0.55]³ (θ₁₀₋₁₂), shared frequency c ∈ [0.4, 1.6] (θ₁₃), phases d ∈ [0, 1]³ (θ₁₄₋₁₆).
Why cosine palettes: the three channels are smooth coordinated waves of one underlying value, so the palette is automatically a continuous, band-limited path through color space — harmonious gradients by construction, never per-pixel hue noise. Ten numbers constitute the entire color system.
Grayscale mode: the output is collapsed to luminance (L = 0.2126R + 0.7152G + 0.0722B, replicated to 3 channels) after this stage; the palette then degenerates into a free nonmonotonic tone curve, which is retained expressivity, not waste.

**Stage 6 — vignette.**
RGB ← RGB · (1 − v·r²), r = normalized distance from center, knob v (θ₁₇ ∈ [0, 0.6]). A crude composition device: darkening corners pulls the eye inward. The generator's only global-layout control.

## 3. Initialization and randomness model

There are two entirely separate sources of variation, and keeping them separate is the point:

**θ (18 numbers) — the controlled part.** Parameterizes the *ensemble*: statistics, tone, palette. This is what optimization moves and what a "random draw" samples (uniform over the box ranges above).

**Seeds (5 integers) — the frozen part.** Each noise role has a fixed seed (base = 101, second = 202, warp-x = 303, warp-y = 404; plus grain seeds where used). Consequences:
- g(θ) is deterministic: same θ → same image, always.
- Structure is stable across resolutions: optimizing at 256² and rendering at 768² shows the *same* cloud, larger — necessary for the low-res-optimize / high-res-render workflow. (Feature *values* still drift with resolution; that is a property of the metrics, handled by two-scale measurement, not of the generator.)
- Resampling seeds at fixed θ yields a visually different image with nearly the same feature vector: θ picks the ensemble, the seed picks the member. The images' apparent richness mostly comes from seed entropy; θ only shapes it.

Parameter-space geometry: optimization does not act on θ directly but on z ∈ ℝ⁸ via θ = range_map(sigmoid(A·z)), A a fixed random 18×8 Gaussian matrix, so the search lives on a smooth 8-D slice of the 18-D box. The sigmoid enforces box constraints smoothly; z = 0 initializes at the box center. This low-rank restriction is optional (rank 18 with A = I recovers the full space) and is known to cap the reachable feature set at ≤ 8 dimensions against a 10-dimensional objective.

Caching: noise fields are memoized on (β rounded to 2 decimals, n, seed), which makes ES loops cheap since candidates mostly re-vary non-spectral knobs.

## 4. Knob-to-metric map (what each handle is for)

| knobs | stage | primarily controls |
|---|---|---|
| θ₀ β | 1 | spectral slope; indirectly fractal D, symmetry |
| θ₁ mix | 1 | non-Gaussianity, entropy |
| θ₂ θ₃ warp | 2 | texture character (marbling); mild effect on D |
| θ₅ θ₆ ridge | 3 | gradient sparsity, edge density (weakly — known bottleneck) |
| θ₄ γ | 4 | luminance histogram → entropy, contrast |
| θ₇₋₁₆ palette | 5 | colorfulness, hue dispersion, saturation, contrast |
| θ₁₇ vignette | 6 | low-frequency balance; (accidentally) symmetry |

## 5. Known limitations

1. No compositional capacity: no objects, figure/ground, horizon, or layout beyond the vignette. Everything produced is texture.
2. β–D–Gini entanglement through the single spectral backbone; the ridge stage is too weak to decouple them. Planned fix: an independent oriented-ridge (curvelet-like) layer with its own density knob.
3. Symmetry is an epiphenomenon of low-frequency dominance, not an independent lever.
4. Apparent image diversity is largely seed entropy, not θ; galleries of "different" outputs at similar θ are members of one ensemble.

## 6. Reference

Implementation: `optimize.py` (functions `spectral_noise`, `gen`, `RANGES`); grayscale collapse in `optimize_bw.py`; the pixels / W^TW / procedural comparison in `compare_param.py`.
