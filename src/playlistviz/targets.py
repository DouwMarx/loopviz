"""2D procedural target-image generator (grayscale).

The "cloud" pipeline from procedural_generator_spec.md (spectral synthesis
-> domain warp -> ridge -> gamma -> vignette) extended with two one-shot,
ES-friendly layers from the procedural-art literature review:

- Level-set figure/ground: threshold a low-frequency composition field into
  a feathered mask and give figure and ground different tonal treatments -
  the composition lever the pure spectral generator lacks.
- Curl-noise LIC flow: smears the field along a divergence-free flow,
  adding coherent directional structure (edge-orientation control).

A Worley crack-network family existed briefly and was removed: it is a
categorically different visual process, and category membership is largely
invisible to the metric feature map, so the BT loop cannot reliably vote it
out - simpler to not generate it at all.

Role here: targets are optimized directly against the metric loss (fast, no
operator in the loop) and then embedded into the playlist operator's free
part via embed.py. Deterministic: fixed seeds per role; theta picks the
ensemble.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

RANGES_2D: dict[str, tuple[float, float]] = {
    # spectral backbone
    "beta": (1.2, 3.2),        # spectral slope of the base field
    "mix": (0.0, 1.0),         # squared-field mix (non-Gaussianity)
    "warp_amp": (0.0, 0.86),   # domain warp strength, fraction of n
    "warp_beta": (2.0, 3.6),   # warp field smoothness
    "ridge_amount": (0.0, 1.0),
    "ridge_sharp": (2.0, 12.0),
    # figure/ground composition
    "fig_mix": (0.0, 1.0),         # strength of figure/ground separation
    "fig_thresh": (0.25, 0.75),    # level-set threshold (figure area)
    "fig_feather": (0.005, 0.12),  # boundary softness, fraction of n
    # flow smear
    "lic_mix": (0.0, 1.0),         # blend of flow-smeared field (0 disables)
    "lic_len": (0.01, 0.06),       # streamline length, fraction of n
    # tone
    "gamma": (0.4, 1.8),
    "vignette": (0.0, 0.6),
}

PARAM_NAMES_2D = list(RANGES_2D)
N_PARAMS_2D = len(PARAM_NAMES_2D)

_SEEDS = {"base": 101, "second": 202, "warp_x": 303, "warp_y": 404,
          "comp": 606, "flow": 707}


def theta2d_to_params(theta: np.ndarray) -> dict[str, float]:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (N_PARAMS_2D,):
        raise ValueError(f"theta must have shape ({N_PARAMS_2D},), got {theta.shape}")
    unit = 1.0 / (1.0 + np.exp(-theta))
    return {name: float(lo + (hi - lo) * u)
            for u, (name, (lo, hi)) in zip(unit, RANGES_2D.items())}


def spectral_noise_2d(n: int, beta: float, seed: int) -> np.ndarray:
    """1/f^beta random field, min-max normalized to [0, 1]."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.hypot(fy, fx)
    with np.errstate(divide="ignore"):
        amp = np.where(f > 0, f ** (-beta / 2.0), 0.0)
    field = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    lo, hi = field.min(), field.max()
    return (field - lo) / (hi - lo) if hi > lo else np.zeros_like(field)


def _flow_smear(T: np.ndarray, length: int) -> np.ndarray:
    """Line-integral-convolution-style smear along a curl-noise flow."""
    n = T.shape[0]
    psi = spectral_noise_2d(n, 3.0, _SEEDS["flow"])
    gy, gx = np.gradient(psi)
    vx, vy = gy, -gx  # curl: divergence-free
    mag = np.hypot(vx, vy) + 1e-12
    vx, vy = vx / mag, vy / mag

    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    acc = T.copy()
    count = np.ones_like(T)
    for sgn in (1.0, -1.0):
        py, px = yy.copy(), xx.copy()
        for _ in range(length):
            dvy = map_coordinates(vy, [py % n, px % n], order=1, mode="wrap")
            dvx = map_coordinates(vx, [py % n, px % n], order=1, mode="wrap")
            py += sgn * dvy
            px += sgn * dvx
            acc += map_coordinates(T, [py % n, px % n], order=1, mode="wrap")
            count += 1
    return acc / count


def generate_target(params: dict[str, float], n: int) -> np.ndarray:
    """theta -> (n, n) grayscale image in [0, 1]."""
    base = spectral_noise_2d(n, params["beta"], _SEEDS["base"])
    second = spectral_noise_2d(n, params["beta"], _SEEDS["second"])
    T = (1 - params["mix"]) * base + params["mix"] * second**2

    a = params["warp_amp"] * n
    if a > 0:
        dx = spectral_noise_2d(n, params["warp_beta"], _SEEDS["warp_x"]) - 0.5
        dy = spectral_noise_2d(n, params["warp_beta"], _SEEDS["warp_y"]) - 0.5
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        # bilinear lookup: nearest-neighbor would alias the spectrum
        T = map_coordinates(T, [(yy + a * dy) % n, (xx + a * dx) % n],
                            order=1, mode="wrap")

    lm = params["lic_mix"]
    if lm > 0.01:
        T = (1 - lm) * T + lm * _flow_smear(T, max(int(params["lic_len"] * n), 2))

    s = params["ridge_amount"]
    if s > 0:
        T = (1 - s) * T + s / (1 + np.exp(-params["ridge_sharp"] * (T - 0.55)))

    fm = params["fig_mix"]
    if fm > 0.01:
        comp = spectral_noise_2d(n, 3.2, _SEEDS["comp"])
        mask = comp > np.quantile(comp, params["fig_thresh"])
        feather = max(params["fig_feather"] * n, 1.0)
        dist = (distance_transform_edt(mask) - distance_transform_edt(~mask))
        soft = 1.0 / (1.0 + np.exp(-dist / feather))
        # figure lifted, ground dimmed: level-set figure/ground separation
        T = T * (1 - fm * 0.6) + fm * (0.65 * soft * T + 0.15 * soft)

    lo, hi = T.min(), T.max()
    if hi > lo:
        T = (T - lo) / (hi - lo)
    T = T ** params["gamma"]

    v = params["vignette"]
    if v > 0:
        yy, xx = np.mgrid[0:n, 0:n]
        r2 = (((yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2)
              / (2 * ((n - 1) / 2) ** 2))
        T = T * (1 - v * r2)
    return np.clip(T, 0.0, 1.0)
