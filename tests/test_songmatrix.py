"""Tests for the pixel-exact song operator (songmatrix) and display
modes (matviz)."""

import numpy as np
import pytest

from loopviz import matviz
from loopviz.sheet import make_sheet
from loopviz.songmatrix import (Plan, build, full_rank_side, materialize,
                                    plan)


def synth_signal(T=4.0, sr=4000, silent=False):
    t = np.arange(int(T * sr)) / sr
    x = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 331 * t)
         + 0.2 * np.sin(2 * np.pi * 97 * t + 1.0))
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t) ** 2  # slow envelope
    if silent:
        x[int(0.4 * x.size):int(0.55 * x.size)] = 0.0  # digital silence
    return x


class TestPlan:
    def test_consistency_from_n(self):
        pl = plan(100.0, n=200, rho=0.5)
        assert pl.N == 100
        assert pl.samples == 200 * 100
        assert pl.f == pytest.approx(pl.samples / 100.0)
        assert pl.rho == pytest.approx(0.5)

    def test_consistency_from_f(self):
        pl = plan(125.0, f=8000, rho=1.0)
        assert pl.n == round(np.sqrt(8000 * 125.0))
        assert pl.N == pl.n
        # implied f is re-derived from integer N, n - close to requested
        assert pl.f == pytest.approx(8000, rel=0.01)

    def test_exactly_one_of_n_f(self):
        with pytest.raises(ValueError):
            plan(100.0)
        with pytest.raises(ValueError):
            plan(100.0, n=100, f=8000)

    def test_full_rank_side(self):
        assert full_rank_side(125.5, 8000) == int(np.ceil(np.sqrt(1004000)))
        # capacity grows with sqrt of duration (up to ceil rounding)
        assert abs(full_rank_side(400, 8000)
                   - 2 * full_rank_side(100, 8000)) <= 1

    def test_print_side(self):
        assert Plan(n=1000, N=500, f=1.0, T=1.0).print_side_mm(0.5) == 500.0


class TestBuild:
    def test_playback_and_shape(self):
        x = synth_signal()
        pl = plan(4.0, n=64, rho=0.5)
        op, W = build(x, pl)
        assert W.shape == (64, 32)
        assert op.playback_error() < 1e-6  # dither floor is -70 dB
        A0 = materialize(op)
        assert A0.shape == (64, 64)
        # A0 advances window k to window k+1 (cyclically)
        assert np.allclose(A0 @ W[:, 0], W[:, 1], atol=1e-8)
        assert np.allclose(A0 @ W[:, -1], W[:, 0], atol=1e-8)

    def test_dither_rescues_silence(self):
        x = synth_signal(silent=True)
        pl = plan(4.0, n=64, rho=0.5)
        op_raw, _ = build(x, pl, dither_db=None)
        op_dith, _ = build(x, pl)  # default -70 dB
        # silent windows wreck the raw Gram; dither restores the operator
        assert op_dith.gram_condition() < 1e-3 * op_raw.gram_condition()
        assert op_dith.playback_error() < 1e-6

    def test_dither_is_inaudible(self):
        x = synth_signal()
        pl = plan(4.0, n=64, rho=0.5)
        _, W_raw = build(x, pl, dither_db=None)
        _, W_dith = build(x, pl)
        assert np.abs(W_raw - W_dith).max() < 1e-3


class TestMatviz:
    def test_gray_range_and_midpoint(self):
        A = np.random.default_rng(0).standard_normal((32, 32))
        g = matviz.gray(A)
        assert g.shape == (32, 32)
        assert g.min() >= 0 and g.max() <= 1
        assert matviz.gray(np.zeros((8, 8)))[0, 0] == pytest.approx(0.5)

    def test_diverging_rgb(self):
        A = np.random.default_rng(1).standard_normal((16, 16))
        rgb = matviz.diverging(A)
        assert rgb.shape == (16, 16, 3)
        assert rgb.min() >= 0 and rgb.max() <= 1

    def test_hinton_full_matrix_raster(self, tmp_path):
        from PIL import Image
        A = np.random.default_rng(2).standard_normal((24, 24))
        out = matviz.hinton(A, tmp_path / "h.png", cell=6)
        im = Image.open(out)
        assert im.size == (24 * 6, 24 * 6)      # every entry gets a cell
        vals = set(np.asarray(im).ravel().tolist())
        assert vals <= {0, 128, 255}            # black / gray bg / white

    def test_bubble_full_matrix_raster(self, tmp_path):
        from PIL import Image
        A = np.random.default_rng(3).standard_normal((16, 16))
        out = matviz.bubble(A, tmp_path / "b.png", cell=5)
        assert Image.open(out).size == (16 * 5, 16 * 5)

    def test_wireframe_writes_file(self, tmp_path):
        A = np.random.default_rng(4).standard_normal((24, 24))
        out = matviz.wireframe(A, tmp_path / "w.png", dpi=40)
        assert out.exists() and out.stat().st_size > 0

    def test_sheet_accepts_rgb_tiles(self, tmp_path):
        gray_tile = np.random.default_rng(3).random((20, 20))
        rgb_tile = np.random.default_rng(4).random((20, 20, 3))
        make_sheet([("gray", gray_tile), ("rgb", rgb_tile)],
                   tmp_path / "sheet.png", tile_size=40)
        assert (tmp_path / "sheet.png").exists()


class TestErrorBudget:
    def _svd(self):
        from loopviz.songmatrix import build, decompose, plan
        x = synth_signal()
        pl = plan(4.0, n=64, rho=0.5)
        op, W = build(x, pl)
        return decompose(op, W), op

    def test_decompose_reconstructs_exactly(self):
        from loopviz.songmatrix import materialize, reconstruct
        svd, op = self._svd()
        A0 = materialize(op)
        assert np.allclose(reconstruct(svd, np.zeros(len(svd.s))), A0,
                           atol=1e-10)

    def test_attenuation_error_formula_matches_direct(self):
        from loopviz.songmatrix import attenuation_errors, reconstruct
        svd, _ = self._svd()
        rng = np.random.default_rng(5)
        a = rng.random(len(svd.s)) * 0.5
        A = reconstruct(svd, a)
        WS = np.roll(svd.W, -1, axis=1)          # exact target: A w_k = w_{k+1}
        direct = np.linalg.norm(A @ svd.W - WS, axis=0)
        assert np.allclose(attenuation_errors(svd, a), direct, atol=1e-9)

    def test_project_feasible_respects_tol(self):
        from loopviz.songmatrix import attenuation_errors, project_feasible
        svd, _ = self._svd()
        a = np.ones(len(svd.s))                  # drop everything: way infeasible
        a_ok, err = project_feasible(svd, a, tol=1e-6)
        assert err <= 1e-6 * (1 + 1e-12)
        assert attenuation_errors(svd, a_ok).max() <= 1e-6 * (1 + 1e-12)

    def test_loop_degradation_exact_and_unstable(self):
        from loopviz.songmatrix import loop_degradation, materialize
        svd, op = self._svd()
        errs = loop_degradation(materialize(op), svd.W, loops=2)
        assert errs[0] < 1e-4 and errs[1] < 1e-4  # dither-level drift
        blowup = loop_degradation(2.0 * np.eye(64), svd.W, loops=2)
        assert blowup == [float("inf")] * 2
