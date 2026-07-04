"""End-to-end: synthetic songs -> operator -> optimized aesthetic render
-> candidate files -> comparison server state -> BT fit. No network needed."""

import json

import numpy as np
import pytest

from playlistviz import bt
from playlistviz.compare_server import CompareState
from playlistviz.config import AudioConfig, OptConfig, RenderConfig, ZConfig
from playlistviz.ingest import build_song_matrix, load_matrix, save_matrix
from playlistviz.loss import equal_weights, loss_vector, scalar_loss
from playlistviz.metrics import features
from playlistviz.operator import PlaylistOperator
from playlistviz.optimize import make_objective, run_es
from playlistviz.render import save_png
from playlistviz.zspace import (N_PARAMS, generate_Z, song_envelopes,
                                theta_to_params)


def synthetic_songs(rng, D=4000, N=5):
    """Songs = mixtures of decaying sines, unit norm - stand-ins for audio."""
    t = np.arange(D)
    X = np.zeros((D, N))
    for j in range(N):
        for _ in range(6):
            f = rng.uniform(0.001, 0.4)
            X[:, j] += rng.uniform(0.3, 1.0) * np.sin(
                2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)
            ) * np.exp(-t / rng.uniform(D / 4, D))
        X[:, j] /= np.linalg.norm(X[:, j])
    return X


@pytest.fixture(scope="module")
def op():
    return PlaylistOperator.from_songs(synthetic_songs(np.random.default_rng(0)))


def test_matrix_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = synthetic_songs(rng)
    cfg = AudioConfig()
    save_matrix(X, ["a", "b", "c", "d", "e"], tmp_path / "songs.npz", cfg)
    X2, meta = load_matrix(tmp_path / "songs.npz")
    assert np.allclose(X, X2, atol=1e-6)  # float32 storage
    assert meta["titles"] == ["a", "b", "c", "d", "e"]


def test_objective_runs_and_preserves_playback(op):
    w = equal_weights()
    objective = make_objective(op, w, ZConfig(rank=4), resolution=64)
    res = objective(np.random.default_rng(2).standard_normal(N_PARAMS))
    assert np.isfinite(res.loss)
    assert res.image.shape == (64, 64, 3)


def test_short_es_run_improves_or_holds(op):
    w = equal_weights()
    objective = make_objective(op, w, ZConfig(rank=4), resolution=64)
    cfg = OptConfig(generations=3, population=4, seed=0)
    best, hist = run_es(objective, cfg)
    assert hist.best_loss[-1] <= hist.best_loss[0]
    assert np.isfinite(best.loss)


def test_candidates_to_bt_fit(tmp_path, op):
    """Write two candidates the way the CLI does, then drive the compare
    state machine and fit."""
    runs = tmp_path / "runs"
    rng = np.random.default_rng(3)
    env = song_envelopes(op.X)
    for i in range(3):
        theta = rng.standard_normal(N_PARAMS)
        params = theta_to_params(theta)
        zf = generate_Z(params, op.D, ZConfig(rank=4), envelopes=env,
                        project_perp=op.project_perp)
        L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
        from playlistviz.render import render
        img = render(L, R, 128, params)
        phi = features(img)
        cand_id = f"cand{i}"
        d = runs / cand_id
        d.mkdir(parents=True)
        save_png(img, d / "presentation.png")
        (d / "candidate.json").write_text(json.dumps({
            "id": cand_id,
            "alpha": None,
            "weights": list(map(float, equal_weights())),
            "theta": list(map(float, theta)),
            "phi": {},
            "loss_vector": list(map(float, loss_vector(phi))),
            "loss_eq": float(scalar_loss(phi, equal_weights())),
        }))

    comps_path = tmp_path / "comparisons.jsonl"
    state = CompareState(runs, comps_path)
    assert len(state.loss_vectors) == 3

    # exhaust all 3 pairs through the state machine
    seen = set()
    for _ in range(3):
        pair = state.next_pair()
        assert pair is not None
        state.record(winner=pair[0], loser=pair[1])
        seen.add(frozenset(pair))
    assert len(seen) == 3
    assert state.next_pair() is None  # all pairs consumed

    fit = bt.fit_bt(bt.load_comparisons(comps_path), state.loss_vectors)
    assert fit.n_comparisons == 3
    assert np.isfinite(fit.w_raw).all()
