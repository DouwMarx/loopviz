import numpy as np
import pytest

from loopviz.operator import PlaylistOperator


@pytest.fixture
def op():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8))
    X /= np.linalg.norm(X, axis=0)
    return PlaylistOperator.from_songs(X)


def test_cyclic_advance_exact(op):
    # A0 x_n = x_{n+1}, including the wraparound A0 x_N = x_1
    assert op.playback_error() < 1e-10


def test_full_cycle_is_identity_on_span(op):
    # A^N = I restricted to span(X)
    v = op.X @ np.arange(1, op.N + 1, dtype=float)  # arbitrary mixture
    out = v.copy()
    for _ in range(op.N):
        out = op.apply_A0(out)
    assert np.allclose(out, v, atol=1e-8)


def test_perp_projector_annihilates_songs(op):
    P = op.project_perp(op.X)
    assert np.abs(P).max() < 1e-10


def test_perp_projector_idempotent(op):
    rng = np.random.default_rng(1)
    M = rng.standard_normal((op.D, 3))
    P1 = op.project_perp(M)
    P2 = op.project_perp(P1)
    assert np.allclose(P1, P2, atol=1e-10)


def test_z_part_invisible_to_playback(op):
    rng = np.random.default_rng(2)
    U = rng.standard_normal((op.D, 4))
    Vp = op.project_perp(rng.standard_normal((op.D, 4)))
    # even a huge Z leaves the playlist exact
    assert op.playback_error(U=U, Vp=Vp, scale=100.0) < 1e-6


def test_z_part_changes_action_off_span(op):
    rng = np.random.default_rng(3)
    U = rng.standard_normal((op.D, 4))
    Vp = op.project_perp(rng.standard_normal((op.D, 4)))
    v = rng.standard_normal(op.D)  # generic vector, not in span(X)
    a = op.apply(v)
    b = op.apply(v, U=U, Vp=Vp, scale=1.0)
    assert not np.allclose(a, b)


def test_factors_reconstruct_operator(op):
    rng = np.random.default_rng(4)
    U = rng.standard_normal((op.D, 3))
    Vp = op.project_perp(rng.standard_normal((op.D, 3)))
    L, R = op.factors(U=U, Vp=Vp, scale=0.5)
    A = L @ R.T
    v = rng.standard_normal(op.D)
    assert np.allclose(A @ v, op.apply(v, U=U, Vp=Vp, scale=0.5), atol=1e-8)


def test_near_duplicate_songs_survive_pinv():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((300, 5))
    X[:, 4] = X[:, 3] + 1e-14 * rng.standard_normal(300)  # near-duplicate
    op = PlaylistOperator.from_songs(X)
    assert np.isfinite(op.Ginv).all()


def test_rejects_more_songs_than_dims():
    with pytest.raises(ValueError):
        PlaylistOperator.from_songs(np.ones((3, 5)))
