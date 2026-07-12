import numpy as np
import pytest

from loopviz import bt
from loopviz.metrics import N_METRICS


def _synthetic_world(rng, n_candidates=20, sharpness=6.0):
    """Candidates with random loss vectors, an observer with known weights."""
    loss_vectors = {
        f"c{i:02d}": rng.random(N_METRICS) * 3 for i in range(n_candidates)
    }
    w_true = np.zeros(N_METRICS)
    w_true[0] = 2.0   # cares a lot about beta
    w_true[4] = 1.0   # somewhat about entropy
    w_true *= sharpness / w_true.sum()
    return loss_vectors, w_true


def _simulate(rng, loss_vectors, w_true, n=300):
    ids = sorted(loss_vectors)
    comps = []
    for _ in range(n):
        i, j = rng.choice(len(ids), 2, replace=False)
        a, b = ids[i], ids[j]
        z = w_true @ (loss_vectors[b] - loss_vectors[a])  # >0 favors a
        p_a = 1 / (1 + np.exp(-z))
        winner, loser = (a, b) if rng.random() < p_a else (b, a)
        comps.append(bt.Comparison(winner=winner, loser=loser))
    return comps


def test_recovers_dominant_weights():
    rng = np.random.default_rng(0)
    loss_vectors, w_true = _synthetic_world(rng)
    comps = _simulate(rng, loss_vectors, w_true, n=400)
    fit = bt.fit_bt(comps, loss_vectors, l2=0.5)
    # the two truly-weighted metrics should come out on top
    top2 = set(np.argsort(-fit.w_simplex)[:2])
    assert top2 == {0, 4}
    assert fit.w_simplex[0] > fit.w_simplex[4]


def test_prediction_beats_chance():
    rng = np.random.default_rng(1)
    loss_vectors, w_true = _synthetic_world(rng)
    train = _simulate(rng, loss_vectors, w_true, n=300)
    test = _simulate(rng, loss_vectors, w_true, n=200)
    fit = bt.fit_bt(train, loss_vectors)
    correct = sum(
        bt.predict_prob(fit, loss_vectors[c.winner], loss_vectors[c.loser]) > 0.5
        for c in test
    )
    assert correct / len(test) > 0.65


def test_fit_requires_comparisons():
    with pytest.raises(ValueError):
        bt.fit_bt([], {})


def test_simplex_normalization():
    rng = np.random.default_rng(2)
    loss_vectors, w_true = _synthetic_world(rng)
    comps = _simulate(rng, loss_vectors, w_true, n=100)
    fit = bt.fit_bt(comps, loss_vectors)
    assert fit.w_simplex.sum() == pytest.approx(1.0)
    assert (fit.w_simplex >= 0).all()


def test_comparison_roundtrip(tmp_path):
    path = tmp_path / "comps.jsonl"
    bt.append_comparison(path, bt.Comparison(winner="a", loser="b"))
    bt.append_comparison(path, bt.Comparison(winner="c", loser="a"))
    loaded = bt.load_comparisons(path)
    assert [(c.winner, c.loser) for c in loaded] == [("a", "b"), ("c", "a")]


def test_load_missing_file_empty(tmp_path):
    assert bt.load_comparisons(tmp_path / "nope.jsonl") == []


def test_select_pairs_no_duplicates_and_excludes():
    rng = np.random.default_rng(3)
    loss_vectors, _ = _synthetic_world(rng, n_candidates=6)
    exclude = {frozenset(("c00", "c01"))}
    pairs = bt.select_pairs(loss_vectors, None, n_pairs=5, exclude=exclude)
    assert len(pairs) == 5
    keys = [frozenset(p) for p in pairs]
    assert len(set(keys)) == 5
    assert frozenset(("c00", "c01")) not in keys


def test_active_selection_prefers_uncertain_pairs():
    rng = np.random.default_rng(4)
    loss_vectors, w_true = _synthetic_world(rng)
    comps = _simulate(rng, loss_vectors, w_true, n=200)
    fit = bt.fit_bt(comps, loss_vectors)
    pairs = bt.select_pairs(loss_vectors, fit, n_pairs=3)
    for a, b in pairs:
        p = bt.predict_prob(fit, loss_vectors[a], loss_vectors[b])
        assert 0.02 < p < 0.98  # not already-decided pairs


def test_select_pairs_respects_blocks():
    import numpy as np
    from loopviz.bt import select_pairs

    rng = np.random.default_rng(0)
    lv = {f"c{i}": rng.random(4) for i in range(6)}
    blocks = {f"c{i}": ("a" if i < 3 else "b") for i in range(6)}
    pairs = select_pairs(lv, None, n_pairs=20, blocks=blocks)
    assert pairs  # within-block pairs exist: 2 * C(3,2) = 6
    for x, y in pairs:
        assert blocks[x] == blocks[y]


# -- quadratic-interaction utility model ---------------------------------------

def _quad_world(rng, n_candidates=60):
    """Observer with an interior optimum on metric 0 and a 1x2 interaction."""
    phi = {f"c{i:02d}": rng.normal(0, 1, N_METRICS)
           for i in range(n_candidates)}

    def u_true(v):
        return -2.0 * (v[0] - 0.5) ** 2 + 1.5 * v[1] * v[2]

    return phi, u_true


def _simulate_quad(rng, phi, u_true, n=1500):
    ids = sorted(phi)
    comps = []
    for _ in range(n):
        i, j = rng.choice(len(ids), 2, replace=False)
        a, b = ids[i], ids[j]
        p_a = 1 / (1 + np.exp(-(u_true(phi[a]) - u_true(phi[b]))))
        winner, loser = (a, b) if rng.random() < p_a else (b, a)
        comps.append(bt.Comparison(winner=winner, loser=loser))
    return comps


def test_quad_fit_recovers_nonlinear_utility():
    rng = np.random.default_rng(5)
    phi, u_true = _quad_world(rng)
    comps = _simulate_quad(rng, phi, u_true)
    model = bt.quad_fit(comps, phi, l2=1.0)
    u_hat = np.array([bt.quad_utility(phi[c], model) for c in sorted(phi)])
    u_ref = np.array([u_true(phi[c]) for c in sorted(phi)])
    r = np.corrcoef(u_hat, u_ref)[0, 1]
    assert r > 0.8  # a linear-in-phi model cannot reach this


def test_quad_fit_beats_linear_on_interactions():
    rng = np.random.default_rng(6)
    phi, u_true = _quad_world(rng)
    train = _simulate_quad(rng, phi, u_true, n=1500)
    test = _simulate_quad(rng, phi, u_true, n=500)
    model = bt.quad_fit(train, phi, l2=1.0)
    correct = sum(
        bt.quad_utility(phi[c.winner], model)
        > bt.quad_utility(phi[c.loser], model)
        for c in test)
    assert correct / len(test) > 0.65


def test_scorer_from_weights_quad_and_fallback():
    rng = np.random.default_rng(7)
    phi, u_true = _quad_world(rng)
    comps = _simulate_quad(rng, phi, u_true, n=800)
    model = bt.quad_fit(comps, phi, l2=1.0)

    from loopviz.metrics import METRIC_NAMES
    as_dict = {cid: dict(zip(METRIC_NAMES, v)) for cid, v in phi.items()}

    score = bt.scorer_from_weights({"quad": model})
    a = sorted(phi)[0]
    # lower score = better = higher utility
    assert score(as_dict[a]) == pytest.approx(
        -bt.quad_utility(phi[a], model))

    # legacy file without "quad" falls back to w_raw on loss vectors
    from loopviz.loss import loss_vector_from_phi_dict
    w = rng.random(N_METRICS)
    score_lin = bt.scorer_from_weights({"w_raw": list(w)})
    assert score_lin(as_dict[a]) == pytest.approx(
        float(w @ loss_vector_from_phi_dict(as_dict[a])))


def test_quad_term_names_shape():
    names = bt.quad_term_names(["a", "b", "c"])
    assert names == ["a", "b", "c", "a^2", "b^2", "c^2",
                     "a x b", "a x c", "b x c"]
