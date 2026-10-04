import json
import threading
import urllib.parse
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest
from PIL import Image

from loopviz.paper import palette as P
from loopviz.paper.palette import PaletteSpec


@pytest.fixture
def gray(tmp_path):
    """A signed-matrix-like grayscale PNG: a full 0..1 gradient, mid=0."""
    g = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    path = tmp_path / "presentation.png"
    Image.fromarray((g * 255).astype(np.uint8), "L").save(path)
    return path, g


# -- color space --------------------------------------------------------------

def test_oklab_srgb_roundtrip():
    # any valid sRGB color survives sRGB -> OKLab -> sRGB (in-gamut by
    # construction, so no clipping to hide a bug)
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.0, (500, 3))
    lab = P.srgb_to_oklab(rgb)
    back = P.oklab_to_srgb(lab[:, 0], lab[:, 1], lab[:, 2])
    assert np.allclose(rgb, back, atol=1e-4)


def test_srgb_to_oklab_known_neutral():
    # mid-gray is neutral (a=b=0) and 0<L<1
    lab = P.srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    assert abs(lab[1]) < 1e-6 and abs(lab[2]) < 1e-6
    assert 0.4 < lab[0] < 0.7


# -- LUT structure ------------------------------------------------------------

def test_oklab_lut_shape_and_center_neutral():
    lut = P.oklab_lut(PaletteSpec("t", C_end=0.1))
    assert lut.shape == (511, 3)
    assert lut.min() >= 0.0 and lut.max() <= 1.0
    center = P.srgb_to_oklab(lut[255])
    assert np.hypot(center[1], center[2]) < 1e-3  # chroma ~ 0 at the middle


def test_oklab_lut_lightness_monotonic_per_arm():
    lut = P.oklab_lut(PaletteSpec("t", L_center=0.2, L_end=0.85, C_end=0.08))
    L = P.srgb_to_oklab(lut)[:, 0]
    mid = 255
    # dark center -> light ends: each arm's lightness increases outward
    left = L[:mid + 1][::-1]      # center outward to the negative end
    right = L[mid:]               # center outward to the positive end
    assert np.all(np.diff(left) >= -1e-6)
    assert np.all(np.diff(right) >= -1e-6)


def test_lut_from_spec_dispatch_and_mpl():
    assert P.lut_from_spec(PaletteSpec("t", kind="oklab")).shape == (511, 3)
    m = P.lut_from_spec(PaletteSpec("t", kind="mpl", cmap="RdBu_r"))
    assert m.shape == (511, 3) and m.min() >= 0 and m.max() <= 1
    with pytest.raises(ValueError):
        P.lut_from_spec(PaletteSpec("t", kind="bogus"))


# -- applying to an image -----------------------------------------------------

def test_apply_spec_endpoints_and_shape(gray):
    _, g = gray
    spec = PaletteSpec("t", C_end=0.08)
    rgb = P.apply_spec(g, spec)
    assert rgb.shape == (64, 64, 3)
    lut = P.lut_from_spec(spec)
    assert np.array_equal(rgb[0, 0], lut[0])       # g=0 -> first color
    assert np.array_equal(rgb[-1, -1], lut[-1])    # g=1 -> last color


def test_apply_spec_deterministic(gray):
    _, g = gray
    spec = PaletteSpec("t")
    assert np.array_equal(P.apply_spec(g, spec), P.apply_spec(g, spec))


# -- printability -------------------------------------------------------------

def test_gamut_report_fields():
    r = P.gamut_report(PaletteSpec("t", C_end=0.08))
    for key in ("srgb_clip_fraction", "max_oklab_chroma", "printable",
                "tier", "method"):
        assert key in r
    assert isinstance(r["printable"], bool)


def test_oversaturated_palette_not_printable():
    # far outside any print gamut - and outside sRGB, so flagged under BOTH
    # the soft-proof and the analytic fallback (method-independent).
    r = P.gamut_report(PaletteSpec("t", C_end=0.30, L_center=0.5, L_end=0.5))
    assert r["srgb_clip_fraction"] > 0.0
    assert r["printable"] is False


def test_low_chroma_light_palette_printable():
    # low chroma, lightness inside coated range: printable under either method
    r = P.gamut_report(PaletteSpec("t", L_center=0.5, L_end=0.75, C_end=0.05))
    assert r["printable"] is True


def test_designed_presets_printable_stock_maps_flagged():
    # every OKLab-designed preset must print faithfully (that is the point)
    designed = [s for s in P.PRESETS.values() if s.kind == "oklab"]
    assert len(designed) >= 12
    for s in designed:
        assert P.gamut_report(s)["printable"], s.name
    # saturated screen maps are honestly flagged, not silently clipped
    assert not P.gamut_report(P.PRESETS["coolwarm"])["printable"]
    assert not P.gamut_report(P.PRESETS["seismic"])["printable"]


def test_presets_all_have_group_labels():
    for s in P.PRESETS.values():
        assert s.group, s.name


def test_all_presets_produce_valid_luts():
    for name, spec in P.PRESETS.items():
        lut = P.lut_from_spec(spec)
        assert lut.shape == (spec.n_lut, 3), name
        assert np.isfinite(lut).all() and lut.min() >= 0 and lut.max() <= 1


# -- reproducible config IO ---------------------------------------------------

def test_save_and_reproduce(gray, tmp_path):
    source, g = gray
    spec = PaletteSpec("mypal", hue_neg=250, hue_pos=55, L_center=0.2,
                       L_end=0.8, C_end=0.09)
    cfg = P.save_config(spec, source, tmp_path / "palettes", source_id="cand42")

    d = json.loads(cfg.read_text())
    assert d["spec"]["name"] == "mypal"
    assert d["source"]["id"] == "cand42"
    assert len(d["source"]["sha256"]) == 64
    assert "printable" in d["gamut"]
    assert "created_utc" in d["provenance"]
    assert (tmp_path / "palettes" / "mypal.png").exists()

    # loaded spec reproduces the exact recolor: re-rendering from the saved
    # config must match the preview byte-for-byte (both read the same source)
    spec2 = P.load_spec(cfg)
    assert spec2 == spec
    out = tmp_path / "repro.png"
    P.apply_config(cfg, source, out)
    reloaded = np.asarray(Image.open(out), dtype=np.uint8)
    preview = np.asarray(Image.open(tmp_path / "palettes" / "mypal.png"),
                         dtype=np.uint8)
    assert np.array_equal(reloaded, preview)


def test_save_config_name_cannot_escape_out_dir(gray, tmp_path):
    # a malicious label from the UI must not write outside out_dir
    source, _ = gray
    out_dir = tmp_path / "palettes"
    cfg = P.save_config(PaletteSpec("../../escaped"), source, out_dir)
    assert out_dir in cfg.parents            # stayed inside out_dir
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path.parent / "escaped.json").exists()


# -- viewer server (real end-to-end over http) --------------------------------

@pytest.fixture
def server(gray, tmp_path):
    from loopviz.paper.palette_server import PaletteState, make_handler

    source, _ = gray
    state = PaletteState(source, tmp_path / "out", source_id="candX")
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}", tmp_path / "out"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read()


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read()


def test_server_index_and_presets(server):
    base, _ = server
    assert b"palette designer" in _get(base + "/")
    presets = json.loads(_get(base + "/presets"))["presets"]
    assert any(p["name"] == "ember" for p in presets)


def test_server_render_returns_png(server):
    base, _ = server
    spec = json.dumps(P.PRESETS["ember"].to_dict())
    png = _get(base + "/render?s=" + urllib.parse.quote(spec))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_server_gamut_and_save(server):
    base, out_dir = server
    spec = P.PRESETS["paper"].to_dict()
    g = json.loads(_post(base + "/gamut", spec))
    assert g["printable"] is True
    spec["name"] = "saved_from_server"
    resp = json.loads(_post(base + "/save", spec))
    assert resp["ok"] and (out_dir / "saved_from_server.json").exists()


# -- default "highest-rated" candidate selection ------------------------------

def _write_candidate(runs, cid, loss_eq, phi):
    d = runs / cid
    d.mkdir(parents=True)
    (d / "presentation.png").write_bytes(b"stub")   # only existence is checked
    (d / "candidate.json").write_text(json.dumps(
        {"id": cid, "loss_eq": loss_eq, "phi": phi}))


def test_best_candidate_prefers_fitted_then_falls_back(tmp_path):
    from loopviz.cli import _best_candidate
    from loopviz.config import Paths
    from loopviz.paper.metrics import METRIC_NAMES, METRICS

    runs = tmp_path / "runs"
    on_target = {m.name: m.target for m in METRICS}   # ~zero loss vector
    off_target = {n: 0.0 for n in METRIC_NAMES}
    # A: best by equal-weight yardstick (low loss_eq) but off the metric targets
    _write_candidate(runs, "A_low_eq", loss_eq=1.0, phi=off_target)
    # B: worse loss_eq, but sits on every target -> best under any weights
    _write_candidate(runs, "B_on_target", loss_eq=99.0, phi=on_target)
    # ignored: experiment dir, and a candidate with no presentation.png
    _write_candidate(runs, "exp_junk", loss_eq=0.0, phi=off_target)
    d = runs / "no_png"
    d.mkdir()
    (d / "candidate.json").write_text(json.dumps(
        {"id": "no_png", "loss_eq": -5.0, "phi": off_target}))

    paths = Paths(root=tmp_path)
    # no fitted weights yet -> equal-weight yardstick picks the low loss_eq
    assert _best_candidate(paths) == "A_low_eq"
    # with fitted weights -> ranks by w.loss; B (on target) wins
    w = [1.0] * len(METRIC_NAMES)
    (runs / "fitted_weights.json").write_text(json.dumps({"w_raw": w}))
    assert _best_candidate(paths) == "B_on_target"


def test_best_candidate_empty_raises(tmp_path):
    from loopviz.cli import _best_candidate
    from loopviz.config import Paths

    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit):
        _best_candidate(Paths(root=tmp_path))
