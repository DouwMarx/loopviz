"""LoopSpec round trip, signal(), and --loop in the relief and plate CLIs."""

import json

import numpy as np
import pytest
import soundfile as sf

from loopviz.cli import main
from loopviz.loopspec import LoopSpec, resolve_loop


def _wav(path, sr=4000, seconds=2.0):
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * 110 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    x = 0.5 * (x + 0.2 * np.random.default_rng(0).standard_normal(x.size))    # peak < 1
    sf.write(str(path), x, sr)
    return x, sr


def test_round_trip_and_signal(tmp_path):
    x, sr = _wav(tmp_path / "s.wav")
    spec = LoopSpec(tmp_path / "s.wav", 0.2, 1.7, "demo", title="t", notes="n")
    p = spec.save(tmp_path / "loops" / "demo.json")
    back = LoopSpec.load(p)
    assert back == spec and back.audio == tmp_path / "s.wav"
    assert json.loads(p.read_text())["audio"] == str(tmp_path / "s.wav")
    sig, T, rate = back.signal(out_wav=tmp_path / "loop.wav")
    assert rate == sr and T == pytest.approx(1.5)
    assert np.allclose(sig, x[800:6800], atol=1e-4)      # 16-bit wav
    assert (tmp_path / "loop.wav").exists()
    with pytest.raises(ValueError):
        LoopSpec(tmp_path / "s.wav", 2.0, 1.0, "bad")


def test_project_spec_is_valid():
    spec = LoopSpec.load("loops/armed_man.json")
    assert spec.name == "armed_man" and spec.T == pytest.approx(12.025)
    assert "Benedictus" in spec.title


def test_resolve_loop_from_audio_flags(tmp_path):
    _wav(tmp_path / "s.wav")

    class A:
        loop = None
        audio = str(tmp_path / "s.wav")
        start = None
        end = None

    spec = resolve_loop(A())
    assert spec.name == "s" and spec.start_s == 0.0 and spec.end_s == pytest.approx(2.0)
    A.start, A.end = 0.5, 1.0
    assert resolve_loop(A()).T == pytest.approx(0.5)
    A.loop = "loops/armed_man.json"
    with pytest.raises(SystemExit):
        resolve_loop(A())


def test_cli_loop_flag(tmp_path, capsys):
    _wav(tmp_path / "s.wav")
    spec = LoopSpec(tmp_path / "s.wav", 0.2, 1.7, "demo")
    spec.save(tmp_path / "demo.json")
    common = ["--bed", "40", "--margin", "2", "--rho", "0.7"]
    main(["relief", "sweep", "--loop", str(tmp_path / "demo.json"), "--pitches", "2.0", *common])
    assert "loop demo 0.2..1.7 s" in capsys.readouterr().out
    with pytest.raises(SystemExit):           # mutually exclusive
        main(["relief", "sweep", "--loop", str(tmp_path / "demo.json"),
              "--audio", str(tmp_path / "s.wav"), *common])
    out = tmp_path / "relief"
    main(["relief", "build", "--loop", str(tmp_path / "demo.json"), "--pitch", "2.0",
          "--out", str(out), *common])
    meta = json.loads((out / "relief.json").read_text())
    assert meta["loop"]["name"] == "demo" and meta["start_s"] == 0.2
    out = tmp_path / "plate"
    main(["plate", "build", "--loop", str(tmp_path / "demo.json"), "--pitch", "2", "--side", "40",
          "--cols", "1", "--rows", "1", "--thickness", "8", "--rho", "0.7", "--no-labels",
          "--out", str(out)])
    m = json.loads((out / "metrics.json").read_text())
    assert m["loop"] == spec.to_dict() and m["n"] == 20 and m["end_s"] == 1.7
