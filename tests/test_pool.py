import json

from playlistviz.pool import candidate_dir, iter_candidate_files, slug, song_dir


def _mk(d, cid):
    d.mkdir(parents=True)
    (d / "candidate.json").write_text(json.dumps({"id": cid}))


def test_slug():
    assert slug("Klein Tambotieboom!") == "klein-tambotieboom"
    assert slug("") == "untitled"


def test_song_dir_layout(tmp_path):
    d = song_dir(tmp_path, "Piano Man", "033.wav", "A2")
    assert d == tmp_path / "songs" / "piano-man_t033" / "A2"
    assert song_dir(tmp_path, "", "007.wav").name == "pre"


def test_iter_and_resolve_both_layouts(tmp_path):
    flat = tmp_path / "cand_flat"
    _mk(flat, "cand_flat")
    nested = song_dir(tmp_path, "Piano Man", "033.wav", "A3") / "cand_nested"
    _mk(nested, "cand_nested")
    (tmp_path / "exp_something").mkdir()
    (tmp_path / "exp_something" / "candidate.json").write_text("{}")

    files = iter_candidate_files(tmp_path)
    ids = {f.parent.name for f in files}
    assert ids == {"cand_flat", "cand_nested"}   # exp_* excluded

    assert candidate_dir(tmp_path, "cand_flat") == flat
    assert candidate_dir(tmp_path, "cand_nested") == nested
    assert candidate_dir(tmp_path, "missing") is None
