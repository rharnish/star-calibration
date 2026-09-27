"""The HPWREN package: its camera table, its ledger, and the cache layout nights.py keeps."""
from __future__ import annotations

import pytest

from star_calibration import ledger
from star_calibration.hpwren import cache_dir, cameras, ledger_path, nights


def test_camera_table_is_the_published_one():
    cams = cameras()
    assert len(cams) == 505
    c = cams["hp-s-mobo-c"]
    assert {"site", "lat", "lon", "elev", "az", "fov", "imager"} <= set(c)
    assert 32 < c["lat"] < 34 and -118 < c["lon"] < -116


def test_shipped_ledger_is_one_sky_model_and_known_cameras():
    rows = ledger.load(ledger_path())
    assert len(rows) >= 100
    cams = cameras()
    assert all(r["camera"] in cams and r["source"].startswith("star:hpwren_") for r in rows)
    hit = ledger.lookup(rows, rows[0]["camera"], rows[0]["epoch"], rows[0]["frame_w"])
    assert hit is not None and hit["rule"] == "same-night"


def test_cache_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    assert cache_dir() == tmp_path and nights.frames_dir() == tmp_path / "nights"


def test_index_and_read_frames_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    d = tmp_path / "nights" / "hp-s-mobo-c" / "20260911_Q1"
    d.mkdir(parents=True)
    epochs = [1789113600 + 60 * i for i in range(10)]
    for e in epochs:
        (d / f"{e}.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    idx = nights.index()
    s = idx["hpwren_20260911_Q1_hp-s-mobo-c"]
    assert s["camera"] == "hp-s-mobo-c" and s["n_frames"] == 10 and s["t0"] == epochs[5]
    assert s["dir"] == "hp-s-mobo-c/20260911_Q1"
    frames = nights.read_frames("hpwren_20260911_Q1_hp-s-mobo-c")
    assert [f[0] for f in frames] == epochs and frames[0][1] == epochs[0] - s["t0"]


def test_ledger_build_dates_each_solve_by_its_block(tmp_path):
    summary = [{"seq": "b1", "camera": "hp-s-mobo-c", "status": "solved", "W": 3072,
                "pose": {"d_az": 0.31234, "d_pitch": 0.0, "d_roll": -0.5, "k_ratio": 0.8861,
                         "k1": -0.078}, "n_stars": 20, "median_px": 1.234, "sky_model": "m"},
               {"seq": "b2", "camera": "hp-s-mobo-c", "status": "failed"}]
    rows = ledger.build(summary, {"b1": 1789113600.0}, tmp_path / "l.json")
    assert rows == ledger.load(tmp_path / "l.json")
    assert rows == [{"camera": "hp-s-mobo-c", "epoch": 1789113600, "frame_w": 3072,
                     "d_az": 0.312, "d_pitch": 0.0, "d_roll": -0.5, "k_ratio": 0.8861,
                     "k1": -0.078, "n_stars": 20, "median_px": 1.23, "source": "star:b1",
                     "sky_model": "m"}]


def test_mixed_sky_models_are_refused(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text('[{"camera": "a", "sky_model": "x"}, {"camera": "b", "sky_model": "y"}]')
    with pytest.raises(ValueError, match="mixes sky models"):
        ledger.load(p)
