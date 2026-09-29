"""The HPWREN package: its camera table, its ledger, and the cache layout nights.py keeps."""
from __future__ import annotations

import json

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


def test_placeholder_frames_are_skipped(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    d = tmp_path / "nights" / "bl-e-mobo-c" / "20260911_Q1"
    d.mkdir(parents=True)
    epochs = [1789113600 + 60 * i for i in range(10)]
    for i, e in enumerate(epochs):
        W, H = (320, 240) if i < 7 else (480, 320)
        (d / f"{e}.jpg").write_bytes(cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))[1].tobytes())
    nights.index()
    seq = "hpwren_20260911_Q1_bl-e-mobo-c"
    real, placeholders = nights.frame_files(seq)
    assert [int(p.stem) for p in real] == epochs[7:] and len(placeholders) == 7
    assert [f[0] for f in nights.read_frames(seq)] == epochs[7:]
    assert nights.frame_at(seq, -1e9)[0] == epochs[7]
    assert nights.jpeg_size((d / f"{epochs[8]}.jpg").read_bytes()) == (480, 320)


def test_ledger_build_dates_each_solve_by_its_block(tmp_path):
    summary = [{"seq": "b1", "camera": "hp-s-mobo-c", "status": "solved", "W": 3072,
                "pose": {"d_az": 0.31234, "d_pitch": 0.0, "d_roll": -0.5, "k_ratio": 0.8861,
                         "k1": -0.078}, "n_stars": 20, "median_px": 1.234, "sky_model": "m"},
               {"seq": "b2", "camera": "hp-s-mobo-c", "status": "failed"}]
    rows = ledger.build(summary, {"b1": 1789113600.0}, tmp_path / "l.json")
    assert rows == ledger.load(tmp_path / "l.json")
    assert rows == [{"camera": "hp-s-mobo-c", "epoch": 1789113600, "frame_w": 3072,
                     "d_az": 0.312, "d_pitch": 0.0, "d_roll": -0.5, "k_ratio": 0.8861,
                     "k1": -0.078, "cx": 0.0, "cy": 0.0, "n_stars": 20, "median_px": 1.23,
                     "source": "star:b1",
                     "sky_model": "m"}]


def test_mixed_sky_models_are_refused(tmp_path):
    p = tmp_path / "mixed.json"
    p.write_text('[{"camera": "a", "sky_model": "x"}, {"camera": "b", "sky_model": "y"}]')
    with pytest.raises(ValueError, match="mixes sky models"):
        ledger.load(p)


def test_added_result_joins_the_summary_but_not_the_ledger(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    import numpy as np
    from star_calibration.hpwren import calibrate, weather
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    pose = {"d_az": 0.3, "d_pitch": 0.0, "d_roll": 0.0, "k_ratio": 0.886, "k1": -0.078}
    r = {"seq": "20191030_CopperCanyon_om-s-mobo-m", "camera": "om-s-mobo-m", "status": "solved",
         "W": 3072, "pose": pose, "n_stars": 11, "median_px": 0.85, "n_tracks": 93}
    with pytest.raises(ValueError, match="needs"):
        calibrate.add(r, None, "figlib")
    r |= {"t0": 1572497000, "lat": 32.5948, "lon": -116.8447, "ref_offset": 680}
    calibrate.add(r, np.zeros((40, 60, 3), np.uint8), "figlib")
    (row,) = calibrate.summary()
    assert row["source"] == "figlib" and calibrate.block(row, {})["t0"] == 1572497000
    assert (tmp_path / "overlays" / f"star_solve_{r['seq']}.jpg").exists()
    assert weather.ref_epoch(row) == 1572497680 and weather.site_latlon(row) == (32.5948, -116.8447)
    calibrate.main(["ledger", str(tmp_path / "l.json")])
    assert json.loads((tmp_path / "l.json").read_text()) == []
