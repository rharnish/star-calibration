"""The pose ledger's rule for when a star-solved correction applies to another date."""

from __future__ import annotations

import pytest

from src.figlib import pose_ledger as L

DAY = L.DAY_S


def entry(camera, day, d_az, frame_w=3072):
    return {"camera": camera, "epoch": day * DAY, "frame_w": frame_w, "d_az": d_az,
            "source": f"star:{camera}@{day}"}


LEDGER = [
    entry("a", 100, 2.0), entry("a", 101, 2.4),          # two solves one night apart
    entry("b", 100, 0.3), entry("b", 900, 0.5),          # agree across a long gap
    entry("om-s", 100, -10.5), entry("om-s", 1800, -0.4),  # re-aimed in between
    entry("c", 1000, 1.5),
]


def test_same_night_takes_the_median():
    hit = L.lookup(LEDGER, "a", 100.5 * DAY)
    assert hit["rule"] == "same-night" and hit["d_az"] == pytest.approx(2.2)


def test_bracketed_agreeing_solves_apply():
    hit = L.lookup(LEDGER, "b", 500 * DAY)
    assert hit["rule"] == "bracketed" and hit["d_az"] == pytest.approx(0.4)


def test_bracketed_disagreeing_solves_mean_the_camera_moved():
    assert L.lookup(LEDGER, "om-s", 900 * DAY) is None
    assert L.lookup(LEDGER, "om-s", 101 * DAY)["d_az"] == pytest.approx(-10.5)


def test_one_sided_only_within_max_days():
    assert L.lookup(LEDGER, "c", 1200 * DAY)["rule"] == "one-sided"
    assert L.lookup(LEDGER, "c", (1000 - L.MAX_DAYS - 1) * DAY) is None


def test_unknown_camera_has_no_correction():
    assert L.lookup(LEDGER, "zz", 100 * DAY) is None


def test_corrected_cam_is_a_no_op_unless_enabled(monkeypatch):
    cam = {"az": 180.0, "fov": 90, "frame_w": 3072}
    monkeypatch.delenv("FIGLIB_POSE_LEDGER", raising=False)
    assert L.corrected_cam("c", cam, 1000 * DAY, LEDGER) == (cam, None)
    monkeypatch.setenv("FIGLIB_POSE_LEDGER", "1")
    out, hit = L.corrected_cam("c", cam, 1000 * DAY, LEDGER)
    assert out["az"] == pytest.approx(181.5) and cam["az"] == 180.0 and hit["rule"] == "same-night"


def test_a_solve_never_crosses_a_change_of_frame_format():
    assert L.lookup(LEDGER, "c", 1000 * DAY, frame_w=2048) is None
    assert L.lookup(LEDGER, "c", 1000 * DAY, frame_w=3072)["d_az"] == pytest.approx(1.5)
