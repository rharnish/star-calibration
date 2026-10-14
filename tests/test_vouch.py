"""Vouching: a fit turned down only for its star count is accepted when another night of
the same camera, solved on its own, lands on the same pose."""
from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from star_calibration.hpwren import calibrate  # noqa: E402

DAY = 86400.0
POSE = {"d_az": 1.0, "d_pitch": -0.5, "d_roll": 0.3, "k_ratio": 0.886, "k1": -0.078}


def row(seq, day, status, n, d_az=1.0, median=0.8, k=0.886):
    r = {"seq": seq, "camera": "hp-s-mobo-c", "t0": 1.79e9 + day * DAY, "status": status,
         "reason": None if status == "solved" else f"best: {n} stars", "n_stars": n,
         "median_px": median, "found_by": "pole", "pose": {**POSE, "d_az": d_az, "k_ratio": k}}
    return r


def test_a_thin_night_close_to_a_solved_one_is_vouched_for():
    rows = [row("a", 0, "solved", 30), row("b", 2, "failed", 6, d_az=1.05)]
    changed = calibrate.vouch(rows)
    assert [r["seq"] for r in changed] == ["b"]
    assert rows[1]["status"] == "solved" and rows[1]["reason"] is None
    assert rows[1]["vouched_by"]["seq"] == "a" and rows[1]["reason_alone"] == "best: 6 stars"


@pytest.mark.parametrize("kw", [dict(d_az=1.3), dict(median=3.5), dict(k=0.6)])
def test_no_vouching_for_a_pose_that_disagrees_or_a_fit_that_fails_another_test(kw):
    rows = [row("a", 0, "solved", 30), row("b", 2, "failed", 6, **kw)]
    assert calibrate.vouch(rows) == [] and rows[1]["status"] == "failed"


def test_too_few_stars_too_far_apart_or_only_a_vouched_neighbour():
    assert calibrate.vouch([row("a", 0, "solved", 30), row("b", 2, "failed", 4)]) == []
    assert calibrate.vouch([row("a", 0, "solved", 30), row("b", 40, "failed", 6)]) == []
    rows = [row("a", 0, "solved", 30), row("b", 2, "failed", 6), row("c", 20, "failed", 6)]
    calibrate.vouch(rows)
    assert rows[2]["vouched_by"]["seq"] == "a"     # b, vouched for itself, doesn't vouch


def test_vouching_is_decided_again_when_the_neighbour_changes():
    rows = [row("a", 0, "solved", 30), row("b", 2, "failed", 6)]
    calibrate.vouch(rows)
    rows[0].update(status="failed", reason="only 3 moving tracks")
    calibrate.vouch(rows)
    assert rows[1]["status"] == "failed" and rows[1]["reason"] == "best: 6 stars"
    assert "vouched_by" not in rows[1] and "reason_alone" not in rows[1]
