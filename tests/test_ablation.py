"""The ablation study's refits and re-solves, on a synthetic night whose truth is known."""
from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from star_calibration import ablation as A  # noqa: E402
from star_calibration.solve import solve, solve_wide  # noqa: E402
from test_solve import synthetic_night  # noqa: E402

TRUTH = (2.4, -0.7, 0.9)


@pytest.fixture(scope="module")
def solved():
    night = synthetic_night("hp-s-mobo-c", TRUTH)
    r = solve(night)
    assert r["status"] == "solved"
    return r, night


def test_the_baseline_refit_is_the_solve(solved):
    r, night = solved
    d = A.refit_experiments(r, night, sizes=(4,), draws=3)
    base = d["rows"][0]
    assert base["kind"] == "baseline" and base["n_stars"] == r["n_stars"]
    for k in A.POSE:
        assert base[k] == pytest.approx(r["pose"][k], abs=1e-3)
    assert base["k_ratio"] == pytest.approx(r["pose"]["k_ratio"], abs=1e-4)


def test_every_star_is_left_out_once_and_radial_cuts_split_by_theta(solved):
    r, night = solved
    d = A.refit_experiments(r, night, sizes=(4,), draws=3)
    loo = [x for x in d["rows"] if x["kind"] == "leave_one_out"]
    assert sorted(x["drop"] for x in loo) == sorted(r["matches"])
    assert all(x["n_stars"] == r["n_stars"] - 1 for x in loo)
    for q in A.RADIAL_FRACTIONS:
        inner = next(x for x in d["rows"] if x["kind"] == "inner" and x["fraction"] == q)
        outer = next(x for x in d["rows"] if x["kind"] == "outer" and x["fraction"] == q)
        assert inner["n_stars"] == outer["n_stars"]
        assert inner["theta_max"] <= outer["theta_max"] and inner["theta_min"] <= outer["theta_min"]


def test_a_free_centre_finds_the_true_one():
    night = synthetic_night("hp-s-mobo-c", TRUTH, centre=(30.0, -20.0))
    r = solve(night)
    assert r["status"] == "solved"
    night.cam = {**night.cam, "cx": 0.0, "cy": 0.0}     # the refit starts from the middle
    base = A.refit_experiments(r, night, centre=True, sizes=(), draws=0)["rows"][0]
    assert base["cx"] == pytest.approx(30.0, abs=3) and base["cy"] == pytest.approx(-20.0, abs=3)


def test_variants_drop_by_radius_and_at_random():
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    v = A.variants(night, seeds=1, fractions=(0.5,))
    n = len(night.tracks)
    assert v["all"] == list(range(n))
    rad = A.track_radius(night)
    inner = set(range(n)) - set(v["drop_inner_third"])
    outer = set(range(n)) - set(v["drop_outer_third"])
    assert max(rad[list(inner)]) <= min(rad[list(outer)])
    assert len(v["keep_50_s0"]) == round(n / 2)


def test_acceptance_thresholds_are_parameters():
    night = synthetic_night("hp-s-mobo-c", TRUTH)
    assert solve(night)["status"] == "solved"
    tight = solve(night, max_median_px=0.01)
    assert tight["status"] == "failed" and tight["reason"].startswith("best:")
    assert solve(night, k_window=(0.5, 0.6))["status"] == "failed"
    few = synthetic_night("hp-s-mobo-c", TRUTH)
    few.tracks = few.tracks[:6]
    assert "moving tracks" in solve_wide(few)["reason"]
    assert "moving tracks" not in (solve_wide(few, min_tracks=4, min_stars=4)["reason"] or "")
