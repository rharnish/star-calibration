"""Pole-from-trails and the lunar ephemeris."""
import math

import numpy as np
import pytest

from star_calibration import moon as M
from star_calibration import pole as P
from star_calibration.hpwren import cameras

CAMS = cameras()


def _epoch(iso):
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("iso,want", [
    ("2024-04-08T18:17", 0.0),    # total solar eclipse -> new moon
    ("2017-08-21T18:26", 0.0),
    ("2026-08-12T17:46", 0.0),
    ("2025-03-14T06:58", 1.0),    # total lunar eclipse -> full moon
    ("2022-11-08T11:00", 1.0),
])
def test_illuminated_fraction_matches_real_eclipses(iso, want):
    assert float(M.illuminated_fraction(_epoch(iso))) == pytest.approx(want, abs=0.02)


def test_moon_is_up_at_a_known_full_moon_transit():
    # the 2025-03-14 lunar eclipse was visible from southern California
    el, _ = M.moon(_epoch("2025-03-14T06:58"), 33.33, -116.92)
    assert el > 30


@pytest.mark.parametrize("pose", [(0.0, 0.0, 0.0), (12.5, -3.2, 4.1), (-170.0, 20.0, -25.0)])
def test_pose_axes_round_trip(pose):
    c = CAMS["bh-w-mobo-c"]
    back = P.pose_from_axes(P.axes_from_pose(c, *pose)[None], c)[0]
    assert back == pytest.approx(np.array(pose), abs=1e-6)


@pytest.mark.parametrize("cam,pose", [
    ("bh-w-mobo-c", (12.5, -3.2, 4.1)),
    ("bm-s-mobo-c", (-170.0, 20.0, -25.0)),   # far outside the old +-30 deg grid
    ("hp-w-mobo-c", (1.0, 0.5, -0.3)),
])
def test_pole_family_contains_the_true_pose(cam, pose):
    """The 1-D family through a known pole must pass through the pose that produced it."""
    c = CAMS[cam]
    p_cam = P.pole_in_cam(c, *pose)
    psi = np.arange(0.0, 360.0, 0.02)
    fam = P.pose_from_axes(P.cam_from_pole(p_cam, c["lat"], psi), c)
    d = np.abs(fam - np.array(pose))
    d[:, 0] = np.minimum(d[:, 0], 360 - d[:, 0])
    d[:, 2] = np.minimum(d[:, 2], 360 - d[:, 2])
    assert d.sum(axis=1).min() < 0.05


def test_pixel_to_cam_inverts_the_lens():
    from star_calibration.fisheye import initial_k, project_fisheye
    c = CAMS["hp-w-mobo-c"]
    W, H, k1 = 3072, 2048, -0.078
    k = 0.886 * initial_k(c, W)
    az = np.array([265.0, 270.0, 285.0, 250.0])
    el = np.array([2.0, 10.0, -5.0, 20.0])
    x, y = project_fisheye(c, az, el, W, H, 0.0, 0.0, 0.0, k, k1)
    d = P.pixel_to_cam(x * W, y * H, W, H, k, k1)
    want = P.axes_from_pose(c, 0.0, 0.0, 0.0) @ np.array(
        [np.cos(np.radians(el)) * np.sin(np.radians(az)),
         np.cos(np.radians(el)) * np.cos(np.radians(az)),
         np.sin(np.radians(el))])
    assert np.abs(d - want.T).max() < 1e-3


def test_fit_pole_recovers_a_synthetic_rotation():
    """Synthesise trails from a known pose and check the closed-form fit returns it."""
    from star_calibration.fisheye import initial_k, project_fisheye
    from star_calibration import catalog as SG
    c = CAMS["hp-w-mobo-c"]
    W, H, k1 = 3072, 2048, -0.078
    k = 0.886 * initial_k(c, W)
    pose = (1.0, 0.5, -0.3)
    t0, offs = 1789112704, np.arange(0, 5400, 60.0)
    names = [r["name"] for r in SG.CATALOG if r["mag"] <= 3.0][:60]
    tracks = []
    for n in names:
        alt, az = SG.altaz(*SG.STARS[n], t0 + offs, c["lat"], c["lon"])
        x, y = project_fisheye(c, az, alt, W, H, *pose, k, k1)
        X, Y = x * W, y * H
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        if ok.sum() >= 8:
            tracks.append({float(o): (float(a), float(b), 50.0)
                           for o, a, b in zip(offs[ok], X[ok], Y[ok])})
    assert len(tracks) >= 8
    f = P.estimate(tracks, W, H, k, k1)
    assert f["norm"] == pytest.approx(1.0, abs=0.02)
    err = math.degrees(math.acos(np.clip(np.dot(f["p_hat"], P.pole_in_cam(c, *pose)), -1, 1)))
    assert err < 0.5


def test_scan_context_matches_the_solve_it_describes():
    """The figure's view of the coarse stage must be the solver's, not a lookalike.

    `fig_solve_process` draws `solve.scan_context`; if that ever drifts from what `solve`
    actually optimises the diagram would quietly become fiction. Check the peak of the psi
    scan lands on the pose the solver committed to.
    """
    pytest.importorskip("scipy")
    from star_calibration.hpwren import calibrate
    from star_calibration.solve import scan_context
    seq = "hpwren_20260911_Q1_wc-n-mobo-c"
    if not (calibrate.tracks_dir() / f"tracks_{seq}.pkl").exists():
        pytest.skip("cached tracks not present ($HPWREN_CACHE)")
    res = next((r for r in calibrate.summary() if r["seq"] == seq), None)
    if res is None or res["status"] != "solved":
        pytest.skip("block not solved in the cache's summary")
    ctx = scan_context(calibrate.night(seq))
    assert ctx["fit"]["norm"] == pytest.approx(1.0, abs=0.05)
    peak = ctx["poses"][int(np.argmax(ctx["scores"]))]
    got = np.array([res["pose"][k] for k in ("d_az", "d_pitch", "d_roll")])
    assert np.abs(peak - got).max() < 1.0


def test_pixel_to_cam_measures_from_the_optical_centre():
    d = P.pixel_to_cam(np.array([1536.0 + 20]), np.array([1024.0 - 30]), 3072, 2048, 1500.0, -0.078,
                       cx=20.0, cy=-30.0)
    assert np.allclose(d[0], [0.0, 0.0, 1.0])
