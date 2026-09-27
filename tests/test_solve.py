"""End to end on a synthetic night: catalog stars through a known camera, back to that camera.

Tracks are made the way the sky makes them -- every catalog star's apparent position, a frame
a minute for 90 minutes, projected through a pose and lens that are not the published ones --
and handed to the solver as a `Night`. Nothing here reads a file but the camera table and
the catalog, so it runs anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from star_calibration import catalog as SG  # noqa: E402
from star_calibration.fisheye import K1, K_RATIO, initial_k, project_fisheye  # noqa: E402
from star_calibration.hpwren import cameras  # noqa: E402
from star_calibration.solve import Night, solve, solve_wide  # noqa: E402

W, H = 3072, 2048
T0 = 1789112704          # 2026-09-11 07:45 UTC, a Q1 block on a moonless night


def synthetic_night(camera: str, pose: tuple[float, float, float], k_ratio: float = K_RATIO,
                    k1: float = K1, jitter_px: float = 0.3, seed: int = 0) -> Night:
    cam = cameras()[camera]
    rng = np.random.default_rng(seed)
    k = k_ratio * initial_k(cam, W)
    vis = SG.visible_stars(cam, T0, mag_limit=4.0, fov_margin_deg=180, min_alt_deg=5)
    offsets = np.arange(0, 90 * 60, 60)
    tracks = []
    for v in vis:
        alt, az = SG.star_altaz(v["name"], T0 + offsets.astype(float), cam["lat"], cam["lon"],
                                cam.get("elev") or 1600.0)
        x, y = project_fisheye(cam, az, alt, W, H, *pose, k, k1)
        x, y = x * W + rng.normal(0, jitter_px, len(x)), y * H + rng.normal(0, jitter_px, len(y))
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < 0.35 * H)     # the detector's sky band
        t = {int(o): (float(xi), float(yi), 100.0) for o, xi, yi, k_ in zip(offsets, x, y, ok) if k_}
        if len(t) >= 8:
            tracks.append(t)
    return Night(camera=camera, cam=cam, t0=T0, tracks=tracks, W=W, H=H, label="synthetic")


def test_grid_solve_recovers_an_offset_camera():
    truth = (2.4, -0.7, 0.9)
    r = solve(synthetic_night("hp-s-mobo-c", truth))
    assert r["status"] == "solved", r.get("reason")
    got = [r["pose"][k] for k in ("d_az", "d_pitch", "d_roll")]
    assert np.allclose(got, truth, atol=0.03)
    assert r["pose"]["k_ratio"] == pytest.approx(K_RATIO, abs=0.003)
    assert r["median_px"] < 1.0 and r["sky_model"] == SG.model_id()


def test_wide_solve_reaches_an_azimuth_the_grid_cannot():
    # 40 deg off the published azimuth is outside the +-30 deg grid; the pole search finds it
    truth = (40.0, 1.0, -0.5)
    r = solve_wide(synthetic_night("hp-s-mobo-c", truth))
    assert r["status"] == "solved" and r["found_by"] == "pole"
    assert np.allclose([r["pose"][k] for k in ("d_az", "d_pitch", "d_roll")], truth, atol=0.05)


def test_too_few_tracks_fails_with_a_reason():
    n = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    n.tracks = n.tracks[:5]
    r = solve(n)
    assert r["status"] == "failed" and "moving tracks" in r["reason"]
