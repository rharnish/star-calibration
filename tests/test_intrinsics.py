"""The per-camera optical centre: the joint fit recovers it, a jump starts a new segment, and
lookup picks the segment for a date."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from star_calibration import catalog as SG  # noqa: E402
from star_calibration import intrinsics as I  # noqa: E402
from star_calibration.fisheye import initial_k, project_fisheye  # noqa: E402
from star_calibration.hpwren import cameras  # noqa: E402

W, H = 3072, 2048
CAM = cameras()["hp-s-mobo-c"]
T0 = 1789112704


def night_obs(t0, pose, centre, n_stars=12, seed=0):
    """One night's matched points, as `intrinsics.observations` returns them, for a camera
    whose true centre is `centre`; its recorded solve assumed the frame's middle."""
    rng = np.random.default_rng(seed)
    c = {**CAM, "cx": centre[0], "cy": centre[1]}
    k = 0.886 * initial_k(CAM, W)
    vis = SG.visible_stars(CAM, t0, mag_limit=4.0, fov_margin_deg=10, min_alt_deg=10)
    X, Y, AZ, AL, n = [], [], [], [], 0
    for v in vis:
        o = t0 + np.arange(0, 5400, 60.0)
        alt, az = SG.star_altaz(v["name"], o, CAM["lat"], CAM["lon"], CAM.get("elev") or 1600.0)
        x, y = project_fisheye(c, az, alt, W, H, *pose, k, -0.078)
        x, y = x * W, y * H
        ok = np.isfinite(x) & (x > 0) & (x < W) & (y > 0) & (y < 0.4 * H)
        if ok.sum() < 8:
            continue
        X.append(x[ok] + rng.normal(0, 0.4, ok.sum()))
        Y.append(y[ok] + rng.normal(0, 0.4, ok.sum()))
        AZ.append(az[ok])
        AL.append(alt[ok])
        n += 1
        if n == n_stars:
            break
    return {"seq": f"n{t0}", "t0": t0, "W": W, "H": H, "n_stars": n,
            "x": np.concatenate(X), "y": np.concatenate(Y), "az": np.concatenate(AZ),
            "alt": np.concatenate(AL), "pose": list(pose), "k_ratio": 0.886, "k1": -0.078}


def test_joint_fit_recovers_the_centre():
    obs = [night_obs(T0 + 86400 * i, (1.0 + 0.1 * i, -0.5, 0.3), (-24.0, 17.0), seed=i)
           for i in range(3)]
    f = I.fit(obs, CAM)
    assert abs(f["cx"] + 24) < 2 and abs(f["cy"] - 17) < 2
    assert f["median_px"] < I.fit(obs, CAM, centre=False)["median_px"]


def test_a_jump_in_centre_starts_a_new_segment_and_lookup_follows_it():
    a = [night_obs(T0 + 86400 * i, (1.0, -0.5, 0.3), (-24.0, 17.0), seed=i) for i in range(3)]
    b = [night_obs(T0 + 86400 * (10 + i), (1.0, -0.5, 0.3), (30.0, -10.0), seed=9 + i)
         for i in range(2)]
    segs = I.series(a + b, CAM)
    assert [s["n_nights"] for s in segs] == [3, 2]
    rows = [{"camera": "hp-s-mobo-c", **s} for s in segs]
    assert I.lookup(rows, "hp-s-mobo-c", W, H, T0 + 86400) == (segs[0]["cx"], segs[0]["cy"])
    assert I.lookup(rows, "hp-s-mobo-c", W, H, T0 + 86400 * 30) == (segs[1]["cx"], segs[1]["cy"])
    assert I.lookup(rows, "hp-s-mobo-c", W, H, T0 - 86400) == (segs[0]["cx"], segs[0]["cy"])
    assert I.lookup(rows, "hp-s-mobo-c", 2048, 1536, T0) == (0.0, 0.0)
    assert I.lookup(rows, "vo-n-mobo-c", W, H, T0) == (0.0, 0.0)


def test_one_thin_night_keeps_the_frame_middle():
    segs = I.series([night_obs(T0, (1.0, -0.5, 0.3), (-24.0, 17.0), n_stars=8)], CAM)
    assert (segs[0]["cx"], segs[0]["cy"]) == (0.0, 0.0)
