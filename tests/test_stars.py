"""The star-calibration primitives that need no archive: catalog geometry, the fisheye
projection, and the predictive track linker.

CI installs only numpy and pytest (requirements-dev.txt), so the linker test, which lives
beside the OpenCV detector and uses scipy's KD-tree, skips where those are absent."""

from __future__ import annotations

import numpy as np
import pytest

from src.figlib.stars import catalog, fisheye
from src.figlib.stars.sun import sun

PALOMAR = (33.36302, -116.83622)
CAM = {"lat": PALOMAR[0], "lon": PALOMAR[1], "az": 180.0, "fov": 90.0}


def test_orion_is_up_from_palomar_before_dawn_in_october():
    # 2024-10-21 12:26 UTC, the first frame of 20241021_PalomarRidge_hp-s-mobo-c
    alt, az = catalog.altaz(*catalog.STARS["Rigel"], 1729513594, *PALOMAR)
    assert 35 < float(alt) < 60 and 180 < float(az) < 260


def test_sun_is_down_at_local_midnight_and_up_at_noon():
    midnight_pdt = 1789110029          # 2026-09-11 00:00:29 PDT
    assert float(sun(midnight_pdt, *PALOMAR)[0]) < -30
    assert float(sun(midnight_pdt + 12 * 3600, *PALOMAR)[0]) > 50


def test_fisheye_puts_the_boresight_at_frame_center_and_is_equidistant():
    W, H = 3072, 2048
    x, y = fisheye.project_fisheye(CAM, np.array([180.0]), np.array([0.0]), W, H)
    assert float(x[0]) == pytest.approx(0.5) and float(y[0]) == pytest.approx(0.5)
    k = fisheye.initial_k(CAM, W)
    # 10 and 20 deg right of the axis on the horizon land 10k and 20k px out, not tan-scaled
    x, _ = fisheye.project_fisheye(CAM, np.array([190.0, 200.0]), np.zeros(2), W, H, k_scale=k)
    px = (np.asarray(x) - 0.5) * W
    assert px[1] / px[0] == pytest.approx(2.0, rel=1e-3)


def test_predictive_linker_follows_a_moving_star_through_noise():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from src.figlib.stars import tracks

    rng = np.random.default_rng(0)
    frames = []
    for i in range(30):
        offset = 60 * i
        star = (1000.0 + 0.1 * offset, 500.0 + 0.02 * offset, 90.0)        # 6 px/min drift
        noise = [(float(x), float(y), 30.0) for x, y in rng.uniform([900, 400], [1400, 600], (15, 2))]
        frames.append((offset, [star] + noise))
    linked = tracks.link_tracks_predictive(frames)
    best = max(linked, key=len)
    assert len(best) == 30
    assert all(abs(best[o][0] - (1000.0 + 0.1 * o)) < 1e-6 for o in best)


def test_sky_model_defaults_to_the_full_chain():
    # proper motion, precession and refraction are on unless a caller switches them off
    assert catalog.MODEL == {"proper_motion": True, "precession": True, "refraction": True}
