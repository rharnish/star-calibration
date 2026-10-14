"""The sky mask: the frame above a skyline, through a pose. No terrain model needed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from star_calibration import fisheye, terrain  # noqa: E402
from star_calibration.hpwren import cameras  # noqa: E402

W, H = 3072, 2048
CAM = {**cameras()["hp-s-mobo-c"], "cx": 0.0, "cy": 0.0}
LEVEL = {"d_az": 0.0, "d_pitch": 0.0, "d_roll": 0.0, "k_ratio": fisheye.K_RATIO, "k1": fisheye.K1}


def flat(el):
    az = np.arange(0.0, 360.0, 0.1)
    return az, np.full_like(az, el)


def test_a_level_camera_under_a_flat_skyline_sees_sky_in_the_top_half():
    m = terrain.sky_mask(CAM, LEVEL, W, H, flat(0.0), margin_deg=0.5)
    assert m.shape == (H, W)
    col = m[:, W // 2]
    edge = np.flatnonzero(col)[-1]
    k = fisheye.K_RATIO * fisheye.initial_k(CAM, W)
    assert abs(edge - (H / 2 - np.radians(0.5) * k)) < 9      # one 8-px cell
    assert col[:edge].all() and not col[edge + 1:].any()


def test_a_higher_skyline_or_a_pitched_up_camera_moves_the_edge():
    base = terrain.sky_mask(CAM, LEVEL, W, H, flat(0.0)).mean()
    assert terrain.sky_mask(CAM, LEVEL, W, H, flat(5.0)).mean() < base
    assert terrain.sky_mask(CAM, {**LEVEL, "d_pitch": 5.0}, W, H, flat(0.0)).mean() > base
