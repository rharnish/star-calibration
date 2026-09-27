"""Reading a pixel through the whole star-solved camera (`fisheye.unproject_fisheye`).

A solved camera has its own lens scale, radial term, pitch and roll; with the camera pitched
or rolled, a pixel's azimuth depends on its row. These pin that the inverse is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from star_calibration.fisheye import initial_k, project_fisheye, unproject_fisheye

CAM = {"lat": 33.0, "lon": -116.8, "az": 135.0, "fov": 90, "elev": 1000.0}
W, H = 3072, 2048


@pytest.mark.parametrize("pose", [(0.0, 0.0, 0.0, 0.886, -0.078),
                                  (1.2, 16.8, -1.4, 0.892, -0.08),     # mpo-n-like pitch
                                  (-4.0, 1.8, -3.2, 0.887, -0.085),    # bi-w-like roll
                                  (4.8, -0.5, 0.3, 0.775, -0.04)])     # bm-like lens
def test_unproject_inverts_project(pose):
    d_az, d_pitch, d_roll, kr, k1 = pose
    k = kr * initial_k(CAM, W)
    az = CAM["az"] + d_az + np.linspace(-45, 45, 13)
    el = np.linspace(-15, 25, 13)
    A, E = np.meshgrid(az, el)
    x, y = project_fisheye(CAM, A, E, W, H, d_az, d_pitch, d_roll, k, k1)
    a2, e2 = unproject_fisheye(CAM, x, y, W, H, d_az, d_pitch, d_roll, k, k1)
    assert np.max(np.abs((a2 - A + 180) % 360 - 180)) < 1e-6
    assert np.max(np.abs(e2 - E)) < 1e-6


def test_pitch_makes_azimuth_depend_on_the_row():
    """The same column, two rows: level camera agrees, pitched camera doesn't."""
    k = 0.886 * initial_k(CAM, W)
    level = [unproject_fisheye(CAM, 0.8, y, W, H, 0, 0, 0, k, -0.078)[0] for y in (0.5, 0.5)]
    pitched = [unproject_fisheye(CAM, 0.8, y, W, H, 0, 16.8, 0, k, -0.078)[0] for y in (0.5, 0.8)]
    assert abs(level[0] - level[1]) < 1e-9
    assert abs(pitched[0] - pitched[1]) > 1.0
