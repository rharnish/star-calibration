"""The overlay draws the fit where the tracks are, and draws something for a failure too."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from star_calibration import overlay  # noqa: E402
from star_calibration.solve import solve  # noqa: E402
from test_solve import H, W, synthetic_night  # noqa: E402

GREEN, MAGENTA = (60, 220, 60), (255, 60, 255)


def _count(img, bgr, tol=40):
    return int((np.abs(img.astype(int) - np.array(bgr)).max(axis=2) < tol).sum())


def test_solved_overlay_puts_the_fit_on_the_tracks():
    from star_calibration import catalog as SG
    from star_calibration.fisheye import initial_k, project_fisheye
    night = synthetic_night("hp-s-mobo-c", (2.4, -0.7, 0.9))
    r = solve(night)
    img = overlay.draw(r, night, np.zeros((H, W, 3), np.uint8))
    assert img.shape[1] == overlay.WIDTH
    assert _count(img, GREEN) > 1000 and _count(img, MAGENTA) > 1000
    # Where the fitted pose puts each matched star, the picture shows its track (green, with
    # magenta on top), not background. Checked well inside each track, on the side away from
    # the reference frame, where the yellow arrows end on the fit. The picture is the frame
    # under a 90 px banner, scaled to WIDTH. The right pose hits 23 of these 26 (orange arcs,
    # arrows and labels cover a few); a pose 1 deg off hits 3.
    s, p, ref = overlay.WIDTH / W, r["pose"], r["ref_offset"]
    cam = night.cam
    on = 0
    for name, j in r["matches"].items():
        offs = sorted(night.tracks[j])
        o = offs[len(offs) // 4] if abs(offs[0] - ref) > abs(offs[-1] - ref) else offs[3 * len(offs) // 4]
        alt, az = SG.star_altaz(name, night.t0 + o, cam["lat"], cam["lon"], cam["elev"])
        x, y = project_fisheye(cam, az, alt, W, H, p["d_az"], p["d_pitch"], p["d_roll"],
                               p["k_ratio"] * initial_k(cam, W), p["k1"])
        px = img[int((float(y) * H + 90) * s), min(int(float(x) * W * s), overlay.WIDTH - 1)]
        on += min(np.abs(px.astype(int) - GREEN).max(), np.abs(px.astype(int) - MAGENTA).max()) < 60
    assert len(r["matches"]) >= 10 and on >= 0.75 * len(r["matches"])


def test_failed_overlay_shows_the_tracks_and_the_published_sky():
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    r = {"seq": "synthetic", "status": "failed", "reason": "test", "n_tracks": len(night.tracks),
         "n_tracks_raw": len(night.tracks)}
    img = overlay.draw(r, night, np.zeros((H, W, 3), np.uint8))
    assert _count(img, (230, 210, 60)) > 500          # cyan: the tracks used
    assert _count(img, (0, 140, 255)) > 500           # orange: stars under the published pose


def test_reference_offset_prefers_the_solve():
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    assert overlay.reference_offset({"ref_offset": 1200}, night) == 1200
    offs = sorted({o for t in night.tracks for o in t})
    assert overlay.reference_offset({}, night) == offs[len(offs) // 2]
