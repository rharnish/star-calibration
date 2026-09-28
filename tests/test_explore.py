"""The replay's data holds every step of the solve, projected, and is valid JSON."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from star_calibration import explore  # noqa: E402
from star_calibration.solve import solve, solve_wide  # noqa: E402
from test_solve import synthetic_night  # noqa: E402


def test_export_holds_every_attempt_and_rung():
    night = synthetic_night("hp-s-mobo-c", (40.0, 1.0, -0.5))
    trace = []
    r = solve_wide(night, trace=trace)
    frames = [(o, f"f/{o}.jpg") for o in sorted({o for t in night.tracks for o in t})]
    d = explore.export(trace, r, night, frames)
    json.dumps(d, allow_nan=False)
    assert d["result"]["status"] == r["status"] == "solved"
    assert len(d["attempts"]) == sum(e["stage"] == "attempt" for e in trace)
    assert sum(len(a["rungs"]) for a in d["attempts"]) == sum(e["stage"] == "rung" for e in trace)
    assert d["attempts"][d["kept"]]["kept"]
    assert d["frames"][0] == [frames[0][0], frames[0][1]]
    # the result's arcs run along the tracks they matched: the synthetic sky is exact
    for name, j in r["matches"].items():
        arc = np.array(d["final"]["fitted_arcs"][name], float)
        track = np.array([p[1:] for p in d["tracks"][j]], float)
        gap = np.hypot(*np.moveaxis(arc[:, None, :] - track[None, :, :], -1, 0))   # arc x track
        assert np.nanmin(gap, axis=1).max() < 2.0
    pole = next(a["pole"] for a in d["attempts"] if a["wide"])
    assert pole["kept"] is not None and len(pole["kept"]) == len(pole["steps"])


def test_nothing_behind_the_camera_lands_in_frame():
    """The lens polynomial folds back past ~118 deg: a star behind the camera must not be
    drawn in the picture."""
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    sky = explore._Sky(night)
    c = night.cam
    behind = (c["az"] + 180.0) % 360.0
    xy = sky.project(np.array([behind]), np.array([-(c.get("pitch") or 0.0)]),
                     (0.0, 0.0, 0.0, 1000.0, -0.078))
    assert np.isnan(xy).all()


def test_a_failure_exports_too():
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    night.tracks = night.tracks[:5]
    trace = []
    r = solve(night, trace=trace)
    d = explore.export(trace, r, night, [])
    json.dumps(d, allow_nan=False)
    assert d["attempts"][0]["verdict"]["reason"] == r["reason"]
