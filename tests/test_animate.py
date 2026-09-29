"""The playback draws every attempt at one picture size, and writes a video a player can open."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from star_calibration import animate  # noqa: E402
from star_calibration.solve import solve_wide  # noqa: E402
from test_solve import H, W, synthetic_night  # noqa: E402


def test_frames_play_every_attempt_at_one_size(tmp_path):
    night = synthetic_night("hp-s-mobo-c", (40.0, 1.0, -0.5))
    trace = []
    r = solve_wide(night, trace=trace, grid="always")   # every attempt, not just a sure pole
    offs = sorted({o for t in night.tracks for o in t})[::10]
    images = ((o, np.full((H, W, 3), 20, np.uint8)) for o in offs)
    pics = list(animate.frames(trace, r, night, images))
    assert len({p.shape for p in pics}) == 1
    assert pics[0].shape[1] == animate.OUT_W
    # the night, then a card, pole, scan, refinement and verdict per attempt, then the result
    assert len(pics) > len(offs) * 2 + 3 * animate.FPS * 4
    kept = animate.frames(trace, r, night, ((o, np.zeros((H, W, 3), np.uint8)) for o in offs),
                          only_kept=True)
    assert sum(1 for _ in kept) < len(pics)

    dest = animate.write(pics[:48], tmp_path / "solve")
    cap = cv2.VideoCapture(str(dest))
    assert dest.suffix == ".mp4" and cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 48


def test_no_frames_is_an_error():
    night = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        list(animate.frames([], {"status": "failed", "reason": "x"}, night, iter([])))


def test_full_frame_and_marked_stars():
    night = synthetic_night("hp-s-mobo-c", (2.4, -0.7, 0.9))
    trace = []
    r = solve_wide(night, trace=trace)
    offs = sorted({o for t in night.tracks for o in t})[::10]

    def pics(**kw):
        return list(animate.frames(trace, r, night,
                                   ((o, np.zeros((H, W, 3), np.uint8)) for o in offs), **kw))
    band, full = pics(), pics(full_frame=True, mark=["Alkaid-Mizar-Alioth", "Polaris"])
    assert full[0].shape[0] > band[0].shape[0] and full[0].shape[1] == band[0].shape[1]
    assert len(full) == len(band) + 6 * animate.FPS       # the closing scene of named stars
    with pytest.raises(ValueError, match="Nostar"):
        next(animate.frames(trace, r, night, iter([]), mark=["Nostar"]))
