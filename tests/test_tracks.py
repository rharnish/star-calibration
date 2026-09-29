"""Track cleaning: what `tracks.clean` keeps and drops, on synthetic tracks shaped like the
cases found in the CDN blocks (docs/records.md, "Track cleaning")."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from star_calibration import tracks as T  # noqa: E402


def star(x0, y0, vx=0.12, vy=0.05, offsets=range(0, 5400, 60), jitter=0.4, seed=0):
    """A star drifting at (vx, vy) px/s with a little centroid noise: 90 frames, 60 s apart."""
    rng = np.random.default_rng(seed)
    return {o: (x0 + vx * o + 0.00001 * o ** 1.5 + rng.normal(0, jitter),
                y0 + vy * o + rng.normal(0, jitter), 80.0) for o in offsets}


def near(a, b, px=1e-9):
    return set(a) == set(b) and all(abs(a[o][0] - b[o][0]) <= px for o in a)


def test_a_star_is_smooth_and_left_alone():
    t = star(1000, 300)
    s = T.shape(t)
    assert s["turn_deg"] < 10 and s["reversals"] == 0 and s["cubic_px"] < 1
    assert not T.squiggly(t)
    assert T.split(t) == [t]


def test_a_slow_star_with_rounded_centroids_is_not_squiggly():
    # Kochab near the pole: ~2 px a frame, rounded to whole pixels, amplitude near threshold
    rng = np.random.default_rng(1)
    t = {o: (round(1080 + 0.035 * o + rng.normal(0, 1)), round(270 + 0.02 * o + rng.normal(0, 1)), 27.0)
         for o in range(0, 5400, 60)}
    assert not T.squiggly(t)


def test_cloud_texture_is_squiggly_and_dropped():
    rng = np.random.default_rng(2)
    xy = np.cumsum(rng.normal(0, 8, (60, 2)), axis=0) + [1500, 400]
    t = {60 * i: (float(x), float(y), 30.0) for i, (x, y) in enumerate(xy)}
    assert T.squiggly(t)
    out, counts = T.clean([t, star(500, 300)])
    assert counts["squiggly"] == 1 and len(out) == 1


def test_a_track_handed_on_after_a_gap_splits_into_both_stars():
    # Markab then Algenib: the second star crosses the first one's pixels 70 min later
    a = star(1500, 100, offsets=range(0, 1500, 60))
    b = {o + 4200: (x + 0.5, y - 0.5, 80.0) for o, (x, y, _) in star(1500, 100, offsets=range(0, 1200, 60), seed=3).items()}
    parts = T.split({**a, **b})
    assert len(parts) == 2 and near(parts[0], a) and near(parts[1], b)


def test_flip_flopping_between_close_stars_gives_two_tracks():
    # the Pleiades: two stars 10 px apart, and the linker alternating between them in runs
    a, b = star(1100, 320, seed=4), star(1108, 314, seed=5)
    mixed = {o: (a if (o // 600) % 2 == 0 else b)[o] for o in a}
    parts = T.split(mixed)
    assert len(parts) == 2
    for p in parts:
        src = a if all(p[o] == a[o] for o in p) else b
        assert all(p[o] == src[o] for o in p)


def test_junk_on_the_front_of_a_star_is_cut_off():
    # Zet Oph: banner noise, held open, until the star rises into it and takes the track over
    rng = np.random.default_rng(6)
    junk = {o: (540 + rng.uniform(-12, 12), 10 + rng.uniform(-8, 8), 26.0) for o in range(0, 1200, 60)}
    s = star(536, 26, vx=0.06, vy=0.08, offsets=range(1200, 5400, 60))
    parts = T.split({**junk, **s})
    assert len(parts) == 1 and near(parts[0], s)


def test_a_short_track_with_no_clean_part_is_kept_whole():
    # a bright star alternating with faint detections a few px off: no 8-point cubic part
    t = star(1520, 290, offsets=range(0, 600, 60))
    t = {o: (x + (6.0 if i % 2 else 0.0), y, 80.0) for i, (o, (x, y, _)) in enumerate(t.items())}
    assert T.split(t) == [t]


def test_banner_tracks_go_and_stars_crossing_the_banner_stay():
    rng = np.random.default_rng(7)
    digit = {o: (700 + 0.01 * o + rng.uniform(-3, 3), 18 + rng.uniform(-2, 2), 30.0)
             for o in range(0, 5400, 60)}                     # ~36 px/h, in the text rows
    crossing = star(900, 5, vx=0.1, vy=0.01)                 # a real star skimming the top
    out, counts = T.clean([digit, crossing], banner_px=60)
    assert counts["banner"] >= 1 and out == [crossing]      # the digit may split, all of it goes
    out, counts = T.clean([digit, crossing])                   # no banner given: kept
    assert counts["banner"] == 0 and crossing in out and len(out) > 1


def test_a_track_repeating_a_longer_one_is_dropped():
    a = star(1000, 300)
    copy = {o: (x + 1.0, y, v) for o, (x, y, v) in a.items() if o < 3000}
    other = star(1400, 300, seed=8)
    assert T.duplicates([a, copy, other]) == {1}
    out, counts = T.clean([a, copy, other])
    assert counts["duplicate"] == 1 and len(out) == 2
