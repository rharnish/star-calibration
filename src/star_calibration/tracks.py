"""Track point sources across a night of frames, not just one frame.

Single-frame star matching (star_match.py) struggles because a static shape/brightness
descriptor can't tell a real star from a lens artifact, and can't break certain shape
symmetries (Orion's Betelgeuse/Bellatrix swap). A sequence adds a completely different kind
of evidence: real stars all drift together at the sidereal rate (~15 deg/hr) as the sky
turns, while lens artifacts and sensor defects don't move at all -- the same fact
`sun_calibrate.drop_static_artifacts` already exploits for the sun's disk, applied here to
whole point-source *tracks* instead of one dot per sequence.

Approach: decode every dark-enough frame of one sequence, detect point sources in each (same
method as star_pixels.py), link them frame-to-frame by nearest neighbor (real motion between
60s frames is small and smooth, so this is a much easier tracking problem than general
multi-object tracking), then keep only tracks that (a) persist across most of the sequence and
(b) actually move a real distance end-to-end -- which throws out static artifacts for free and
leaves a much smaller, cleaner candidate pool for star_match.py's triangle matching.

`collect` takes the frames from any source -- an iterable of (epoch, offset_s, jpeg bytes),
offsets counted from whatever epoch the caller treats as the block's reference -- so the same
code tracks HPWREN CDN night blocks (`hpwren.nights.read_frames`) and night sequences inside
other archives alike.
"""
from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .sun import sun as sun_altaz


MAX_POINTS_PER_FRAME = 2000


def detect_points(img: np.ndarray, sky_frac: float = 0.35, thresh: int = 25,
                    area_range: tuple[int, int] = (1, 16)) -> list[tuple[float, float, float]]:
    """Same point-source detector as star_pixels.py: median-blur-subtract, threshold, small
    connected components in the sky region."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    sky = g[: int(H * sky_frac)]
    resid = cv2.subtract(sky, cv2.medianBlur(sky, 21))
    peaks = (resid > thresh).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(peaks, 8)
    out = []
    for k in range(1, n):
        area = int(st[k, cv2.CC_STAT_AREA])
        if area_range[0] <= area <= area_range[1]:
            out.append((float(cen[k][0]), float(cen[k][1]), int(resid[lab == k].max())))
    return out


def link_tracks(frames_points: list[tuple[int, list]], max_step_px: float = 20.0,
                max_gap_s: float | None = None) -> list[dict]:
    """Greedy nearest-neighbor linking across consecutive frames (sorted by offset). Each
    track is a dict of offset -> (x, y, amp).

    With max_gap_s None an unmatched track stays open indefinitely, which is harmless over a
    90-minute block but not over a night: a star lost behind cloud can be picked up hours
    later by whichever other star wanders within max_step_px of where it vanished. A whole
    night passes max_gap_s (as link_tracks_predictive already does) to close such tracks."""
    tracks: list[dict] = []
    open_tracks: list[tuple[int, dict]] = []  # (last_offset, track)
    for offset, points in frames_points:
        used = set()
        still_open = []
        for last_offset, track in open_tracks:
            lx, ly, _ = track[last_offset]
            cands = [(np.hypot(px - lx, py - ly), j) for j, (px, py, pa) in enumerate(points)
                      if j not in used]
            if cands:
                d, j = min(cands)
                if d < max_step_px:
                    track[offset] = points[j]
                    used.add(j)
                    still_open.append((offset, track))
                    continue
            if max_gap_s is None or offset - last_offset <= max_gap_s:
                still_open.append((last_offset, track))  # no match this frame, keep waiting briefly
        open_tracks = still_open
        for j, p in enumerate(points):
            if j not in used:
                t = {offset: p}
                tracks.append(t)
                open_tracks.append((offset, t))
    return tracks


def detect_points_adaptive(img: np.ndarray, sky_frac: float = 0.35, k_sigma: float = 7.0,
                           area_range: tuple[int, int] = (2, 40)) -> list[tuple[float, float, float]]:
    """Point sources for the monochrome (NIR) units, whose sky noise sigma is 1.7-3.1 gray
    levels against the color units' near-zero: a fixed `thresh=25` there is only 8-15 sigma
    of *grain*, and yielded 800-3700 "moving" tracks of noise per sequence
    (plume-triangulation's NOTES.md, 2026-09-13). Smooth by sigma 1 px first (a star spreads over a few pixels, grain doesn't),
    subtract the same median-21 background, and threshold at k_sigma times a noise sigma
    taken from the 16th-84th percentile spread -- the MAD is 0 on integer-quantized sky."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sky = g[: int(g.shape[0] * sky_frac)]
    r = cv2.GaussianBlur(sky.astype(np.float32), (0, 0), 1.0) - cv2.medianBlur(sky, 21).astype(np.float32)
    sub = r[::4, ::4]
    sigma = max((np.percentile(sub, 84.13) - np.percentile(sub, 15.87)) / 2, 0.5)
    n, lab, st, cen = cv2.connectedComponentsWithStats((r > k_sigma * sigma).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if area_range[0] <= st[i, cv2.CC_STAT_AREA] <= area_range[1]:
            out.append((float(cen[i][0]), float(cen[i][1]), float(r[lab == i].max())))
    return out


def link_tracks_predictive(frames_points: list[tuple[int, list]], max_gap_s: float = 240.0) -> list[dict]:
    """Nearest-neighbor linking with a constant-velocity prediction, for noisy detections.

    `link_tracks` accepts the nearest point within 20 px of a track's last position, which on
    a grainy sensor hops between noise hits and draws the jagged tracks in
    star_solve_*_lp-s-mobo-m.jpg. Stars move smoothly, so once a track has two points,
    predict where it should be and accept only a point within 3 px (+0.02 px/s of gap);
    the first link, with no velocity yet, gets 4 px + 0.2 px/s (12 px at a 60 s frame).
    Matches are assigned one-to-one, closest first across all tracks, and a track closes
    after max_gap_s without a detection. Each track is a dict of offset -> (x, y, amp)."""
    from scipy.spatial import cKDTree
    tracks: list[dict] = []
    open_tracks: list[dict] = []
    for offset, points in frames_points:
        P = np.array([p[:2] for p in points]) if points else np.zeros((0, 2))
        tree = cKDTree(P) if len(P) else None
        cands = []
        for ti, tr in enumerate(open_tracks):
            dt = offset - tr["last"]
            if tr["v"] is None:
                pred, gate = tr["pos"], 4.0 + 0.2 * dt
            else:
                pred, gate = tr["pos"] + tr["v"] * dt, 3.0 + 0.02 * dt
            if tree is not None:
                for j in tree.query_ball_point(pred, gate):
                    cands.append((float(np.hypot(*(P[j] - pred))), ti, j))
        cands.sort()
        used_t, used_p = set(), set()
        for _, ti, j in cands:
            if ti in used_t or j in used_p:
                continue
            used_t.add(ti)
            used_p.add(j)
            tr = open_tracks[ti]
            v = (P[j] - tr["pos"]) / (offset - tr["last"])
            tr["v"] = v if tr["v"] is None else 0.5 * tr["v"] + 0.5 * v
            tr["pos"], tr["last"] = P[j], offset
            tr["track"][offset] = points[j]
        open_tracks = [tr for tr in open_tracks if offset - tr["last"] <= max_gap_s]
        for j, p in enumerate(points):
            if j not in used_p:
                t = {offset: p}
                tracks.append(t)
                open_tracks.append({"track": t, "last": offset, "pos": P[j], "v": None})
    return tracks


def moving_tracks(tracks: list[dict], min_frames: int = 8, min_span_px: float = 50.0) -> list[dict]:
    """Keep tracks seen in enough frames AND that actually moved -- a static hot pixel or
    lens artifact gets detected in nearly every frame at (almost) the same spot; a real star
    doesn't."""
    out = []
    for t in tracks:
        if len(t) < min_frames:
            continue
        offsets = sorted(t)
        (x0, y0, _), (x1, y1, _) = t[offsets[0]], t[offsets[-1]]
        if np.hypot(x1 - x0, y1 - y0) >= min_span_px:
            out.append(t)
    return out


def shape(track: dict, min_step_px: float = 10.0) -> dict:
    """How a track moves: `n` points, `rate_px_h` (end-to-end distance over its duration),
    `turn_deg` (median angle between consecutive steps of at least min_step_px),
    `reversals` (share of those steps that turn more than 90 degrees) and `cubic_px` (rms about a cubic in time, per coordinate).

    A star drifts smoothly: its steps turn by a few degrees and a cubic leaves well under a
    pixel. Cloud texture, haze and noise linked by nearest neighbour wander: over the solved
    CDN blocks, a median turn above 40 degrees or more than 30% reversals marks 1.3% of the
    tracks matched to a star and 29% of the rest (docs/records.md, "Track cleaning")."""
    o = np.array(sorted(track), float)
    xy = np.array([track[k][:2] for k in sorted(track)], float)
    span = float(np.hypot(*(xy[-1] - xy[0])))
    dur = o[-1] - o[0]
    out = {"n": len(o), "rate_px_h": span / dur * 3600 if dur > 0 else 0.0,
           "turn_deg": 0.0, "reversals": 0.0, "cubic_px": 0.0}
    # turning between points at least min_step_px apart: a slow star (near the pole, or
    # faint, with a centroid that jitters by a pixel) moves 1-5 px a frame, and rounding
    # alone would swing its direction by tens of degrees
    anchors = [0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - xy[anchors[-1]])) >= min_step_px:
            anchors.append(i)
    steps = np.diff(xy[anchors], axis=0)
    if len(steps) >= 2:
        a = np.arctan2(steps[:, 1], steps[:, 0])
        turn = np.degrees(np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))
        out["turn_deg"] = float(np.median(turn))
        out["reversals"] = float(np.mean(turn > 90))
    if len(o) >= 5:
        t = (o - o[0]) / max(dur, 1.0)
        V = np.vander(t, 4)
        r = xy - V @ np.linalg.lstsq(V, xy, rcond=None)[0]
        out["cubic_px"] = float(np.sqrt(np.mean(r ** 2)))
    return out


def wandering(track: dict, max_turn_deg: float = 40.0, max_reversals: float = 0.3) -> bool:
    """A track that wanders rather than drifts; see `shape`."""
    s = shape(track)
    return s["turn_deg"] > max_turn_deg or s["reversals"] > max_reversals


def _cubic_parts(t: np.ndarray, xy: np.ndarray, tol_px: float, min_frames: int,
                 min_span_px: float, max_parts: int, rng) -> list[np.ndarray]:
    """Masks of the points in one gap-free piece that follow one cubic path each: RANSAC the
    largest set within tol_px of a cubic in time, keep it if it passes `moving_tracks`' test,
    then do the same with what's left."""
    parts = []
    left = np.ones(len(t), bool)
    while left.sum() >= min_frames and len(parts) < max_parts:
        idx = np.where(left)[0]
        tt, pp = t[idx], xy[idx]
        V = np.vander(tt, 4)
        # 4-point samples bunched in time extrapolate wildly across the rest; skip them
        S = np.sort(rng.integers(0, len(idx), (400, 4)), axis=1)
        S = S[(np.diff(S, axis=1) > 0).all(1) & (tt[S[:, 3]] - tt[S[:, 0]] >= 0.1 * np.ptp(tt))]
        if not len(S):
            break
        coef = np.linalg.solve(V[S], pp[S])                       # (samples, 4, 2)
        err = np.hypot(*(np.einsum("nk,skd->snd", V, coef) - pp).transpose(2, 0, 1))
        inl = err[np.argmax((err < tol_px).sum(1))] < tol_px
        for _ in range(3):                                         # polish on the inliers
            if inl.sum() < 4:
                break
            c = np.linalg.lstsq(V[inl], pp[inl], rcond=None)[0]
            new = np.hypot(*(V @ c - pp).T) < tol_px
            if new.sum() <= inl.sum():
                break
            inl = new
        keep = idx[inl]
        if len(keep) < min_frames or np.hypot(*(xy[keep[-1]] - xy[keep[0]])) < min_span_px:
            break
        m = np.zeros(len(t), bool)
        m[keep] = True
        parts.append(m)
        left &= ~m
    return parts


def split(track: dict, tol_px: float = 3.0, gap_s: float = 600.0, min_frames: int = 8,
          min_span_px: float = 50.0, max_parts: int = 3) -> list[dict]:
    """A linked track as the star tracks it contains, largest first.

    Nearest-neighbour linking sometimes hands a track from one star to another: a track left
    open picks up a later star that passes its last position (Markab then Algenib, 70 min
    apart on the same pixels), or it hops between close stars (the Pleiades). What isn't a
    star gets attached too, such as a clock digit in HPWREN's banner. Each star's points
    follow one smooth path, so: cut at gaps longer than gap_s (a cubic across a long gap is
    unconstrained), and in each piece take the largest set of points within tol_px of one
    cubic in time, then the next largest from what's left. Every part passes
    `moving_tracks`' test; points in no part are dropped. A track one cubic already explains
    comes back unchanged, and so does one with no part of min_frames points.

    On the solved CDN blocks this left the tracks that stay on their star untouched, took 91%
    of the off-star points out of the rest, and found a second star in 1 of 8 of them."""
    o = sorted(track)
    xy = np.array([track[k][:2] for k in o], float)
    t = np.array(o, float)
    dur = max(t[-1] - t[0], 1.0)
    if len(o) >= 5:
        V = np.vander((t - t[0]) / dur, 4)
        if np.hypot(*(V @ np.linalg.lstsq(V, xy, rcond=None)[0] - xy).T).max() < tol_px:
            return [track]
    rng = np.random.default_rng(0)                               # the same parts every run
    cuts = [0, *(np.where(np.diff(t) > gap_s)[0] + 1), len(o)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < min_frames:
            continue
        tp = t[a:b]
        for m in _cubic_parts((tp - tp[0]) / max(tp[-1] - tp[0], 1.0), xy[a:b], tol_px,
                              min_frames, min_span_px, max_parts, rng):
            out.append({o[a + i]: track[o[a + i]] for i in np.where(m)[0]})
    # nothing better found: a short track with a few stray points is still its star's track
    return sorted(out, key=len, reverse=True) or [track]


def duplicates(tracks: list[dict], tol_px: float = 3.0, min_share: float = 0.5) -> set[int]:
    """Indices of tracks that repeat a longer track: within tol_px of it in at least
    min_share of their own frames. Splitting finds some stars a second time, since a star a
    track hopped to often has its own track as well."""
    from scipy.spatial import cKDTree
    frames: dict[int, list[tuple[int, float, float]]] = {}
    for i, t in enumerate(tracks):
        for o, v in t.items():
            frames.setdefault(o, []).append((i, v[0], v[1]))
    shared: dict[tuple[int, int], int] = {}
    for pts in frames.values():
        if len(pts) < 2:
            continue
        P = np.array([p[1:] for p in pts])
        for a, b in cKDTree(P).query_pairs(tol_px):
            i, j = sorted((pts[a][0], pts[b][0]), key=lambda k: (-len(tracks[k]), k))
            shared[i, j] = shared.get((i, j), 0) + 1               # j is the shorter
    return {j for (i, j), n in shared.items() if n >= min_share * len(tracks[j])}


def clean(tracks: list[dict], banner_px: int = 0, max_banner_rate_px_h: float = 60.0,
          min_frames: int = 8, min_span_px: float = 50.0,
          min_wandering_share: float = 0.5) -> tuple[list[dict], dict]:
    """Linked tracks with what isn't a star taken out, and counts of what went.

    Each track is split into the stars it contains (`split`). A wandering track (`wandering`)
    keeps only its largest smooth part, and only if that part holds min_wandering_share of
    its points: a star with junk attached (banner digits, a noisy tail) qualifies, cloud
    texture rarely does. Over the solved CDN blocks the share reached 0.5 for 57% of the
    wandering tracks matched to a star and 15% of the rest. Testing the whole track instead
    would drop the star with the junk; testing only the parts would let a cubic threaded
    through a few points of noise pass.

    Then near-stationary tracks in the top banner_px rows go (a camera's burned-in text:
    HPWREN's clock digits change every frame and link into slow "tracks" that drag the pole
    fit toward zero; real stars cross the banner too, hence the rate test), and so do parts
    that repeat a longer track (`duplicates`)."""
    counts = {"linked": len(tracks), "wandering": 0, "split": 0, "banner": 0}
    parts = []
    for t in tracks:
        p = split(t, min_frames=min_frames, min_span_px=min_span_px)
        if wandering(t):
            if p == [t] or len(p[0]) < min_wandering_share * len(t):
                counts["wandering"] += 1
                continue
            p = p[:1]
        counts["split"] += p != [t]
        parts += p
    out = []
    for t in parts:
        y = np.median([v[1] for v in t.values()])
        if y < banner_px and shape(t)["rate_px_h"] < max_banner_rate_px_h:
            counts["banner"] += 1
        else:
            out.append(t)
    dup = duplicates(out)
    counts["duplicate"] = len(dup)
    out = [t for i, t in enumerate(out) if i not in dup]
    counts["kept"] = len(out)
    return out, counts


def collect(frames: Iterable[tuple[int, int, bytes]], lat: float, lon: float,
            mono: bool = False, el_max: float = -8.0, max_step_px: float = 20.0,
            min_frames: int = 8, min_span_px: float = 50.0, keep_frames: bool = True,
            max_gap_s: float | None = None, label: str = "frames"):
    """Moving tracks from one camera's frames, plus the decoded frames.

    `frames` is (epoch, offset_s, jpeg bytes) in time order; a track is a dict of
    offset_s -> (x, y, amp). Frames with the sun above `el_max` at (lat, lon) are skipped.
    `mono` switches to the detector and linker for the monochrome (NIR) units.
    keep_frames=False keeps only the first decoded frame (enough for the frame size): a whole
    night is ~470 frames of 19 MB each. Returns (tracks, {offset: (epoch, image)}).
    """
    frames = list(frames)
    frames_points = []
    decoded = {}
    for epoch, offset, blob in frames:
        el, az = sun_altaz(epoch, lat, lon)
        if el > el_max:
            continue
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if keep_frames or not decoded:
            decoded[offset] = (epoch, img)
        pts = detect_points_adaptive(img) if mono else detect_points(img)
        frames_points.append((offset, pts))
    print(f"{label}: {len(frames_points)}/{len(frames)} frames dark enough (sun el <= {el_max})")
    # A sky full of city glow or lit cloud texture yields thousands of "point sources" per
    # frame, and nearest-neighbor linking over that is quadratic: hpwren bl-n-mobo-c ran a
    # worker for 37 minutes on it. Stars are tens to a few hundred per frame, so give up
    # early and say why.
    counts = sorted(len(p) for _, p in frames_points)
    if counts and counts[len(counts) // 2] > MAX_POINTS_PER_FRAME:
        print(f"  median {counts[len(counts) // 2]} point sources per frame > {MAX_POINTS_PER_FRAME}: "
              f"not a star field, skipping")
        return [], decoded
    tracks = (link_tracks_predictive(frames_points, **({} if max_gap_s is None else
                                                       {"max_gap_s": max_gap_s})) if mono
              else link_tracks(frames_points, max_step_px=max_step_px, max_gap_s=max_gap_s))
    good = moving_tracks(tracks, min_frames=min_frames, min_span_px=min_span_px)
    print(f"  {len(tracks)} raw tracks -> {len(good)} moving, persistent tracks")
    return good, decoded

