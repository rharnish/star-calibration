"""Track point sources across a whole night sequence, not just one frame.

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
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from .. import corpus as C
from ..detect_yolo import read_frames
from .sun import sun as sun_altaz

OUT = ROOT / "out" / "sky"
CAMS = json.loads((ROOT / "data/meta/cams.json").read_text())
SEQS = {s["seq"]: s for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
from . import nights as hpwren_nights  # recent CDN nights, same shape as FIgLib sequences
SEQS.update(hpwren_nights.sequences())


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
    of *grain*, and yielded 800-3700 "moving" tracks of noise per sequence (NOTES.md,
    2026-09-13). Smooth by sigma 1 px first (a star spreads over a few pixels, grain doesn't),
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


def collect(seq: str, el_max: float = -8.0, max_step_px: float = 20.0,
            min_frames: int = 8, min_span_px: float = 50.0, keep_frames: bool = True,
            max_gap_s: float | None = None):
    """Moving tracks for one sequence, plus the decoded frames and the sequence record.

    keep_frames=False keeps only the first decoded frame (enough for the frame size, which is
    all solve.load_tracks reads): a whole night is ~470 frames of 19 MB each."""
    s = SEQS[seq]
    mono = CAMS[s["camera"]].get("imager") == "monochrome"
    if seq.startswith("hpwren_"):
        frames = hpwren_nights.read_frames(seq)
    else:
        stem = seq.split("#")[0]
        arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}
        frames = sorted(read_frames(arch[stem]), key=lambda f: f[1])  # by offset
    frames_points = []
    decoded = {}
    for epoch, offset, blob in frames:
        if offset - offset != 0:
            pass
        el, az = sun_altaz(epoch, s["lat"], s["lon"])
        if el > el_max:
            continue
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if keep_frames or not decoded:
            decoded[offset] = (epoch, img)
        pts = detect_points_adaptive(img) if mono else detect_points(img)
        frames_points.append((offset, pts))
    print(f"{seq}: {len(frames_points)}/{len(frames)} frames dark enough (sun el <= {el_max})")
    # A sky full of city glow or lit cloud texture yields thousands of "point sources" per
    # frame, and nearest-neighbor linking over that is quadratic: hpwren bl-n-mobo-c ran a
    # worker for 37 minutes on it. Stars are tens to a few hundred per frame, so give up
    # early and say why.
    counts = sorted(len(p) for _, p in frames_points)
    if counts and counts[len(counts) // 2] > MAX_POINTS_PER_FRAME:
        print(f"  median {counts[len(counts) // 2]} point sources per frame > {MAX_POINTS_PER_FRAME}: "
              f"not a star field, skipping")
        return [], decoded, s
    tracks = (link_tracks_predictive(frames_points) if mono
              else link_tracks(frames_points, max_step_px=max_step_px, max_gap_s=max_gap_s))
    good = moving_tracks(tracks, min_frames=min_frames, min_span_px=min_span_px)
    print(f"  {len(tracks)} raw tracks -> {len(good)} moving, persistent tracks")
    return good, decoded, s


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "20241021_PalomarRidge_hp-s-mobo-c"
    good, decoded, s = collect(seq)
    for t in sorted(good, key=lambda t: -len(t))[:10]:
        offsets = sorted(t)
        (x0, y0, a0), (x1, y1, a1) = t[offsets[0]], t[offsets[-1]]
        print(f"  track: {len(t)} frames, offsets {offsets[0]}..{offsets[-1]}, "
              f"({x0:.0f},{y0:.0f}) amp{a0} -> ({x1:.0f},{y1:.0f}) amp{a1}, "
              f"moved {np.hypot(x1-x0, y1-y0):.1f}px")
