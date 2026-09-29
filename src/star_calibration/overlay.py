"""Draw a solve on the block's own frame: the quickest way to see whether it is right.

Solved: matched tracks in thick green, each matched star's predicted arc under the fitted
pose in thin magenta on top of it, so agreement shows as magenta running inside green.
Unmatched tracks are thin grey. The same stars under the *published* pose (with the fitted
lens) run in orange, and a yellow arrow at the reference frame goes from where the published
table puts each star to where the fit has it -- the pose error, drawn.
Failed: every track the solver used in cyan, and the bright catalog stars' predicted arcs in
orange under the published pose with the shared lens -- what the camera should have seen if
the published table were right, which is usually the fastest way to see why it didn't solve.
A fit that fell below the acceptance cutoff is drawn like a solve, and labelled as such.

`draw` takes the result, the `Night` it was solved from, and one decoded frame (BGR, as
cv2.imdecode gives it), ideally the one nearest `reference_offset`. It returns the picture,
cropped to the band of sky with something drawn in it unless `full` is set;
where it goes is the caller's business (hpwren.calibrate writes the cache's overlays/).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import catalog as SG
from .fisheye import K1, K_RATIO, initial_k, project_fisheye
from .solve import Night, prune

WIDTH = 1600


def reference_offset(result: dict, night: Night) -> int:
    """The frame to draw on: the solve's reference frame, or the middle of the tracks."""
    offs = sorted({o for t in night.tracks for o in t}) or [0]
    return result.get("ref_offset", offs[len(offs) // 2])


def draw(result: dict, night: Night, frame: np.ndarray, full: bool = False) -> np.ndarray:
    r, tracks, c = result, night.tracks, night.cam
    offs = sorted({o for t in tracks for o in t}) or [0]
    ref = reference_offset(r, night)
    H, W = frame.shape[:2]
    canvas = cv2.convertScaleAbs(frame, alpha=2.0, beta=10)
    solved = "pose" in r   # draw the best fit even when it fell below the solve cutoff
    matched = {j: n for n, j in r.get("matches", {}).items()}
    keep = set(prune(tracks)) if tracks else set()

    y_max = 0.0
    for j, t in enumerate(tracks):
        pts = np.array([t[o][:2] for o in sorted(t)], np.int32)
        y_max = max(y_max, float(pts[:, 1].max()))
        if solved and j in matched:
            cv2.polylines(canvas, [pts], False, (60, 220, 60), 7, cv2.LINE_AA)
        elif solved:
            cv2.polylines(canvas, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
        elif j in keep:
            cv2.polylines(canvas, [pts], False, (230, 210, 60), 2, cv2.LINE_AA)

    k0 = initial_k(c, W)
    if solved:
        p = r["pose"]
        pose = (p["d_az"], p["d_pitch"], p["d_roll"], p["k_ratio"] * k0, p["k1"])
        names, color, width = list(matched.values()), (255, 60, 255), 2
    else:
        pose = (0.0, 0.0, 0.0, K_RATIO * k0, K1)
        names = [v["name"] for v in SG.visible_stars(c, night.t0 + ref, mag_limit=3.0,
                                                     fov_margin_deg=20)]
        color, width = (0, 140, 255), 2
    ep = night.t0 + np.array(offs, float)
    for n in names:
        alt, az = SG.altaz(*SG.STARS[n], ep, c["lat"], c["lon"])
        x, y = project_fisheye(c, az, alt, W, H, *pose)
        pts = np.c_[x * W, y * H]
        ins = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        if ins.sum() < 2:
            continue
        pts = pts[ins].astype(np.int32)
        y_max = max(y_max, float(pts[:, 1].max()))
        cv2.polylines(canvas, [pts], False, color, width, cv2.LINE_AA)
        mid = pts[len(pts) // 2]
        cv2.putText(canvas, n, (int(mid[0]) + 10, int(mid[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 2, cv2.LINE_AA)

    if solved:
        # The same stars under the published pose (no correction) with the fitted lens, so the
        # gap is the pose error alone; a yellow arrow at the reference epoch runs from where the
        # published table puts each star to where the fit (and the sky) has it.
        pub = (0.0, 0.0, 0.0, pose[3], pose[4])
        j_ref = offs.index(ref) if ref in offs else len(offs) // 2
        for n in names:
            alt, az = SG.altaz(*SG.STARS[n], ep, c["lat"], c["lon"])
            xp, yp = project_fisheye(c, az, alt, W, H, *pub)
            xf, yf = project_fisheye(c, az, alt, W, H, *pose)
            pts = np.c_[xp * W, yp * H]
            ins = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            if ins.sum() >= 2:
                cv2.polylines(canvas, [pts[ins].astype(np.int32)], False, (0, 140, 255), 2,
                              cv2.LINE_AA)
            if not np.isfinite([xp[j_ref], yp[j_ref], xf[j_ref], yf[j_ref]]).all():
                continue
            a = (int(xp[j_ref] * W), int(yp[j_ref] * H))
            b = (int(xf[j_ref] * W), int(yf[j_ref] * H))
            if 0 <= a[0] < W and 0 <= a[1] < H:
                y_max = max(y_max, float(a[1]))
            cv2.arrowedLine(canvas, a, b, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.08)

    if not full:   # the band of sky with something drawn in it, unless asked for the whole frame
        canvas = canvas[: int(min(H, max(0.55 * H, y_max + 120)))]
    if solved:
        p = r["pose"]
        tag = "SOLVED" if r["status"] == "solved" else "BELOW CUTOFF, best fit"
        line2 = (f"{tag}  {r['n_stars']} stars, median {r['median_px']:.2f}px  "
                 f"d_az {p['d_az']:+.2f}  d_pitch {p['d_pitch']:+.2f}  d_roll {p['d_roll']:+.2f}  "
                 f"k {p['k_ratio']:.3f}x  k1 {p['k1']:+.3f}   green=track, magenta=fitted, "
                 f"orange=published pose, yellow=published->fitted")
    else:
        line2 = (f"FAILED  {r['reason']}   cyan=tracks used, "
                 f"orange=bright stars at published pose + shared lens")
    line1 = (f"{night.label or night.camera}   {c.get('imager')}   "
             f"{r.get('n_tracks', 0)}/{r.get('n_tracks_raw', 0)} tracks")
    banner = np.zeros((90, canvas.shape[1], 3), np.uint8)
    for k, text in enumerate((line1, line2)):
        cv2.putText(banner, text, (14, 36 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
                    (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([banner, canvas])
    scale = WIDTH / out.shape[1]
    return cv2.resize(out, (WIDTH, int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def write(image: np.ndarray, dest) -> None:
    """JPEG at quality 88: the overlays' format."""
    cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
