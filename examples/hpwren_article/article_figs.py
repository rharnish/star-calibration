"""Figures and numbers for the HPWREN article (article.ipynb).

Each figure is one function that returns something the notebook displays: an
IPython Image (camera frames, drawn on with OpenCV), a matplotlib figure, or Markdown
(tables and sentences whose numbers come from the cache). The notebook's code cells
are one-liners calling these, so the exported article carries only prose and outputs.

Solving is cached per (camera, day) for the kernel's lifetime: a figure tweak re-runs
one cell, not the solves. Reads frames from the shared cache ($HPWREN_CACHE); fetches
only what isn't there yet. The CDN keeps frames for about 90 days, so re-running a figure
from nights older than that needs them already in the cache.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))
import terrain  # noqa: E402  examples/terrain.py

from star_calibration import catalog, overlay, pole  # noqa: E402
from star_calibration.cross_night import pairs  # noqa: E402
from star_calibration.fisheye import K1, K_RATIO, initial_k, theta_limit_deg  # noqa: E402
from star_calibration.intrinsics import lookup  # noqa: E402
from star_calibration.solve import Night, solve_wide  # noqa: E402
from star_calibration.tracks import clean, collect, detect_points  # noqa: E402
from star_calibration.hpwren import cache_dir, cameras, intrinsics_path, nights  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_W = 3072                 # px: width of every frame figure: the cameras' own, so a reader
                             # who opens one full size sees the frame as recorded
FIG_DPI = 200                # matplotlib figures, about 2000-2800 px wide
plt.rcParams["figure.dpi"] = FIG_DPI
BIG = {False: (1, 80), True: (2, 120)}   # first-pass blob sizes: colour, monochrome

CYAN, ORANGE, MAGENTA = (255, 230, 0), (0, 150, 255), (255, 0, 255)
GOLD, GREEN, WHITE = (0, 200, 255), (90, 255, 90), (255, 255, 255)


# ---------------------------------------------------------------- display helpers

def jpeg(img_bgr, width=OUT_W, quality=85) -> Image:
    h, w = img_bgr.shape[:2]
    if w > width:
        img_bgr = cv2.resize(img_bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return Image(data=buf.tobytes(), format="jpeg")


def jpeg_fig(fig, quality=88) -> Image:
    """A matplotlib figure as JPEG: the photographic ones run to megabytes as PNG."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=FIG_DPI, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": quality})
    plt.close(fig)
    return Image(data=buf.getvalue(), format="jpeg")


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- solving

def read_block(name, day, q=1):
    nights.fetch(name, day, q=q, n_frames=90 if q == 1 else 3)
    d = cache_dir() / "nights" / name / f"{day}_Q{q}"
    paths = sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))
    epochs = [int(p.stem) for p in paths]
    t0 = epochs[len(epochs) // 2]
    return t0, [(e, e - t0, p.read_bytes()) for e, p in zip(epochs, paths)]


def decode(blob):
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


@lru_cache(maxsize=None)
def solve(name, day):
    """Track and solve one camera's first 90 minutes of `day` in two passes, as the article
    describes: big blobs first (the brightest stars count) for the pose, then stars of the
    usual size anywhere above the DEM skyline as that pose places it."""
    cam = cameras()[name]
    t0, frames = read_block(name, day)
    mono = cam["imager"] == "monochrome"
    intr = json.loads(intrinsics_path().read_text())

    def track(detect):
        linked, decoded = collect(frames, lat=cam["lat"], lon=cam["lon"], mono=mono,
                                  max_gap_s=240 if mono else 300, detect=detect, keep_frames=False)
        return clean(linked, banner_px=60)[0], decoded

    tracks1, decoded = track({"area_range": BIG[mono]})
    H, W = next(iter(decoded.values()))[1].shape[:2]
    cx, cy = lookup(intr, name, W, H, t0)
    cam_c = {**cam, "cx": cx, "cy": cy}
    night1 = Night(camera=name, cam=cam_c, t0=t0, tracks=tracks1, W=W, H=H)
    trace = []
    first = solve_wide(night1, trace=trace)

    mask = terrain.sky_mask(cam_c, first["pose"], W, H, terrain.skyline(cam))
    tracks2, _ = track({"sky_mask": mask})
    night = Night(camera=name, cam=cam_c, t0=t0, tracks=tracks2, W=W, H=H, label=f"{name} {day} Q1")
    r = solve_wide(night)
    return dict(name=name, day=day, cam=cam, cam_c=cam_c, frames=frames, t0=t0, W=W, H=H,
                night1=night1, first=first, trace=trace, mask=mask, night=night, r=r)


def last_frame(name, day):
    """(epoch, image) of the block's last frame, about 01:29 local."""
    _, frames = read_block(name, day)
    epoch, _, blob = frames[-1]
    return epoch, decode(blob)


# ---------------------------------------------------------------- the pole (step 3)

def cam_to_px(v, W, H, k, k1, cx, cy):
    """Camera-frame unit vectors (right, up, boresight) to pixels; NaN past the lens limit."""
    v = np.atleast_2d(v)
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    s = np.where(np.sin(th) > 1e-9, np.sin(th), 1.0)
    r = k * th * (1 + k1 * th ** 2)
    x, y = W / 2 + cx + r * v[:, 0] / s, H / 2 + cy - r * v[:, 1] / s
    bad = th > math.radians(min(92.0, theta_limit_deg(k1)))
    return np.where(bad, np.nan, x), np.where(bad, np.nan, y)


def ring(p, rho_deg, n=720):
    p = p / np.linalg.norm(p)
    a = np.cross(p, [0, 0, 1.0]); a /= np.linalg.norm(a)
    b = np.cross(p, a)
    t = np.linspace(0, 2 * np.pi, n)
    r = math.radians(rho_deg)
    return np.cos(r) * p + np.sin(r) * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def polyline(img, x, y, ox, oy, col, w):
    pts = []
    for xi, yi in list(zip(x, y)) + [(np.nan, np.nan)]:
        if np.isfinite(xi) and np.isfinite(yi):
            pts.append((int(xi + ox), int(yi + oy)))
            continue
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, col, w, cv2.LINE_AA)
        pts = []


def pole_fit(s):
    """The pole from the first pass's trails alone, and its distance from the solved pose's."""
    c, W, H = s["cam_c"], s["W"], s["H"]
    k = K_RATIO * initial_k(c, W)
    est = pole.estimate(s["night1"].tracks, W, H, k, K1, c["cx"], c["cy"])
    P = s["r"]["pose"]
    p_true = pole.pole_in_cam(c, P["d_az"], P["d_pitch"], P["d_roll"])
    err = math.degrees(math.acos(np.clip(est["p_hat"] @ p_true, -1, 1)))
    return est, k, err


def pole_figure(name, day, base="last", pad=(40, 420, 40, 40), rhos=range(10, 61, 10), arrows=False):
    """Trails (green) and circles about the fitted pole (gold) on a canvas big enough to
    show the pole. `base` is "last" (the last frame) or "stack" (the night's maximum).
    `arrows` draws the measured and predicted motion (motion_arrows) in place of the trails."""
    s = solve(name, day)
    c, W, H = s["cam_c"], s["W"], s["H"]
    if base == "stack":
        img = None
        for _, _, blob in s["frames"]:
            im = decode(blob)
            img = im if img is None else np.maximum(img, im)
        img = cv2.convertScaleAbs(img, alpha=2.0)
    else:
        img = cv2.convertScaleAbs(decode(s["frames"][-1][2]), alpha=1.6)
    est, k, _ = pole_fit(s)
    pl, pt, pr, pb = pad
    canvas = np.full((H + pt + pb, W + pl + pr, 3), 18, np.uint8)
    canvas[pt:pt + H, pl:pl + W] = img
    cv2.rectangle(canvas, (pl, pt), (pl + W - 1, pt + H - 1), (90, 90, 90), 2)
    for rho in rhos:
        x, y = cam_to_px(ring(est["p_hat"], rho), W, H, k, K1, c["cx"], c["cy"])
        polyline(canvas, x, y, pl, pt, GOLD, 2)
    if arrows:
        motion_arrows(canvas, s, est, k, pl, pt)
    else:
        for t in s["night1"].tracks:
            xy = np.array([t[o][:2] for o in sorted(t)])
            cv2.polylines(canvas, [np.int32(xy + [pl, pt])], False, GREEN, 5, cv2.LINE_AA)
    x, y = cam_to_px(est["p_hat"], W, H, k, K1, c["cx"], c["cy"])
    X, Y = int(x[0]) + pl, int(y[0]) + pt
    cv2.drawMarker(canvas, (X, Y), WHITE, cv2.MARKER_CROSS, 90, 5, cv2.LINE_AA)
    cv2.circle(canvas, (X, Y), 30, WHITE, 4, cv2.LINE_AA)
    return jpeg(canvas)


def heading_scan(name, day):
    """Step 4's search, drawn against what is actually scanned: the turn about the fitted pole,
    measured from the best match. Top: the best score in each 1° of turn. Bottom: the
    azimuth, tilt and roll that each turn gives, since the turn moves all three."""
    s = solve(name, day)
    e = next(e for e in s["trace"] if e["stage"] == "scan" and "psi" in e)
    psi, sc, poses = np.asarray(e["psi"]), np.asarray(e["scores"]), np.asarray(e["poses"])
    az = (s["cam"]["az"] + poses[:, 0]) % 360
    tilt, roll = s["cam"]["pitch"] + poses[:, 1], s["cam"]["roll"] + poses[:, 2]
    i = int(np.argmax(sc))
    best_az = az[i]
    t = (psi - psi[i] + 180) % 360 - 180
    o = np.argsort(t)
    t, sc, az, tilt, roll = t[o], sc[o], az[o], tilt[o], roll[o]
    b = np.floor(t).astype(int)
    bins = np.arange(-180, 180)
    best = np.array([sc[b == k].max() if (b == k).any() else 0.0 for k in bins])
    k = int(np.argmax(best))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [1, 1.15]})
    a1.bar(bins + 0.5, best, width=1.0, color="#9aa4b1")
    a1.bar(bins[k] + 0.5, best[k], width=1.6, color="#c2410c")
    a1.annotate(f"azimuth {best_az:.1f}°", (bins[k] + 0.5, best[k]), xytext=(8, -4),
                textcoords="offset points", va="top", fontsize=10, color="#c2410c")
    a1.set_ylabel("match score")
    azp = az.astype(float).copy()
    azp[np.r_[False, np.abs(np.diff(az)) > 180]] = np.nan     # break the line where azimuth wraps
    for y, label, colour in ((azp, "azimuth", "#1f4e79"), (tilt, "tilt", "#2f855a"),
                             (roll, "roll", "#9b2c86")):
        a2.plot(t, y, color=colour, lw=1.6)
        a2.annotate(label, (180, y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=colour, annotation_clip=False)
    a2.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    a2.axvline(0, color="#c2410c", lw=0.8, ls=":")
    a2.set_ylabel("angle it gives (degrees)")
    a2.set_yticks(range(-180, 361, 90))
    a2.set_xlim(-180, 180)
    a2.set_xticks(range(-180, 181, 45))
    a2.set_xlabel("turn about the pole, from the best match (degrees)")
    for a in (a1, a2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    plt.close(fig)   # shown once, as the return value
    return fig


# ---------------------------------------------------------------- the solve on its frame

def solve_overlay(name, day, full=False, first_frac=0.35):
    """Matched trails (green), predicted arcs (magenta), the nominal pose's stars (orange).
    `full` draws the whole frame, with everything clipped at the terrain skyline (as the
    solved pose places it, drawn in cyan, the ground dimmed) and a faint line where the first pass's search, the top
    `first_frac` of the frame, stops."""
    s = solve(name, day)
    ref = overlay.reference_offset(s["r"], s["night"])
    _, _, blob = min(s["frames"], key=lambda f: abs(f[1] - ref))
    frame = decode(blob)
    width0, overlay.WIDTH = overlay.WIDTH, OUT_W   # the library draws at 1600
    try:
        out = overlay.draw(s["r"], s["night"], frame, full=full)
    finally:
        overlay.WIDTH = width0
    if not full:
        return jpeg(out)
    W, H = s["W"], s["H"]
    ow = out.shape[1]
    oh = round(H * ow / W)
    u = ow / 1600                                             # marks were sized at 1600 px
    top = out.shape[0] - oh                                    # the banner overlay.draw adds
    sky = terrain.sky_mask(s["cam_c"], s["r"]["pose"], W, H, terrain.skyline(s["cam"]), margin_deg=0.0)
    sky = cv2.resize(sky.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
    plain = cv2.resize(cv2.convertScaleAbs(frame, alpha=2.0, beta=10), (ow, oh), interpolation=cv2.INTER_AREA)
    body = out[top:]
    body[~sky] = (plain[~sky] * 0.6).astype(np.uint8)        # the ground, dimmed
    az, el = terrain.skyline(s["cam"])
    x, y = terrain.project(s["cam_c"], az, el, W, H, s["r"]["pose"])
    ok = np.isfinite(x) & np.isfinite(y)
    xy = np.c_[x, y][ok] * (ow / W)
    xy = xy[np.argsort(xy[:, 0])]
    xy = xy[(xy[:, 0] >= -50) & (xy[:, 0] <= ow + 50)]
    cv2.polylines(body, [np.int32(xy)], False, (255, 255, 0), round(2 * u), cv2.LINE_AA)   # the skyline, cyan
    y = round(first_frac * oh)
    line = body.copy()
    for x in range(0, ow, round(24 * u)):
        cv2.line(line, (x, y), (x + round(12 * u), y), (255, 255, 255), round(u), cv2.LINE_AA)
    cv2.putText(line, f"first pass: top {first_frac:.0%}", (ow - round(230 * u), y - round(8 * u)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * u, (255, 255, 255), round(u), cv2.LINE_AA)
    out[top:] = cv2.addWeighted(line, 0.55, body, 0.45, 0)
    return jpeg(out)


def search_region(name, day):
    """The second pass's search region: red is left out; dashed is where the top-35% search stops."""
    s = solve(name, day)
    img = cv2.convertScaleAbs(decode(s["frames"][-1][2]), alpha=2.5, beta=10)
    out = ~s["mask"]
    img[out] = (img[out] * 0.45 + np.array([0, 0, 110])).astype(np.uint8)
    y35 = int(0.35 * s["H"])
    for x in range(0, s["W"], 60):
        cv2.line(img, (x, y35), (x + 30, y35), (0, 255, 255), 4)
    return jpeg(img)


# ---------------------------------------------------------------- constellations

FIGURES = {
    "Orion": [["Betelgeuse", "Lam Ori", "Bellatrix"], ["Betelgeuse", "Alnitak", "Saiph"],
              ["Bellatrix", "Mintaka", "Rigel"], ["Mintaka", "Alnilam", "Alnitak"],
              ["Bellatrix", "Pi-3 Ori"], ["Pi-5 Ori", "Pi-4 Ori", "Pi-3 Ori"]],
    "Taurus": [["Alnath", "Eps Tau", "Del-1 Tau", "Gam Tau", "The-2 Tau", "Aldebaran", "Zet Tau"],
               ["Gam Tau", "Lam Tau", "Omi Tau"]],
    "Gemini": [["Castor", "Pollux"], ["Castor", "Eps Gem", "Mu Gem", "Eta Gem"],
               ["Pollux", "Del Gem", "Lam Gem", "Xi Gem"], ["Del Gem", "Alhena"]],
    "Auriga": [["Capella", "Menkalinan", "The Aur", "Alnath", "Hassaleh", "Capella"]],
    "Canis Minor": [["Procyon", "Gomeisa"]],
    "Canis Major": [["Mirzam", "Sirius", "Omi-2 CMa", "Wezen", "Adhara"], ["Wezen", "Aludra"]],
    "Lepus": [["Arneb", "Mu Lep", "Eps Lep", "Nihal", "Arneb"], ["Nihal", "Gam Lep", "Del Lep"]],
}
LABELS = {"Betelgeuse", "Rigel", "Aldebaran", "Castor", "Pollux", "Capella", "Procyon", "Sirius"}
PLEIADES = "Alcyone"
R_STAR = 16


def figure_pixels(s, epoch, pose):
    """Each figure star's pixel through `pose`, or through the camera table's pose if None."""
    cam = s["cam_c"] if pose else s["cam"]
    names = sorted({n for segs in FIGURES.values() for seg in segs for n in seg} | {PLEIADES})
    alt_az = [catalog.star_altaz(n, epoch, cam["lat"], cam["lon"], cam["elev"]) for n in names]
    alt, az = np.array([a for a, _ in alt_az]), np.array([z for _, z in alt_az])
    x, y = terrain.project(cam, az, alt, s["W"], s["H"], pose)
    return {n: (float(xi), float(yi)) for n, xi, yi, a in zip(names, x, y, alt)
            if np.isfinite(xi) and np.isfinite(yi) and a > -2}


def _segment(img, p, q, colour, width, solid, dash=18):
    p, q = np.array(p), np.array(q)
    d = np.linalg.norm(q - p)
    if d <= 2 * R_STAR:
        return
    p, q = p + (q - p) * R_STAR / d, q - (q - p) * R_STAR / d
    n = 1 if solid else max(1, int((d - 2 * R_STAR) / dash))
    for i in range(0, n, 1 if solid else 2):
        a, b = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), colour, width, cv2.LINE_AA)


def draw_figures(img, pix, colour, width, solid=True, labels=True):
    for con, segs in FIGURES.items():
        placed = []
        for seg in segs:
            for a, b in zip(seg[:-1], seg[1:]):
                if a in pix and b in pix:
                    _segment(img, pix[a], pix[b], colour, width, solid)
            placed += [pix[n] for n in seg if n in pix]
        for n in {n for seg in segs for n in seg} & set(pix):
            cv2.circle(img, tuple(int(v) for v in pix[n]), R_STAR, colour, 2, cv2.LINE_AA)
        if labels and placed:
            x, y = np.mean([p[0] for p in placed]), max(p[1] for p in placed) + 60
            cv2.putText(img, con, (int(x) - 60, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4, cv2.LINE_AA)
    if labels:
        for n in LABELS & set(pix):
            cv2.putText(img, n, (int(pix[n][0]) + 20, int(pix[n][1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
        if PLEIADES in pix:
            cv2.circle(img, tuple(int(v) for v in pix[PLEIADES]), 40, colour, 3, cv2.LINE_AA)
            cv2.putText(img, "Pleiades", (int(pix[PLEIADES][0]) + 46, int(pix[PLEIADES][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)


def night_canvas(frame, gain=2.5):
    # Brightened, with each star swollen to 5 px so it survives downscaling.
    return cv2.dilate(cv2.convertScaleAbs(frame, alpha=gain, beta=10), np.ones((5, 5), np.uint8))


SIDEREAL_LAG_S = 86400 - 86164.0905   # a star reaches the same place this much earlier each day


def drift(img, s, epoch, since_epoch, con="Orion", colour=WHITE, alpha=0.45):
    """Faintly, where `con` stood at the same clock time on the night of `since_epoch`, and the
    path each of its stars took from there night by night. The stars return to the same place
    ~3 min 56 s earlier each day, so at a fixed clock time a star creeps along the arc it
    travels in one night: the path is that arc, through the solved pose."""
    days = (epoch - since_epoch) / 86400
    then = epoch - days * SIDEREAL_LAG_S
    layer = img.copy()
    names = sorted({n for seg in FIGURES[con] for n in seg})
    cam = s["cam_c"]
    for n in names:
        ts = np.linspace(then, epoch, 120)
        alt_az = [catalog.star_altaz(n, t, cam["lat"], cam["lon"], cam["elev"]) for t in ts]
        alt, az = np.array([a for a, _ in alt_az]), np.array([z for _, z in alt_az])
        x, y = terrain.project(cam, az, alt, s["W"], s["H"], s["r"]["pose"])
        ok = np.isfinite(x) & np.isfinite(y) & (alt > -2)
        pts = np.c_[x[ok], y[ok]].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(layer, [pts], False, colour, 2, cv2.LINE_AA)
    pix = {k: v for k, v in figure_pixels(s, then, s["r"]["pose"]).items() if k in names}
    for seg in FIGURES[con]:
        for a, b in zip(seg[:-1], seg[1:]):
            if a in pix and b in pix:
                _segment(layer, pix[a], pix[b], colour, 2, solid=False)
    for p in pix.values():
        cv2.circle(layer, tuple(int(v) for v in p), R_STAR, colour, 2, cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0), pix


def constellations(name, solve_day, day=None, nominal=None, drift_con=None):
    """Stick figures on `day`'s 01:29 frame through the pose solved on `solve_day` (cyan), and,
    by default only on the night solved, through the camera table's pose (dashed orange).
    drift_con: also, faintly, where that constellation stood at 01:29 on `solve_day` and the
    path it crept along to `day` (`drift`)."""
    s = solve(name, solve_day)
    day = day or solve_day
    nominal = (day == solve_day) if nominal is None else nominal
    epoch, frame = last_frame(name, day)
    img = night_canvas(frame)
    if drift_con and day != solve_day:
        img, pix = drift(img, s, epoch, last_frame(name, solve_day)[0], drift_con)
        if pix:
            from datetime import datetime
            when = datetime.strptime(solve_day, "%Y%m%d").strftime("%-d %b")
            x, y = min(p[0] for p in pix.values()), np.mean([p[1] for p in pix.values()])
            cv2.putText(img, when, (int(x) - 230, int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (200, 200, 200), 3, cv2.LINE_AA)
    if nominal:
        draw_figures(img, figure_pixels(s, epoch, None), ORANGE, 2, solid=False, labels=False)
    draw_figures(img, figure_pixels(s, epoch, s["r"]["pose"]), CYAN, 3)
    return jpeg(img)


def closeups(name, solve_day, panels, follow="Aldebaran"):
    """Full-resolution crops on several nights, all through the `solve_day` pose.
    `panels` is [(day, title), ...]; each crop follows the star `follow`."""
    s = solve(name, solve_day)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.6))
    for ax, (day, title) in zip(np.atleast_1d(axes), panels):
        epoch, frame = last_frame(name, day)
        img = cv2.dilate(cv2.convertScaleAbs(frame, alpha=3.0, beta=10), np.ones((3, 3), np.uint8))
        pix = figure_pixels(s, epoch, s["r"]["pose"])
        draw_figures(img, pix, CYAN, 2, labels=False)
        for n in ("Aldebaran", "Betelgeuse", "Rigel", "Bellatrix", "Alnath", "Capella"):
            if n in pix:
                cv2.putText(img, n, (int(pix[n][0]) + 20, int(pix[n][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, CYAN, 2, cv2.LINE_AA)
        cx, cy = (int(v) for v in pix[follow])
        x0 = min(max(0, cx - 600), s["W"] - 1200)
        y0 = min(max(0, cy - 450), s["H"] - 900)
        ax.imshow(rgb(img[y0:y0 + 900, x0:x0 + 1200]))
        ax.set_title(title, loc="left")
        ax.axis("off")
    fig.tight_layout()
    plt.close(fig)   # shown once, as the return value
    return fig


def star_offsets(name, solve_day, day, mag_limit=3.5, radius=10):
    """Catalog stars brighter than `mag_limit` that the `solve_day` pose puts in the sky region
    of `day`'s 01:29 frame: (predicted, found within `radius` px, median px, median (dx, dy))."""
    s = solve(name, solve_day)
    epoch, frame = last_frame(name, day)
    cam, pose, mask = s["cam_c"], s["r"]["pose"], s["mask"]
    pts = np.array([p[:2] for p in detect_points(frame, sky_mask=mask)])
    stars = [r["name"] for r in catalog.CATALOG if r["mag"] <= mag_limit]
    alt_az = [catalog.star_altaz(n, epoch, cam["lat"], cam["lon"], cam["elev"]) for n in stars]
    alt, az = np.array([a for a, _ in alt_az]), np.array([z for _, z in alt_az])
    x, y = terrain.project(cam, az, alt, frame.shape[1], frame.shape[0], pose)
    H, W = mask.shape
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    ok[ok] = mask[y[ok].astype(int), x[ok].astype(int)]
    d, j = cKDTree(pts).query(np.c_[x[ok], y[ok]], distance_upper_bound=radius)
    hit = np.isfinite(d)
    v = pts[j[hit]] - np.c_[x[ok], y[ok]][hit]
    return int(ok.sum()), int(hit.sum()), float(np.median(d[hit])), np.median(v, axis=0)


# ---------------------------------------------------------------- terrain

@lru_cache(maxsize=None)
def _crests(name, solve_day, day_q6):
    s = solve(name, solve_day)
    _, frames = read_block(name, day_q6, q=6)
    daylit = decode(frames[0][2])
    az, el, km, sky = terrain.crests(s["cam"], min_km=0.5)
    views = {"nominal": (terrain.project(s["cam"], az, el, s["W"], s["H"]), ORANGE),
             "solved": (terrain.project(s["cam_c"], az, el, s["W"], s["H"], s["r"]["pose"]), MAGENTA)}
    return daylit, sky, views


def _draw_crests(img, x, y, sky, colour, r=3):
    ok = np.isfinite(x) & np.isfinite(y)
    for xi, yi, si in zip(x[ok], y[ok], sky[ok]):
        cv2.circle(img, (int(xi), int(yi)), r, (0, 255, 255) if si else colour, -1)


def crests(name, solve_day, day_q6=None, rows=(800, 1500)):
    """Terrain crests on the afternoon frame: nominal pose (left), solved pose (right)."""
    daylit, sky, views = _crests(name, solve_day, day_q6 or solve_day)
    W = daylit.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for ax, (label, ((x, y), colour)) in zip(axes, views.items()):
        img = daylit.copy()
        _draw_crests(img, x, y, sky, colour)
        ax.imshow(rgb(img[rows[0]:rows[1]]), extent=(0, W, rows[1], rows[0]))
        ax.set_title(f"{label} orientation", loc="left")
        ax.axis("off")
    fig.tight_layout()
    plt.close(fig)   # shown once, as the return value
    return fig


def crest_windows(name, solve_day, windows, day_q6=None):
    """Close-ups at (x, y) centres: nominal on top, solved below."""
    daylit, sky, views = _crests(name, solve_day, day_q6 or solve_day)
    fig, axes = plt.subplots(2, len(windows), figsize=(16, 6.5))
    for row, (label, ((x, y), colour)) in enumerate(views.items()):
        for col, (x0, y0) in enumerate(windows):
            ax = axes[row, col]
            ax.imshow(rgb(daylit))
            m = (x > x0 - 250) & (x < x0 + 250)
            c = np.array(colour[::-1]) / 255
            ax.plot(x[m & ~sky], y[m & ~sky], ".", ms=2, color=c)
            ax.plot(x[m & sky], y[m & sky], ".", ms=2.5, color="yellow")
            ax.set_xlim(x0 - 250, x0 + 250)
            ax.set_ylim(y0 + 110, y0 - 110)
            ax.set_title(label, loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    plt.close(fig)   # shown once, as the return value
    return fig


def crest_shift(name, solve_day, day_q6=None):
    """The vertical shift (px; negative is up) at which each pose's crests best sit on edges."""
    daylit, sky, views = _crests(name, solve_day, day_q6 or solve_day)
    edge = terrain.edge_map(daylit)
    shifts = np.arange(-30, 31)
    return {k: int(shifts[terrain.edge_score(edge, x, y, shifts).argmax()]) for k, ((x, y), _) in views.items()}


# ---------------------------------------------------------------- the network

@lru_cache(maxsize=None)
def network():
    """The v0.4.0 network run's 316 CDN blocks as `calibrate solve` last left them, and each
    solve's nearest sibling night under 30 days away. The blocks are pinned to the keys of
    static/cloud.json, so nights solved since (buff-n/w in mid-October, for the panorama)
    don't move the article's numbers."""
    from star_calibration.hpwren.calibrate import summary
    indexed = nights.sequences()
    run = json.loads((HERE / "static" / "cloud.json").read_text())
    results = [r for r in summary() if "source" not in r and r["seq"] in indexed and r["seq"] in run]
    t0 = {k: v["t0"] for k, v in nights.sequences().items()}
    ps = [p for p in pairs(results, t0, cameras()) if p["days_apart"] < 30]
    return results, ps


def network_numbers() -> dict:
    results, ps = network()
    ok = [r for r in results if r["status"] == "solved"]
    fail = [r for r in results if r["status"] != "solved"]
    reason = lambda r: r.get("reason") or ""
    sep = np.array([p["sep_deg"] for p in ps])
    days = sorted(r["seq"].split("_")[1] for r in results)
    k = np.array([r["pose"]["k_ratio"] for r in ok])
    shared = k[k > 0.85]
    return dict(
        nights=len(results), cameras=len({r["camera"] for r in results}),
        solved=len(ok), solved_cameras=len({r["camera"] for r in ok}),
        stars_median=float(np.median([r["n_stars"] for r in ok])),
        px_median=float(np.median([r["median_px"] for r in ok])),
        deg_median=float(np.median([r["median_px"] * 180 / math.pi
                                    / (r["pose"]["k_ratio"] * initial_k(cameras()[r["camera"]], r["W"]))
                                    for r in ok])),
        agree_median=float(np.median(sep)), agree_p90=float(np.percentile(sep, 90)),
        within_005=float(np.mean(sep <= 0.05)), n_pairs=len(sep),
        failed=len(fail),
        dark=sum(reason(r).startswith("only ") for r in fail),
        no_rotation=sum(reason(r).startswith("no coherent") for r in fail),
        too_few=sum(not reason(r).startswith(("only ", "no coherent")) for r in fail),
        first=days[0], last=days[-1],
        k_shared_median=float(np.median(shared)),
        k_within_1pc=float(np.mean(np.abs(shared / np.median(shared) - 1) <= 0.01)),
        k_n_wide=int((k <= 0.85).sum()),
    )


def network_table() -> Markdown:
    n = network_numbers()
    rows = [
        ("Nights tried", f"{n['nights']}, on {n['cameras']} cameras"),
        ("Calibrated", f"{n['solved']} nights, on {n['solved_cameras']} cameras"),
        ("Stars matched per night", f"{n['stars_median']:.0f} (median)"),
        ("Fit to the star trails", f"{n['px_median']:.2f} pixels (median), about {n['deg_median']:.3f}°"),
        ("Two nights of one camera, under 30 days apart",
         f"agree to {n['agree_median']:.3f}° (median); {n['within_005'] * 100:.0f}% within 0.05°"),
    ]
    md = "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    return Markdown(md)


def network_lead() -> Markdown:
    n = network_numbers()
    month = {"07": "July", "08": "August", "09": "September", "10": "October"}
    return Markdown(
        f"Of {n['nights']} nights tried, {n['solved']} gave a calibration, and {n['solved_cameras']} "
        f"of the {n['cameras']} cameras tried have at least one. The nights are 90-minute blocks "
        f"from HPWREN's public archive, {month[n['first'][4:6]]} to {month[n['last'][4:6]]} 2026.")


CLEAR = 10   # forecast cloud cover (%) at or under which a night counts as clear


@lru_cache(maxsize=None)
def moon_rows():
    """Each block with the moon at its reference frame (moon.conditions) and the forecast
    cloud there (static/cloud.json, from scripts/weather_fill.py)."""
    from star_calibration import moon
    results, _ = network()
    cloud = json.loads((HERE / "static" / "cloud.json").read_text())
    seqs = nights.sequences()
    out = []
    for r in results:
        cam = cameras()[r["camera"]]
        t = seqs[r["seq"]]["t0"] + r.get("ref_offset", 2700)
        out.append({**r, **moon.conditions(t, cam["lat"], cam["lon"]), "t": t, "cloud": cloud[r["seq"]]})
    return out


def moon_numbers() -> dict:
    """Moon up against moon down. Clear nights: how many calibrated. Calibrated nights: stars
    matched as a ratio to the same camera's moonless median, and pose agreement with another
    moonless night of the same camera 3-30 days apart (every such pair, so both columns span
    the same spread of dates)."""
    import itertools
    from collections import defaultdict
    from star_calibration.cross_night import disagreement
    rows = moon_rows()
    up = lambda r: r["moon_up"]
    out = {}
    for key, want in (("down", False), ("up", True)):
        clear = [r for r in rows if up(r) == want and r["cloud"] <= CLEAR]
        out[key] = dict(clear=len(clear), solved=sum(r["status"] == "solved" for r in clear),
                        illum_median=float(np.median([r["illum"] for r in clear])),
                        el_median=float(np.median([r["moon_el"] for r in clear])),
                        ratio=[], sep=[])
    by = defaultdict(list)
    for r in rows:
        if r["status"] == "solved":
            by[r["camera"]].append(r)
    for cam, rs in by.items():
        dark = [r for r in rs if not up(r)]
        if not dark:
            continue
        med = np.median([r["n_stars"] for r in dark])
        for r in rs:
            if up(r) and len(dark) >= 1 or not up(r) and len(dark) >= 2:
                out["up" if up(r) else "down"]["ratio"].append(r["n_stars"] / med)
        for a, b in itertools.combinations(rs, 2):
            if up(a) and up(b) or not 2.5 <= abs(a["t"] - b["t"]) / 86400 <= 30:
                continue
            key = "up" if up(a) or up(b) else "down"
            out[key]["sep"].append(disagreement(cameras()[cam], a["pose"], b["pose"])[0])
    for v in out.values():
        v["n_pairs"], v["n_cal"] = len(v["sep"]), len(v["ratio"])
        v["ratio"], v["sep"] = float(np.median(v["ratio"])), float(np.median(v["sep"]))
    return out


def moon_table() -> Markdown:
    m = moon_numbers()
    d, u = m["down"], m["up"]
    pc = lambda v: f"{v['solved']} of {v['clear']} ({100 * v['solved'] / v['clear']:.0f}%)"
    rows = [
        ("Clear nights calibrated", pc(d), pc(u)),
        ("Stars matched, against the same camera's moonless nights", "1× (the reference)", f"{u['ratio']:.2f}×"),
        ("Orientation against a moonless night of the same camera, 3–30 days apart",
         f"{d['sep']:.3f}° (median)", f"{u['sep']:.3f}° (median)"),
    ]
    md = "| | Moon down | Moon up |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return Markdown(md)


def nights_numbers() -> dict:
    """The split-half study (static/split_half.json, from scripts/split_half.py):
    per number of nights, the median and 90th percentile error of the optical centre (px on the
    sensor) and of the pointing that the lens and centre leave once each night's pose is fitted
    (px), over the stars (top 60% of the frame) and below them (bottom third). The yardsticks:
    each night's star fit (median residual) and two nights' pointing agreement, in px."""
    rows = json.loads((HERE / "static" / "split_half.json").read_text())
    n = network_numbers()
    deg_per_px = n["deg_median"] / n["px_median"]
    out = {"star_fit_px": n["px_median"], "night_to_night_px": n["agree_median"] / deg_per_px,
           "deg_per_px": deg_per_px, "by_n": {}}
    for k in sorted({r["n"] for r in rows}):
        s = [r for r in rows if r["n"] == k]
        out["by_n"][k] = {"pairs": len(s), "cameras": len({r["camera"] for r in s}),
                          **{f"{q}_{f}": float(fn([r[q] for r in s]))
                             for q in ("centre", "shape_top", "shape_low")
                             for f, fn in (("med", np.median), ("p90", lambda v: np.percentile(v, 90)))}}
    return out


def nights_figure(max_n=3):
    """Left: the optical centre's error against nights used. Right: the pointing error the
    lens and centre leave, against the per-night star fit and the night-to-night agreement."""
    m = nights_numbers()
    ns = [k for k in m["by_n"] if k <= max_n]
    B = [m["by_n"][k] for k in ns]
    BLUE, ORANGE_ = "#2a78d6", "#eb6834"
    INK, MUTED = "#2b2b2b", "#77766f"
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bbb")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(ns)
        ax.set_xlim(0.7, max_n + 0.3)
        ax.set_xlabel("nights used", color=INK, fontsize=10)

    def series(ax, q, col, label, dx=0.0):
        x = np.array(ns) + dx
        med = [r[f"{q}_med"] for r in B]
        p90 = [r[f"{q}_p90"] for r in B]
        ax.vlines(x, med, p90, color=col, lw=1.2, alpha=0.5)
        ax.plot(x, p90, ls="none", marker="_", ms=10, mew=1.5, color=col, alpha=0.6)
        ax.plot(x, med, color=col, lw=2, marker="o", ms=8, mec="white", mew=1.5, label=label)

    series(a, "centre", BLUE, "median (bar: 90th percentile)")
    a.set_ylim(0, None)
    a.set_ylabel("error in the lens center (pixels)", color=INK, fontsize=10)
    a.set_title("Where the lens is centered on the sensor", color=INK, fontsize=11, loc="left")
    for x, r in zip(ns, B):
        a.annotate(f"{r['cameras']} cameras", (x, 0), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=8, color=MUTED)
    a.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    series(b, "shape_top", BLUE, "where the stars are", -0.06)
    series(b, "shape_low", ORANGE_, "below the skyline", 0.06)
    for yv, txt in ((m["star_fit_px"], "one night's fit to its stars (median)"),
                    (m["night_to_night_px"], "two nights' agreement (median)")):
        b.axhline(yv, color=MUTED, lw=1, ls=(0, (4, 3)))
        b.text(max_n + 0.28, yv + 0.012, txt, ha="right", va="bottom", fontsize=8.5, color=MUTED)
    b.set_ylim(0, 0.9)
    b.set_ylabel("error in where pixels point (pixels)", color=INK, fontsize=10)
    b.set_title("What that does to where pixels point", color=INK, fontsize=11, loc="left")
    b.legend(frameon=False, fontsize=8.5, loc="upper center", labelcolor=INK, ncol=2,
             bbox_to_anchor=(0.5, 0.86))
    fig.tight_layout(w_pad=3)
    plt.close(fig)
    return fig


def static(name) -> Image | None:
    """A figure drawn elsewhere (plume_figs.py, from FIgLib frames). These files aren't in the
    repository; without them this returns None and build.py keeps the committed output."""
    path = HERE / "static" / name
    return Image(filename=str(path)) if path.exists() else None


def motion_arrows(canvas, s, est, k, pl, pt, dt_s=900, grid=170, every=16, span=6):
    """Draw on `canvas` (the frame at offset pl, pt) the trail velocities measured (green) and
    the velocity field the fitted pole predicts (gold): each arrow is how far a star moves in
    `dt_s` seconds. Predicted arrows come from d_dot = omega * (p x d) at every grid point,
    mapped back to pixels through the lens."""
    c, W, H = s["cam_c"], s["W"], s["H"]
    p = est["p"]

    def arrow(x0, y0, x1, y1, col, w):
        if np.all(np.isfinite([x0, y0, x1, y1])):
            cv2.arrowedLine(canvas, (int(x0 + pl), int(y0 + pt)), (int(x1 + pl), int(y1 + pt)),
                            col, w, cv2.LINE_AA, tipLength=0.25)

    # Predicted: the rigid rotation about the fitted pole, on a grid over the frame.
    gx, gy = np.meshgrid(np.arange(grid / 2, W, grid), np.arange(grid / 2, H, grid))
    sky = s["mask"][gy.astype(int), gx.astype(int)]            # only where stars can be
    gx, gy = gx[sky], gy[sky]
    d = pole.pixel_to_cam(gx.ravel(), gy.ravel(), W, H, k, K1, c["cx"], c["cy"])
    d2 = d + pole.HANDEDNESS * pole.OMEGA * dt_s * np.cross(d, p)   # as pole.fit_pole: V = sign*OMEGA*(D x p)
    x1, y1 = cam_to_px(d2 / np.linalg.norm(d2, axis=1, keepdims=True), W, H, k, K1, c["cx"], c["cy"])
    for a, b, e, f in zip(gx.ravel(), gy.ravel(), x1, y1):
        arrow(a, b, e, f, GOLD, 3)

    # Measured: a few points along each trail, the motion over the next `span` frames.
    for t in s["night1"].tracks:
        offs = sorted(t)
        xy = np.array([t[o][:2] for o in offs])
        for i in range(0, len(offs) - span, every):
            v = (xy[i + span] - xy[i]) / (offs[i + span] - offs[i])
            arrow(*xy[i], *(xy[i] + v * dt_s), GREEN, 6)


def velocity_figure(name, day, dt_s=900, grid=170, every=16, span=6,
                    pad=(40, 60, 40, 40), crop=True, dim=0.55):
    """Measured trail velocities (green) over the velocity field the fitted pole predicts
    (gold), over the night's stacked frames; see motion_arrows."""
    s = solve(name, day)
    c, W, H = s["cam_c"], s["W"], s["H"]
    est, k, _ = pole_fit(s)
    pl, pt, pr, pb = pad
    img = None
    for _, _, blob in s["frames"]:
        im = decode(blob)
        img = im if img is None else np.maximum(img, im)
    canvas = np.full((H + pt + pb, W + pl + pr, 3), 18, np.uint8)
    canvas[pt:pt + H, pl:pl + W] = (cv2.convertScaleAbs(img, alpha=2.0) * dim).astype(np.uint8)
    cv2.rectangle(canvas, (pl, pt), (pl + W - 1, pt + H - 1), (90, 90, 90), 2)

    motion_arrows(canvas, s, est, k, pl, pt, dt_s, grid, every, span)

    x, y = cam_to_px(est["p_hat"], W, H, k, K1, c["cx"], c["cy"])
    X, Y = int(x[0]) + pl, int(y[0]) + pt
    cv2.drawMarker(canvas, (X, Y), WHITE, cv2.MARKER_CROSS, 90, 5, cv2.LINE_AA)
    cv2.circle(canvas, (X, Y), 30, WHITE, 4, cv2.LINE_AA)
    if crop:   # drop the ground below the sky
        canvas = canvas[:pt + int(np.where(s["mask"].any(axis=1))[0].max()) + 80]
    return jpeg(canvas)


@lru_cache(maxsize=None)
def _crest_points(name, min_km=0.5):
    return terrain.crests(cameras()[name], min_km=min_km)


LABEL_PEAKS = ["Cuyamaca Peak", "Sycuan Peak", "Monument Peak", "Corte Madera Mountain",
               "Lyons Peak", "Hauser Mountain", "Tecate Peak"]


def ridges(name, solve_day, day, q=6, frame=0, min_km=0.5, peaks=()):
    """A whole daytime frame with every terrain crest drawn through the `solve_day` pose,
    colored by distance (purple near, yellow far), and the named `peaks` labeled where the
    same pose puts them. Nothing is fitted to the picture."""
    s = solve(name, solve_day)
    _, frames = read_block(name, day, q=q)
    daylit = decode(frames[frame][2])
    az, el, km, sky = _crest_points(name, min_km)
    x, y = terrain.project(s["cam_c"], az, el, s["W"], s["H"], s["r"]["pose"])
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < s["W"]) & (y >= 0) & (y < s["H"])
    fig, ax = plt.subplots(figsize=(14, 14 * s["H"] / s["W"] + 0.6))
    ax.imshow(rgb(daylit))
    from matplotlib.colors import LogNorm
    sc = ax.scatter(x[ok], y[ok], c=km[ok], s=1.2, cmap="plasma", norm=LogNorm(0.5, 80), linewidths=0)
    ax.set_xlim(0, s["W"]); ax.set_ylim(s["H"], 0)
    ax.axis("off")
    if peaks:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=3, foreground="black")]
        pk = {p["name"]: p for p in visible_peaks(name, solve_day)}
        for i, n in enumerate(sorted(peaks, key=lambda n: pk[n]["x"])):
            p = pk[n]
            ly = p["y"] - 150 - 70 * (i % 3)
            ax.plot([p["x"], p["x"]], [p["y"] - 8, ly + 10], color="white", lw=0.8)
            ax.plot(p["x"], p["y"], marker="v", ms=6, color="white", mec="black", mew=0.6)
            ax.text(p["x"], ly, f"{n}\n{p['km']:.0f} km", color="white", fontsize=9, ha="center",
                    va="bottom", path_effects=halo, linespacing=1.1)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, aspect=50)
    cb.set_ticks([0.5, 1, 2, 5, 10, 20, 50])
    cb.set_ticklabels(["0.5", "1", "2", "5", "10", "20", "50 km"])
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    plt.close(fig)
    return jpeg_fig(fig)


# ---------------------------------------------------------------- named peaks

def peak_catalog():
    """Named mountains within 90 km of San Miguel, from Wikidata (CC0; mostly from USGS GNIS),
    saved in static/peaks_wd.json: [(name, lat, lon)], duplicates dropped."""
    import re
    rows, seen = [], set()
    for b in json.loads((HERE / "static" / "peaks_wd.json").read_text())["results"]["bindings"]:
        name = re.sub(r"\s*\(.*\)", "", b["itemLabel"]["value"])
        if re.fullmatch(r"Q\d+", name) or name in seen:
            continue
        lon, lat = map(float, re.findall(r"[-\d.]+", b["coord"]["value"]))
        seen.add(name)
        rows.append((name, lat, lon))
    return rows


@lru_cache(maxsize=None)
def visible_peaks(name, solve_day):
    """Every catalog peak with its bearing, elevation angle and distance from the camera,
    whether terrain in front hides it, and its pixel through the solved pose."""
    s = solve(name, solve_day)
    cam = s["cam"]
    h0 = terrain.eye_height(cam)
    out = []
    for pname, lat, lon in peak_catalog():
        h = float(terrain.terrain_height(np.array([lat]), np.array([lon]))[0])
        az, el, dist = terrain.az_el(cam["lat"], cam["lon"], h0, lat, lon, h)
        ds = np.arange(100.0, dist - 150.0, 30.0)
        block = -90.0
        if len(ds):
            la, lo = terrain.destination(cam["lat"], cam["lon"], np.full(len(ds), az), ds)
            hh = terrain.terrain_height(la, lo)
            block = float(np.degrees(np.arctan2(
                hh - h0 - ds ** 2 * (1 - terrain.K_REFR) / (2 * terrain.R_EARTH), ds)).max())
        x, y = terrain.project(s["cam_c"], np.array([az]), np.array([el]), s["W"], s["H"], s["r"]["pose"])
        x, y = float(x[0]), float(y[0])
        out.append(dict(name=pname, lat=lat, lon=lon, h=h, az=az, el=el, km=dist / 1000,
                        visible=bool(el >= block - 0.02), x=x, y=y,
                        in_frame=bool(np.isfinite(x) and 0 <= x < s["W"] and 0 <= y < s["H"])))
    return out


def horizontal_fov(name, solve_day):
    """Bearings (deg) of the frame's left and right edges at the horizon, through the solved pose."""
    s = solve(name, solve_day)
    az = np.arange(s["cam"]["az"] - 90, s["cam"]["az"] + 90, 0.05)
    x, _ = terrain.project(s["cam_c"], az, np.zeros_like(az), s["W"], s["H"], s["r"]["pose"])
    ok = np.isfinite(x) & (x >= 0) & (x < s["W"])
    return float(az[ok].min()), float(az[ok].max())


def frustum_map(name, solve_day, peaks=LABEL_PEAKS, extent=(-40, 62, -32, 40), reach_km=70):
    """Overhead map: shaded terrain, the camera's horizontal field of view through the solved
    pose, its solved and nominal headings, and the named peaks with their bearings."""
    from matplotlib.colors import LightSource
    import matplotlib.patheffects as pe
    s = solve(name, solve_day)
    cam = s["cam"]
    lat0, lon0 = cam["lat"], cam["lon"]
    kx = 111.32 * math.cos(math.radians(lat0))      # km per degree of longitude here
    e = np.linspace(extent[0], extent[1], 600)
    n = np.linspace(extent[2], extent[3], 600)
    E, N = np.meshgrid(e, n)
    h = terrain.terrain_height(lat0 + N / 111.32, lon0 + E / kx)
    step_m = (e[1] - e[0]) * 1000
    shade = LightSource(azdeg=315, altdeg=40).hillshade(h[::-1], vert_exag=3, dx=step_m, dy=step_m)[::-1]
    rgbimg = np.dstack([0.55 + 0.4 * shade] * 3)
    rgbimg[h <= 1] = (0.78, 0.86, 0.93)              # sea
    fig, ax = plt.subplots(figsize=(10, 10 * (extent[3] - extent[2]) / (extent[1] - extent[0])))
    ax.imshow(rgbimg, extent=extent, origin="lower")

    def xy(az, km):
        a = np.radians(az)
        return km * np.sin(a), km * np.cos(a)

    a0, a1 = horizontal_fov(name, solve_day)
    arc = np.linspace(a0, a1, 100)
    wx, wy = xy(arc, reach_km)
    ax.fill(np.r_[0, wx, 0], np.r_[0, wy, 0], color="#f59e0b", alpha=0.18, lw=0)
    for a in (a0, a1):
        ax.plot(*zip((0, 0), xy(a, reach_km)), color="#b45309", lw=1)
    head = (cam["az"] + s["r"]["pose"]["d_az"]) % 360
    ax.plot(*zip((0, 0), xy(head, reach_km)), color="#b45309", lw=1.6, label=f"solved azimuth {head:.1f}°")
    ax.plot(*zip((0, 0), xy(cam["az"], reach_km)), color="#555", lw=1.2, ls="--",
            label=f"{cam['az']:.0f}°")

    # The US-Mexico border here is one straight line, from the Pacific to the Colorado River.
    b_lat, b_lon = np.array([32.5343, 32.7180]), np.array([-117.1241, -114.7195])
    t = np.linspace(0, 1, 50)
    ax.plot((b_lon[0] + t * np.diff(b_lon) - lon0) * kx, (b_lat[0] + t * np.diff(b_lat) - lat0) * 111.32,
            color="#666", lw=1, ls=(0, (6, 3)))
    ax.text(extent[1] - 2, (32.5343 + (lon0 + (extent[1] - 2) / kx + 117.1241) / 2.4046 * 0.1837 - lat0) * 111.32 - 2.5,
            "MEXICO", ha="right", va="top", fontsize=8, color="#555")

    ax.text(extent[0] + 3, 0, "Pacific\nOcean", fontsize=9, color="#4b6a88", style="italic", va="center")
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    pk = {p["name"]: p for p in visible_peaks(name, solve_day)}
    for nme in peaks:
        p = pk[nme]
        px, py = xy(p["az"], p["km"])
        ax.plot([0, px], [0, py], color="#1f2937", lw=0.6, alpha=0.6)
        ax.plot(px, py, marker="^", ms=7, color="#1f2937")
        ax.annotate(f"{nme}\n{p['az']:.1f}°, {p['km']:.0f} km", (px, py), xytext=(6, 4),
                    textcoords="offset points", fontsize=8.5, path_effects=halo)
    ax.plot(0, 0, marker="o", ms=7, color="#b45309", mec="white")
    ax.annotate("Mt. San Miguel\ncamera", (0, 0), xytext=(-6, -6), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, path_effects=halo)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("km east of the camera"); ax.set_ylabel("km north of the camera")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    plt.close(fig)
    return jpeg_fig(fig)


# ---------------------------------------------------------------- panoramas

def daylight_frame(name, day, q=4, index=119):
    """One daytime frame, by its place in the CDN's list for the Q block (119: about 11:00
    in Q4), fetched into the block's directory unless it's there already."""
    listing = nights._get(f"{nights.CDN}/hpwren-cameras/{name}/{day[:4]}/{day}/{day}_{name}_Q{q}.txt")
    file = [l.strip() for l in listing.decode().splitlines() if l.strip().endswith(".jpg")][index]
    p = cache_dir() / "nights" / name / f"{day}_Q{q}" / file
    if not p.exists() or p.stat().st_size == 0:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(nights._get(f"{nights.CDN}/MTA/{name}/large/{day}/Q{q}/{file}"))
    return decode(p.read_bytes())


def _usable(img):
    """Where a frame shows the scene: not the black outside the lens circle (dark regions
    joined to the border; shadows inside the circle stay), the timestamp line or the logos."""
    dark = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) <= 12).astype(np.uint8)
    _, lab = cv2.connectedComponents(dark)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    ok = ~np.isin(lab, edge[edge > 0])
    ok[:44] = False
    ok[:190, :620] = False
    ok[1870:2020, 2570:3010] = False
    return ok


def stitch(frames, poses, az, el, hard=False):
    """Blend frames {name: image} onto the (az, el) grid. poses: {name: (cam, pose)}, pose None
    for the camera table's pointing with the shared lens. Each camera's weight grows with the
    distance from the edge of its picture, and its exposure is matched to its neighbours'
    over the overlaps. hard: each point from the single camera that sees it most centrally,
    so a misplaced camera shows as a break at the seam rather than a blur."""
    names = list(frames)
    warped = []
    for n in names:
        img = frames[n]
        H, W = img.shape[:2]
        cam, pose = poses[n]
        x, y = terrain.project(cam, az.ravel(), el.ravel(), W, H, pose)
        mx = np.nan_to_num(x.reshape(az.shape), nan=-1).astype(np.float32)
        my = np.nan_to_num(y.reshape(az.shape), nan=-1).astype(np.float32)
        dist = cv2.distanceTransform(_usable(img).astype(np.uint8), cv2.DIST_L2, 5)
        w = cv2.remap(dist, mx, my, cv2.INTER_LINEAR, borderValue=0)
        warped.append((cv2.remap(img, mx, my, cv2.INTER_LINEAR).astype(np.float32), w))
    rows, rhs = [np.ones(len(names))], [np.zeros(3)]
    for i in range(len(names)):
        j = (i + 1) % len(names)
        both = (warped[i][1] > 20) & (warped[j][1] > 20)
        if both.sum() > 1000:
            r = np.zeros(len(names)); r[i], r[j] = 1, -1
            rows.append(r)
            rhs.append(np.log(warped[j][0][both].mean(0) + 1) - np.log(warped[i][0][both].mean(0) + 1))
    gain = np.exp(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0])
    acc, wsum = 0, 0
    best = np.argmax(np.stack([w for _, w in warped]), axis=0)
    for i, ((img, w), g) in enumerate(zip(warped, gain)):
        w4 = (best == i) * (w > 0) if hard else np.clip(w, 0, 600) ** 4
        acc = acc + np.clip(img * g, 0, 255) * w4[..., None]
        wsum = wsum + w4
    return (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)


SITE_NIGHTS = {"buff": {"n": "20261016", "e": "20260913", "s": "20260913", "w": "20261016"}}


def panorama(site="buff", day="20261016", ppd=16, el_range=(-28, 14),
             zooms=((219, 249, -16, 6), (292, 322, -18, 4))):
    """A site's four color cameras joined into one sweep, north in the middle, twice: through
    the camera table's pointing with the shared lens, and through each camera's star solve.
    Each point comes from the one camera that sees it most centrally, with no blending, so a
    misplaced camera shows as a break at the seam. Below, each `zooms` box (az from, az to,
    el from, el to) both ways, at the cameras' own resolution."""
    names = {d: f"{site}-{d}-mobo-c" for d in "nesw"}
    frames = {n: daylight_frame(n, day) for n in names.values()}
    poses = ({}, {})
    for d, n in names.items():
        s = solve(n, SITE_NIGHTS[site][d])
        poses[0][n] = (s["cam"], None)
        poses[1][n] = (s["cam_c"], s["r"]["pose"])

    def grid(a0, a1, e0, e1, k):
        return np.meshgrid(np.arange(a0, a1, 1 / k) % 360, np.arange(e1, e0, -1 / k))

    A, E = grid(-180, 180, *el_range, ppd)
    strips = [stitch(frames, p, A, E, hard=True) for p in poses]
    boxes = [[stitch(frames, p, *grid(*z, 30), hard=True) for p in poses] for z in zooms]

    INK = "#2b2b2b"
    fig_w = 14
    strip_h = fig_w * strips[0].shape[0] / strips[0].shape[1]
    box_h = [(fig_w / 2) * b[0].shape[0] / b[0].shape[1] for b in boxes]
    fig = plt.figure(figsize=(fig_w, 2 * strip_h + sum(box_h) + 0.4 * (2 + len(zooms))))
    gs = fig.add_gridspec(2 + len(zooms), 2, height_ratios=[strip_h, strip_h, *box_h],
                          hspace=0.16, wspace=0.03)
    ticks = [(-180, "S"), (-135, "SW"), (-90, "W"), (-45, "NW"), (0, "N"), (45, "NE"),
             (90, "E"), (135, "SE"), (180, "S")]
    titles = ("Nominal orientation, shared lens", "Star-calibrated orientation and lens")
    for row, (img, title) in enumerate(zip(strips, titles)):
        ax = fig.add_subplot(gs[row, :])
        ax.imshow(rgb(img), extent=(-180, 180, *el_range), aspect="auto")
        for k, z in enumerate(zooms):
            a0 = (z[0] + 180) % 360 - 180
            ax.add_patch(plt.Rectangle((a0, z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec="white", lw=1.2))
            ax.text(a0 + 0.8, z[3] - 0.8, str(k + 1), color="white", fontsize=10, va="top")
        ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks])
        ax.set_yticks([])
        ax.tick_params(length=0, labelsize=9, colors=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    for k, pair in enumerate(boxes):
        for col, img in enumerate(pair):
            ax = fig.add_subplot(gs[2 + k, col])
            ax.imshow(rgb(img))
            ax.axis("off")
            ax.set_title(f"{k + 1}: {('nominal', 'star-calibrated')[col]}", fontsize=10,
                         color=INK, loc="left")
    plt.close(fig)
    return jpeg_fig(fig)
