"""A solve, played back: how one block's trails became a pose, or where they stopped.

Every picture is drawn from what the solver actually computed, recorded by passing `trace=[]`
to `solve` or `solve_wide` -- nothing is re-derived here, so the video can't drift from the
solver. Scenes, in overlay.py's colours:

  night       the block's frames in order, each track (cyan) growing as its star moves
  attempt     a title card per attempt `solve_wide` made: which search, under which lens
  pole        (pole attempts) each track's direction of motion, then the celestial pole the
              trails imply, with |p|, which is 1 only for a coherent sky under the right lens
  start       (grid attempts) the bright stars where the published pose puts them (orange)
  scan        the bright stars under each candidate pose of the coarse search, over a strip
              plotting the coincidence score; the five starts it hands on are marked
  refine      each start's ladder: the predicted arcs (magenta) of the stars paired with a
              track (green) slide onto the tracks as the tolerance tightens
  verdict     the acceptance tests, passed or failed, or the reason the attempt stopped
  result      overlay.draw's picture of the result that was kept

`frames` yields the pictures and `write` encodes them. Nothing here reads a file; where the
frames come from and where the video goes is the caller's business (hpwren.calibrate animate).

    trace = []
    result = solve_wide(night, trace=trace)
    write(frames(trace, result, night, images), "solve.mp4")
"""
from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from . import catalog as SG
from . import overlay
from .fisheye import initial_k, project_fisheye
from .solve import Night, windowed

OUT_W = 1280
BANNER = 64
STRIP = 110
FPS = 24

GREEN, MAGENTA, GREY = (60, 220, 60), (255, 60, 255), (150, 150, 150)
CYAN, ORANGE, YELLOW = (230, 210, 60), (0, 140, 255), (0, 255, 255)
RED, WHITE = (60, 60, 255), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _attempts(trace: list[dict]) -> tuple[list[list[dict]], int | None]:
    """The trace split into one event list per attempt, and the index of the kept one."""
    runs, kept = [], None
    for e in trace:
        if e["stage"] == "attempt":
            runs.append([e])
        elif e["stage"] == "kept":
            kept = e["attempt"]
        elif runs:
            runs[-1].append(e)
    return runs, (kept if kept is not None else (0 if len(runs) == 1 else None))


def _first(events: list[dict], stage: str) -> dict | None:
    return next((e for e in events if e["stage"] == stage), None)


def _describe(attempt: dict, setup: dict | None, k0: float) -> str:
    if not attempt["wide"]:
        return "grid around the published pose (az +-30, pitch +-15, roll +-10 deg), shared lens"
    k = setup["lens"][0] / k0 if setup else None
    lens = (f"lens measured from the trails, {k:.3f}x" if attempt["k_ratio"] is not None
            else f"shared lens, {k:.3f}x" if k else "shared lens")
    return f"pole search, {lens}"


class _Stage:
    """The fixed geometry every picture shares: the sky band of the frame, scaled to OUT_W,
    between a text banner and a strip for plots."""

    def __init__(self, night: Night, ref_frame: np.ndarray, window=None, full: bool = False):
        self.night, self.c, self.W, self.H = night, night.cam, night.W, night.H
        self.s = OUT_W / self.W
        y_max = max((v[1] for t in night.tracks for v in t.values()), default=0.55 * self.H)
        crop = self.H if full else int(min(self.H, max(0.55 * self.H, y_max + 160)))
        self.view_h = int(round(crop * self.s / 2)) * 2
        bg = cv2.convertScaleAbs(ref_frame, alpha=2.0, beta=10)[:crop]
        self.bg = cv2.resize(bg, (OUT_W, self.view_h), interpolation=cv2.INTER_AREA)
        self.size = (OUT_W, BANNER + self.view_h + STRIP)
        self._arcs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.offs = sorted({o for t in night.tracks for o in t}) or [0]
        self.raw = windowed(night.tracks, window)[0]   # the tracks as the solver clipped them

    def view(self, frame: np.ndarray | None = None, dim: float = 0.6) -> np.ndarray:
        if frame is None:
            return (self.bg * dim).astype(np.uint8)
        f = cv2.convertScaleAbs(frame, alpha=2.0, beta=10)[: int(self.view_h / self.s)]
        return cv2.resize(f, (OUT_W, self.view_h), interpolation=cv2.INTER_AREA)

    def px(self, xy) -> np.ndarray:
        return (np.asarray(xy, float) * self.s).astype(np.int32)

    def project(self, az, alt, p5) -> np.ndarray:
        x, y = project_fisheye(self.c, az, alt, self.W, self.H, *p5)
        return np.c_[np.atleast_1d(x) * self.W, np.atleast_1d(y) * self.H]

    def arc(self, name: str, offs: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """A star's (alt, az) at the given offsets, as the solver computes them."""
        if name not in self._arcs:
            c, ep = self.c, self.night.t0 + np.array(self.offs, float)
            self._arcs[name] = SG.star_altaz(name, ep, c["lat"], c["lon"], c.get("elev") or 1600.0)
        alt, az = self._arcs[name]
        j = np.searchsorted(self.offs, offs)
        return alt[j], az[j]

    def tracks(self, img, ids, color, width, upto: float | None = None):
        for j in ids:
            t = self.night.tracks[j]
            o = [k for k in sorted(t) if upto is None or k <= upto]
            if len(o) >= 2:
                cv2.polylines(img, [self.px([t[k][:2] for k in o])], False, color, width,
                              cv2.LINE_AA)

    def compose(self, view: np.ndarray, lines: list[str], strip: np.ndarray | None = None,
                color=WHITE) -> np.ndarray:
        top = np.zeros((BANNER, OUT_W, 3), np.uint8)
        for k, text in enumerate(lines[:2]):
            cv2.putText(top, text, (12, 26 + 28 * k), FONT, 0.68, color if k else WHITE, 1,
                        cv2.LINE_AA)
        bottom = strip if strip is not None else np.zeros((STRIP, OUT_W, 3), np.uint8)
        return np.vstack([top, view, bottom])


def _card(st: _Stage, lines: list[str], color=WHITE) -> np.ndarray:
    view = st.view(dim=0.25)
    for k, text in enumerate(lines):
        cv2.putText(view, text, (40, 80 + 44 * k), FONT, 1.0 if k == 0 else 0.8,
                    color if k == 0 else WHITE, 2 if k == 0 else 1, cv2.LINE_AA)
    return st.compose(view, [])


def _plot(values: np.ndarray, upto: int, marks: list[int] = (), label: str = "") -> np.ndarray:
    """A score curve across the strip, drawn up to index `upto`, with numbered marks."""
    strip = np.zeros((STRIP, OUT_W, 3), np.uint8)
    n, top = len(values), max(float(np.max(values)) if len(values) else 0.0, 1e-9)
    x = 20 + np.arange(n) * (OUT_W - 40) / max(n - 1, 1)
    y = STRIP - 18 - np.asarray(values, float) / top * (STRIP - 40)
    if upto >= 1:
        cv2.polylines(strip, [np.c_[x[: upto + 1], y[: upto + 1]].astype(np.int32)], False,
                      ORANGE, 1, cv2.LINE_AA)
        cv2.line(strip, (int(x[min(upto, n - 1)]), 16), (int(x[min(upto, n - 1)]), STRIP - 14),
                 WHITE, 1)
    for k, i in enumerate(marks):
        p = (int(x[i]), int(y[i]))
        cv2.circle(strip, p, 5, MAGENTA, -1, cv2.LINE_AA)
        cv2.putText(strip, str(k + 1), (p[0] + 6, p[1] - 6), FONT, 0.5, MAGENTA, 1, cv2.LINE_AA)
    cv2.putText(strip, label, (20, 14), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    return strip


def _ladder_strip(rungs: list[dict], ladder, current: int | None, start: int, n_starts: int):
    """One box per rung of the ladder: its tolerance, and the pairs it kept once it's run."""
    strip = np.zeros((STRIP, OUT_W, 3), np.uint8)
    cv2.putText(strip, f"start {start + 1} of {n_starts}: tolerance ladder (pairs kept, median px)",
                (20, 18), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    w = (OUT_W - 40) // max(len(ladder), 1)
    for k, (thr, free) in enumerate(ladder):
        x0 = 20 + k * w
        r = rungs[k] if k < len(rungs) else None
        col = GREY if r is None else RED if r["after"] is None else GREEN
        cv2.rectangle(strip, (x0 + 4, 30), (x0 + w - 4, STRIP - 10), col,
                      2 if k == current else 1)
        cv2.putText(strip, f"{thr} px{' +lens' if free else ''}", (x0 + 12, 54), FONT, 0.55,
                    WHITE, 1, cv2.LINE_AA)
        if r is not None:
            txt = f"{len(r['pairs'])} pairs" + (f", {r['median_px']:.1f}" if r["median_px"] else "")
            cv2.putText(strip, txt, (x0 + 12, 84), FONT, 0.5, col, 1, cv2.LINE_AA)
    return strip


def _bright(st: _Stage, ref: int, wide: bool):
    """The stars the coarse score projects: mag <= 3.5 of the solver's candidate pool."""
    vis = SG.visible_stars(st.c, st.night.t0 + ref, mag_limit=4.0,
                           fov_margin_deg=180 if wide else 45, min_alt_deg=-5)
    b = [v for v in vis if v["mag"] <= 3.5]
    return np.array([v["az"] for v in b]), np.array([v["alt"] for v in b])


def _dots(img, st: _Stage, pts: np.ndarray, color, r=5):
    for x, y in st.px(pts):
        if 0 <= x < OUT_W and 0 <= y < st.view_h:
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)


def _scene_pole(st: _Stage, ev: list[dict], used: list[int], ref: int, title: str):
    pole, scan, setup = _first(ev, "pole"), _first(ev, "scan"), _first(ev, "setup")
    base = st.view()
    st.tracks(base, used, CYAN, 1)
    # The fit's own input: one velocity sample per step between consecutive frames of each
    # track, in pole.samples' order, and which of them the fit kept (yellow) or trimmed (red).
    # A track's step is only a few pixels, so its direction is drawn as a fixed-length arrow.
    segs = []
    for j in used:
        t = st.raw[j]
        o = sorted(t)
        segs += [(j, t[a][:2], t[b][:2]) for a, b in zip(o[:-1], o[1:]) if b > a]
    inl = pole.get("inliers") if pole else None
    ticks = base.copy()
    if inl is not None and len(inl) == len(segs):
        count: dict[int, int] = {}
        for (j, a, b), keep in zip(segs, inl):
            count[j] = count.get(j, 0) + 1
            if keep and count[j] % 4:        # every kept step would be a solid smear
                continue
            d = np.subtract(b, a)
            if np.hypot(*d) < 1e-6:
                continue
            m = st.px(np.add(a, b) / 2)
            q = m + (d / np.hypot(*d) * (18 if keep else 22)).astype(np.int32)
            cv2.arrowedLine(ticks, tuple(int(v) for v in m), tuple(int(v) for v in q),
                            YELLOW if keep else RED, 1 if keep else 2, cv2.LINE_AA, tipLength=0.4)
        lines = [title, f"the fit's input: {len(segs)} frame-to-frame steps, "
                        f"{int(np.sum(inl))} kept (yellow, every 4th drawn), "
                        f"{len(segs) - int(np.sum(inl))} trimmed as outliers (red)"]
    else:
        lines = [title, "the trails' motion: the sky turns about one axis, so it fixes the pole"]
    for _ in range(int(2.5 * FPS)):
        yield st.compose(ticks, lines, color=YELLOW)
    fit = pole and pole["fit"]
    if fit is None:
        return
    view = ticks.copy()
    text = (f"celestial pole in closed form: |p| {fit['norm']:.3f} (1 = a coherent sky), "
            f"{100 * fit['inlier_frac']:.0f}% inliers")
    if scan is not None and setup is not None:
        p5 = (*scan["poses"][0], *setup["lens"])
        xy = st.project(np.array([0.0]), np.array([st.c["lat"]]), p5)[0]
        x, y = st.px(xy)
        inside = 0 <= x < OUT_W and 0 <= y < st.view_h
        if inside:
            cv2.drawMarker(view, (int(x), int(y)), WHITE, cv2.MARKER_CROSS, 40, 3, cv2.LINE_AA)
            cv2.putText(view, "pole", (int(x) + 14, int(y) - 14), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
        else:
            e = (int(np.clip(x, 30, OUT_W - 30)), int(np.clip(y, 30, st.view_h - 30)))
            c0 = (OUT_W // 2, st.view_h // 2)
            cv2.arrowedLine(view, (c0[0] + (e[0] - c0[0]) * 3 // 4, c0[1] + (e[1] - c0[1]) * 3 // 4),
                            e, WHITE, 3, cv2.LINE_AA, tipLength=0.2)
            text += "  (pole off frame)"
    for _ in range(int(2.5 * FPS)):
        yield st.compose(view, [title, text], color=WHITE)


def _scene_scan(st: _Stage, ev: list[dict], used: list[int], ref: int, wide: bool, title: str):
    scan, starts, setup = _first(ev, "scan"), _first(ev, "starts"), _first(ev, "setup")
    if scan is None or setup is None or not len(scan["scores"]):
        return
    az, alt = _bright(st, ref, wide)
    base = st.view()
    st.tracks(base, used, CYAN, 2)
    poses, scores = np.asarray(scan["poses"]), np.asarray(scan["scores"])
    if wide:
        order = np.arange(len(scores))          # sweep psi 0 -> 360
        label = "coincidence score over the turn about the pole, psi 0-360 deg"
        what = "turning the sky about the pole: stars (orange) that land on a track score"
    else:
        order = np.arange(len(scores))[::-1]    # the grid's best 200, worst first
        label = "coincidence score of the grid's best 200 poses, worst to best"
        what = "the grid's best poses: stars (orange) that land on a track score"
    vals = scores[order]
    marks = []
    if starts:
        for p in starts["poses"]:
            k = int(np.argmin(np.abs(poses[order] - np.array(p)).max(axis=1)))
            marks.append(k)
    n_frames = int(6 * FPS)
    step = max(1, len(order) // n_frames)
    for u in range(0, len(order), step):
        view = base.copy()
        _dots(view, st, st.project(az, alt, (*poses[order[u]], *setup["lens"])), ORANGE)
        yield st.compose(view, [title, what], _plot(vals, u, label=label), color=ORANGE)
    view = base.copy()
    if starts and starts["poses"]:
        _dots(view, st, st.project(az, alt, (*starts["poses"][0], *setup["lens"])), ORANGE)
    msg = f"{len(marks)} distinct best poses go on to refinement" if marks else "no pose scored"
    for _ in range(int(2 * FPS)):
        yield st.compose(view, [title, msg], _plot(vals, len(vals) - 1, marks, label),
                         color=MAGENTA)


def _scene_refine(st: _Stage, ev: list[dict], used: list[int], ref: int, title: str):
    setup, starts = _first(ev, "setup"), _first(ev, "starts")
    rungs = [e for e in ev if e["stage"] == "rung"]
    if setup is None or not rungs:
        return
    ladder = setup["ladder"]
    n_starts = len(starts["poses"]) if starts else 1
    for s in sorted({r["start"] for r in rungs}):
        mine = [r for r in rungs if r["start"] == s]
        for k, r in enumerate(mine):
            ids = {j for _, j in r["pairs"]}
            steps = 10 if r["after"] is not None else 1
            for f in range(steps):
                a = (f + 1) / steps
                p5 = r["before"] if r["after"] is None else (1 - a) * r["before"] + a * r["after"]
                view = st.view()
                st.tracks(view, [j for j in used if j not in ids], CYAN, 1)
                st.tracks(view, ids, GREEN, 5)
                for name, j in r["pairs"]:
                    o = sorted(st.night.tracks[j])
                    alt, az = st.arc(name, o)
                    pts = st.px(st.project(az, alt, p5))
                    cv2.polylines(view, [pts], False, MAGENTA, 2, cv2.LINE_AA)
                if r["after"] is None:
                    lines = [title, f"start {s + 1}: only {len(r['pairs'])} stars within "
                                    f"{r['thr']} px, fewer than 4 -- this start is dropped"]
                    col = RED
                else:
                    lines = [title, f"start {s + 1}, tolerance {r['thr']} px"
                                    f"{', lens free' if r['free_lens'] else ''}: "
                                    f"{len(r['pairs'])} stars paired, median {r['median_px']:.2f} px"
                                    f"   green=track, magenta=star under this pose"]
                    col = GREEN
                strip = _ladder_strip(mine[: k + 1], ladder, k, s, n_starts)
                frame = st.compose(view, lines, strip, color=col)
                for _ in range(int(1.2 * FPS) if r["after"] is None else 1):
                    yield frame
            for _ in range(int(0.4 * FPS)):
                yield frame


def _scene_verdict(st: _Stage, ev: list[dict], title: str, kept: bool | None):
    v = _first(ev, "verdict")
    if v is None:
        return
    lines = [title]
    if v.get("tests"):
        lines += [f"{'PASS' if ok else 'FAIL'}  {text}" for text, ok in v["tests"]]
    else:
        lines += [f"FAILED: {v['reason']}"]
    if kept is not None:
        lines += ["", "this attempt's result is the one kept" if kept else "not kept"]
    view = st.view(dim=0.25)
    for k, text in enumerate(lines):
        col = (WHITE if k == 0 or not text else GREEN if text.startswith("PASS")
               else RED if text.startswith("FAIL") else GREY)
        cv2.putText(view, text, (40, 70 + 42 * k), FONT, 0.9 if k else 0.8, col, 2, cv2.LINE_AA)
    frame = st.compose(view, ["verdict", "solved" if v["status"] == "solved" else "failed"],
                       color=GREEN if v["status"] == "solved" else RED)
    for _ in range(int(3 * FPS)):
        yield frame


def frames(trace: list[dict], result: dict, night: Night,
           images: Iterable[tuple[int, np.ndarray]], only_kept: bool = False,
           full_frame: bool = False, mark: Iterable[str] = ()) -> Iterator[np.ndarray]:
    """The video's pictures, all the same size, from a solve's `trace`, its `result`, the
    `night` it solved, and the block's frames as (offset, BGR image) in time order.

    `images` is read once and only the frame nearest the reference is held, so a caller can
    pass a generator that decodes as it goes. `only_kept` skips the attempts `solve_wide`
    didn't keep, for a shorter video of the method rather than a diagnosis. `full_frame`
    draws the whole frame instead of the band of sky the tracks occupy. `mark` names stars to
    point out on the frame under the solved pose, in a closing scene: each item is a star
    ("Polaris") or a chain joined by "-" ("Alkaid-Mizar-Alioth-Megrez"), drawn as lines.
    """
    mark = [m.split("-") for m in mark]
    unknown = sorted({n for chain in mark for n in chain} - set(SG.STARS))
    if unknown:
        raise ValueError(f"not in the catalog: {', '.join(unknown)}")
    runs, kept = _attempts(trace)
    ref = overlay.reference_offset(result, night)
    used = runs[0][0]["tracks"] if runs else list(range(len(night.tracks)))
    images = iter(images)
    first = next(images, None)
    if first is None:
        raise ValueError("no frames to draw on")
    ref_frame = first[1]
    st = _Stage(night, ref_frame, result.get("window"), full_frame)
    k0 = initial_k(night.cam, night.W)
    label = night.label or night.camera

    # night: the frames in order, tracks growing
    def play(o, img):
        view = st.view(img)
        st.tracks(view, used, CYAN, 2, upto=o)
        return st.compose(view, [f"{label}   {night.cam.get('imager')}",
                                 f"the night: {len(used)} moving tracks, +{o / 60:.0f} min"],
                          color=CYAN)

    best = abs(first[0] - ref)
    for o, img in itertools.chain([first], images):
        if abs(o - ref) < best:
            best, ref_frame = abs(o - ref), img
        pic = play(o, img)
        yield pic
        yield pic
    st = _Stage(night, ref_frame, result.get("window"), full_frame)
    for _ in range(FPS):
        yield pic

    for a, ev in enumerate(runs):
        if only_kept and kept is not None and a != kept:
            continue
        at, setup = ev[0], _first(ev, "setup")
        title = f"attempt {a + 1} of {len(runs)}: {_describe(at, setup, k0)}"
        card = _card(st, [f"attempt {a + 1} of {len(runs)}", _describe(at, setup, k0)])
        for _ in range(int(1.5 * FPS)):
            yield card
        if at["wide"]:
            yield from _scene_pole(st, ev, used, ref, title)
        elif setup is not None:
            az, alt = _bright(st, ref, False)
            view = st.view()
            st.tracks(view, used, CYAN, 2)
            _dots(view, st, st.project(az, alt, (0.0, 0.0, 0.0, *setup["lens"])), ORANGE)
            for _ in range(int(2 * FPS)):
                yield st.compose(view, [title, "the published pose: where the camera table "
                                               "puts the bright stars (orange)"], color=ORANGE)
        yield from _scene_scan(st, ev, used, ref, at["wide"], title)
        yield from _scene_refine(st, ev, used, ref, title)
        yield from _scene_verdict(st, ev, title,
                                  None if len(runs) == 1 or kept is None else a == kept)

    # result: the overlay the gallery shows, fitted into the same picture size
    pic = overlay.draw(result, night, ref_frame, full=full_frame)
    h = st.size[1]
    scale = min(OUT_W / pic.shape[1], h / pic.shape[0])
    pic = cv2.resize(pic, (int(pic.shape[1] * scale), int(pic.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    out = np.zeros((h, OUT_W, 3), np.uint8)
    y0, x0 = (h - pic.shape[0]) // 2, (OUT_W - pic.shape[1]) // 2
    out[y0:y0 + pic.shape[0], x0:x0 + pic.shape[1]] = pic
    for _ in range(int(4 * FPS)):
        yield out
    if mark and "pose" in result:
        yield from _scene_mark(st, result, ref, mark)


def _scene_mark(st: _Stage, result: dict, ref: int, chains: list[list[str]]):
    """Named stars on the reference frame where the solved pose puts them: the pose read
    the other way, from pixels to sky."""
    p, c = result["pose"], st.c
    p5 = (p["d_az"], p["d_pitch"], p["d_roll"], p["k_ratio"] * initial_k(c, st.W), p["k1"])
    ep = np.array([st.night.t0 + ref], float)
    view = st.view(dim=1.0)
    names = []
    for chain in chains:
        pts = []
        for n in chain:
            alt, az = SG.star_altaz(n, ep, c["lat"], c["lon"], c.get("elev") or 1600.0)
            x, y = st.px(st.project(az, alt, p5))[0]
            ok = alt[0] > 0 and 0 <= x < OUT_W and 0 <= y < st.view_h
            pts.append((int(x), int(y)) if ok else None)
        for a, b in zip(pts[:-1], pts[1:]):
            if a and b:
                cv2.line(view, a, b, WHITE, 1, cv2.LINE_AA)
        for n, q in zip(chain, pts):
            if q and n not in names:
                names.append(n)
                cv2.circle(view, q, 9, YELLOW, 1, cv2.LINE_AA)
                cv2.putText(view, n, (q[0] + 12, q[1] - 10), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
    lines = [f"{st.night.label or st.night.camera}: the solved pose, read back onto the sky",
             f"{', '.join(names) or 'none of the named stars are in view'}"
             f" -- where the solved pose puts them"]
    for _ in range(int(6 * FPS)):
        yield st.compose(view, lines, color=YELLOW)


def write(pictures: Iterable[np.ndarray], dest, fps: int = FPS) -> Path:
    """Encode `pictures` as H.264 MP4 through the system's ffmpeg, which browsers play. With no
    ffmpeg on the PATH, OpenCV's own MPEG-4 (mp4v), which players like VLC play but browsers
    don't. (OpenCV's wheel also writes VP9 WebM, but at ~0.4 s a picture, too slow to use.)"""
    dest = Path(dest).with_suffix(".mp4")
    it = iter(pictures)
    first = next(it)
    h, w = first.shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for pic in itertools.chain([first], it):
                proc.stdin.write(np.ascontiguousarray(pic).tobytes())
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {dest}")
        return dest
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"OpenCV can't write {dest}")
    for pic in itertools.chain([first], it):
        vw.write(pic)
    vw.release()
    return dest
