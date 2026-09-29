"""A solve as data for the browser: everything the replay page draws, already projected.

`export` turns a solve's `trace` (see `solve.solve`), its result and the `Night` it solved
into one JSON-ready dict, in the frame's own pixels. Every star position the page shows is
projected here, by the solver's camera model, so the page only draws and the model exists
in one language. The page can show only the poses the solver actually tried: the scan's
poses (sampled), each rung's pose before and after its refit, and the result.

What it holds (docs/records.md has the fields):

  night       the tracks as the solver clipped them, which of them it used, and the frames
              to play them over (the caller's URLs: nothing here reads or copies a frame)
  attempts    one per attempt `solve_wide` made: the pole fit's own samples, kept or
              trimmed; the bright stars at the published pose; the coarse scan's scores and
              the bright stars under a sample of its poses; every rung of every start, with
              each paired star's arc before and after the refit; the verdict
  final       the kept result drawn as overlay.py draws it, and the catalog stars in frame
              under its pose (the published one for a failure), for labels

    trace = []
    result = solve_wide(night, trace=trace)
    data = export(trace, result, night, [(offset, "frames/123.jpg"), ...])

The page itself is hpwren's explore.html; `calibrate explore` writes both.
"""
from __future__ import annotations

import math

import numpy as np

from . import catalog as SG
from .animate import _attempts, _describe, _first
from .fisheye import K1, K_RATIO, initial_k, project_fisheye
from .solve import Night, prune, windowed

FORMAT = 1
ARC_POINTS = 24        # an arc is smooth: this many points draw it
SCAN_SAMPLES = 160     # poses of the coarse scan whose stars are kept


def _num(a, d: int = 1):
    """Rounded nested lists, None where a value isn't finite (JSON has no NaN)."""
    def clean(v):
        return [clean(u) for u in v] if isinstance(v, list) else v if math.isfinite(v) else None
    return clean(np.round(np.asarray(a, float), d).tolist())


def _thin(n: int, k: int = ARC_POINTS) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int)) if n else np.zeros(0, int)


class _Sky:
    """The night's stars in pixels: star positions at the tracks' offsets, and projection."""

    def __init__(self, night: Night):
        self.night, self.c, self.W, self.H = night, night.cam, night.W, night.H
        self.offs = sorted({o for t in night.tracks for o in t}) or [0]
        self._arcs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def project(self, az, alt, p5) -> np.ndarray:
        """Pixels; NaN for a star too far off the boresight to be in the picture."""
        x, y = project_fisheye(self.c, az, alt, self.W, self.H, *p5)
        return np.c_[np.atleast_1d(x) * self.W, np.atleast_1d(y) * self.H]

    def inside(self, xy: np.ndarray, pad: float = 0.0) -> np.ndarray:
        return ((xy[:, 0] >= -pad) & (xy[:, 0] < self.W + pad)
                & (xy[:, 1] >= -pad) & (xy[:, 1] < self.H + pad))

    def altaz(self, name: str, offs) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._arcs:
            c, ep = self.c, self.night.t0 + np.array(self.offs, float)
            self._arcs[name] = SG.star_altaz(name, ep, c["lat"], c["lon"], c.get("elev") or 1600.0)
        alt, az = self._arcs[name]
        j = np.searchsorted(self.offs, offs)
        return alt[j], az[j]

    def arc(self, name: str, offs, p5) -> np.ndarray:
        """A star's path over the given offsets under pose `p5`, thinned."""
        offs = np.asarray(offs)[_thin(len(offs))]
        alt, az = self.altaz(name, offs)
        return self.project(az, alt, p5)

    def at(self, stars: list[dict], epoch: float, p5, pad: float = 0.0):
        """Names and pixels of the given stars (visible_stars rows) that land in frame."""
        if not stars:
            return [], np.zeros((0, 2))
        xy = self.project(np.array([s["az"] for s in stars]), np.array([s["alt"] for s in stars]),
                          p5)
        ok = self.inside(xy, pad)
        return [s["name"] for s, k in zip(stars, ok) if k], xy[ok]


def _pole(sky: _Sky, raw: list[dict], used: list[int], ev: list[dict]) -> dict | None:
    pole, scan, setup = _first(ev, "pole"), _first(ev, "scan"), _first(ev, "setup")
    if pole is None:
        return None
    # The fit's own input, in pole.samples' order: one step per pair of consecutive frames
    # of each track, drawn at its midpoint in the direction the star moved.
    steps = []
    for j in used:
        t = raw[j]
        o = sorted(t)
        if len(o) < 2:
            continue
        for a, b in zip(o[:-1], o[1:]):
            if b > a:
                p, q = np.array(t[a][:2], float), np.array(t[b][:2], float)
                d = q - p
                n = float(np.hypot(*d)) or 1.0
                steps.append([*(p + q) / 2, *(d / n), j])
    inl = pole.get("inliers")
    kept = ("".join("1" if k else "0" for k in inl)
            if inl is not None and len(inl) == len(steps) else None)
    xy = None
    if scan is not None and setup is not None and pole["fit"] is not None:
        # the celestial pole (az 0, alt = latitude) under any of the scan's poses: the scan
        # only turns the sky about it, so they all put it in the same place
        p = sky.project(np.array([0.0]), np.array([sky.c["lat"]]), (*scan["poses"][0],
                                                                     *setup["lens"]))[0]
        xy = _num(p)
    return {"fit": pole["fit"], "steps": _num(steps, 2), "kept": kept, "xy": xy}


def _scan(sky: _Sky, ev: list[dict], stars: list[dict], epoch: float, wide: bool) -> dict | None:
    scan, starts, setup = _first(ev, "scan"), _first(ev, "starts"), _first(ev, "setup")
    if scan is None or setup is None or not len(scan["scores"]):
        return None
    poses, scores = np.asarray(scan["poses"], float), np.asarray(scan["scores"], float)
    if wide:
        order, x = np.arange(len(scores)), np.asarray(scan["psi"], float)
        label = "turn about the pole, psi (deg)"
    else:
        order = np.arange(len(scores))[::-1]    # the grid's best 200, worst to best
        x = np.arange(len(scores), 0, -1, dtype=float)
        label = "grid pose, by rank (200 = worst kept, 1 = best)"
    poses, scores = poses[order], scores[order]
    marks = []
    for p in (starts or {}).get("poses", []):
        marks.append(int(np.argmin(np.abs(poses - np.array(p)).max(axis=1))))
    idx = sorted(set(np.linspace(0, len(order) - 1, min(len(order), SCAN_SAMPLES))
                     .round().astype(int).tolist()) | set(marks))
    samples = [{"i": i, "xy": _num(sky.at(stars, epoch, (*poses[i], *setup["lens"]))[1], 0)}
               for i in idx]
    return {"label": label, "x": _num(x, 1), "scores": _num(scores, 2), "poses": _num(poses, 2),
            "samples": samples, "starts": marks}


def _rungs(sky: _Sky, raw: list[dict], ev: list[dict]) -> list[dict]:
    out = []
    for r in (e for e in ev if e["stage"] == "rung"):
        arcs = {}
        for name, j in r["pairs"]:
            offs = sorted(raw[j])
            arcs[name] = {"before": _num(sky.arc(name, offs, r["before"])),
                          "after": None if r["after"] is None else _num(sky.arc(name, offs, r["after"]))}
        out.append({"start": r["start"], "thr": r["thr"], "free_lens": r["free_lens"],
                    "dropped": r["after"] is None, "median_px": r["median_px"],
                    "before": _num(r["before"], 4),
                    "after": None if r["after"] is None else _num(r["after"], 4),
                    "pairs": [[n, int(j)] for n, j in r["pairs"]], "arcs": arcs})
    return out


def _final(sky: _Sky, result: dict, raw: list[dict], ref: int) -> dict:
    """The kept result as overlay.draw shows it, and the catalog stars in frame for labels."""
    c, k0 = sky.c, initial_k(sky.c, sky.W)
    epoch = sky.night.t0 + ref
    out: dict = {"fitted_arcs": {}, "published_arcs": {}, "arrows": [], "bright_arcs": {}}
    if "pose" in result:
        p = result["pose"]
        pose = (p["d_az"], p["d_pitch"], p["d_roll"], p["k_ratio"] * k0, p["k1"])
        pub = (0.0, 0.0, 0.0, pose[3], pose[4])
        for name, j in result.get("matches", {}).items():
            offs = sorted(raw[j]) if j < len(raw) and raw[j] else sky.offs
            out["fitted_arcs"][name] = _num(sky.arc(name, offs, pose))
            out["published_arcs"][name] = _num(sky.arc(name, offs, pub))
            alt, az = SG.star_altaz(name, np.array([epoch], float), c["lat"], c["lon"],
                                    c.get("elev") or 1600.0)
            a, b = sky.project(az, alt, pub)[0], sky.project(az, alt, pose)[0]
            out["arrows"].append([name, *_num(a), *_num(b)])
    else:
        pose = (0.0, 0.0, 0.0, K_RATIO * k0, K1)
        for s in SG.visible_stars(c, epoch, mag_limit=3.0, fov_margin_deg=20):
            xy = sky.arc(s["name"], sky.offs, pose)
            if sky.inside(xy).sum() >= 2:
                out["bright_arcs"][s["name"]] = _num(xy)
    vis = SG.visible_stars(c, epoch, mag_limit=4.0, fov_margin_deg=180, min_alt_deg=0)
    names, xy = sky.at(vis, epoch, pose)
    mag = {s["name"]: s["mag"] for s in vis}
    out["sky"] = [[n, round(mag[n], 2), *_num(p)] for n, p in zip(names, xy)]
    return out


def export(trace: list[dict], result: dict, night: Night,
           frames: list[tuple[int, str]]) -> dict:
    """The replay page's data for one solve: `trace` and `result` from `solve_wide(night,
    trace=trace)` (or `solve`), and `frames` as (offset from t0, URL) in time order, the
    pictures to play the night over. The frames' URLs are the caller's: the page loads them
    as given."""
    sky = _Sky(night)
    runs, kept = _attempts(trace)
    raw = windowed(night.tracks, result.get("window"))[0]
    used = runs[0][0]["tracks"] if runs else prune(raw)
    ref = int(result.get("ref_offset", sky.offs[len(sky.offs) // 2]))
    epoch = night.t0 + ref
    k0 = initial_k(night.cam, night.W)

    attempts = []
    for a, ev in enumerate(runs):
        at, setup = ev[0], _first(ev, "setup")
        lens = list(setup["lens"]) if setup else None
        bright = [s for s in SG.visible_stars(night.cam, epoch, mag_limit=4.0,
                                              fov_margin_deg=180 if at["wide"] else 45,
                                              min_alt_deg=-5) if s["mag"] <= 3.5]
        published = None
        if setup is not None:
            names, xy = sky.at(bright, epoch, (0.0, 0.0, 0.0, *setup["lens"]))
            published = {"names": names, "xy": _num(xy)}
        v, starts = _first(ev, "verdict"), _first(ev, "starts")
        attempts.append({
            "wide": at["wide"], "k_ratio": at["k_ratio"], "title": _describe(at, setup, k0),
            "kept": kept == a, "lens": _num(lens, 4) if lens else None,
            "lens_ratio": None if lens is None else round(lens[0] / k0, 4),
            "ladder": [list(r) for r in setup["ladder"]] if setup else [],
            "pole": _pole(sky, raw, at["tracks"], ev) if at["wide"] else None,
            "published": published,
            "scan": _scan(sky, ev, bright, epoch, at["wide"]),
            "starts": [{"pose": _num(p, 2), "score": round(float(s), 2)}
                       for p, s in zip(starts["poses"], starts["scores"])] if starts else [],
            "rungs": _rungs(sky, raw, ev),
            "verdict": None if v is None else {
                "status": v["status"], "reason": v["reason"], "start": v.get("start"),
                "tests": [[t, bool(ok)] for t, ok in v.get("tests") or []]},
        })

    keys = ("status", "reason", "pose", "n_stars", "median_px", "rmse_px", "matches",
            "per_star_px", "mags", "n_tracks", "n_tracks_raw", "solver", "found_by")
    return {
        "format": FORMAT, "seq": night.label or night.camera, "camera": night.camera,
        "imager": night.cam.get("imager"), "W": night.W, "H": night.H, "t0": night.t0,
        "ref": ref, "result": {k: result[k] for k in keys if k in result},
        "frames": [[int(o), url] for o, url in frames],
        "tracks": [_num([[o, *t[o][:2]] for o in sorted(t)]) for t in raw],
        "used": [int(j) for j in used],
        "attempts": attempts, "kept": kept,
        "final": _final(sky, result, raw, ref),
    }
