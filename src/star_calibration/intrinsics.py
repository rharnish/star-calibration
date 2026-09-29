"""Per-camera optical centre, from all of a camera's solved nights at once.

`solve` assumes the lens's optical centre is the middle of the frame. On HPWREN's units it
isn't: freeing it in a refit of the 130 solved CDN blocks (2026-09-27) put it a median 23 px
off (middle half 14-37, max 81), dropped the median residual from 1.26 to 0.74 px, and
rescued 8 of 11 near-miss failures. It is a property of the unit, stable from night to night
(cp-w-mobo-c within 6 px over 12 nights), so it is fitted per camera from all its nights
together, not per solve: one night has too few stars to pin it, the centre trades off against
pose by up to ~1-1.5 deg, and a centre free in every solve would let a wrong fit slip under
the 3 px test.

The joint fit (`fit`) takes a camera's solved nights, each with its matched stars and
tracks, and solves for one shared (cx, cy, k, k1) and a pose per night, robust loss. A unit
can be swapped or its lens disturbed, so a camera's nights become a *series* (`series`):
each night's own centre is fitted alone first, and a new segment starts where two
consecutive nights both sit more than `JUMP_PX` from the segment so far. A segment with too
little to go on (fewer than `MIN_NIGHTS` nights and `MIN_STARS` matched stars) keeps the
frame's middle.

Records: {"camera", "W", "H", "from", "to", "cx", "cy", "k_ratio", "k1", "n_nights",
"n_stars", "median_px", "median_px_centred", "nights"}, epochs in seconds; `lookup` picks the
segment for a date. docs/records.md has the field list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import catalog as SG
from .fisheye import initial_k, project_fisheye

JUMP_PX = 25.0
MIN_NIGHTS = 2
MIN_STARS = 20


def observations(result: dict, night) -> dict | None:
    """One solved night's matched track points and their stars' (az, alt): the fit's input."""
    if result.get("status") != "solved" or not result.get("matches"):
        return None
    c = night.cam
    X, Y, AZ, AL = [], [], [], []
    for name, j in result["matches"].items():
        t = night.tracks[j]
        o = np.array(sorted(t), float)
        alt, az = SG.star_altaz(name, night.t0 + o, c["lat"], c["lon"], c.get("elev") or 1600.0)
        X += [t[k][0] for k in sorted(t)]
        Y += [t[k][1] for k in sorted(t)]
        AZ.append(az)
        AL.append(alt)
    p = result["pose"]
    return {"seq": result["seq"], "t0": night.t0, "W": night.W, "H": night.H,
            "n_stars": len(result["matches"]), "x": np.array(X), "y": np.array(Y),
            "az": np.concatenate(AZ), "alt": np.concatenate(AL),
            "pose": [p["d_az"], p["d_pitch"], p["d_roll"]], "k_ratio": p["k_ratio"], "k1": p["k1"]}


def fit(obs: list[dict], cam: dict, centre: bool = True) -> dict:
    """Shared (cx, cy, k_ratio, k1) and one pose per night, by robust least squares over every
    matched point of every night. centre=False holds the centre at the frame's middle, for the
    residual to compare against."""
    W, H = obs[0]["W"], obs[0]["H"]
    k0 = initial_k(cam, W)
    n = len(obs)
    x0 = np.r_[[v for o in obs for v in o["pose"]],
               np.median([o["k_ratio"] for o in obs]) * k0, np.median([o["k1"] for o in obs]),
               [0.0, 0.0] if centre else []]

    def resid(q):
        k, k1 = q[3 * n], q[3 * n + 1]
        c = {**cam, "cx": q[3 * n + 2], "cy": q[3 * n + 3]} if centre else cam
        out = []
        for i, o in enumerate(obs):
            x, y = project_fisheye(c, o["az"], o["alt"], W, H, *q[3 * i: 3 * i + 3], k, k1)
            out.append(np.r_[x * W - o["x"], y * H - o["y"]])
        return np.nan_to_num(np.concatenate(out), nan=1e3)

    s = least_squares(resid, x0, loss="soft_l1", f_scale=4.0, x_scale="jac", max_nfev=2000)
    per_night = np.split(s.fun, np.cumsum([2 * len(o["x"]) for o in obs])[:-1])
    e = np.concatenate([np.hypot(*np.split(f, 2)) for f in per_night])   # (dx..., dy...) each
    q = s.x
    return {"cx": float(q[3 * n + 2]) if centre else 0.0, "cy": float(q[3 * n + 3]) if centre else 0.0,
            "k_ratio": float(q[3 * n] / k0), "k1": float(q[3 * n + 1]),
            "poses": [list(map(float, q[3 * i: 3 * i + 3])) for i in range(n)],
            "median_px": float(np.median(e))}


def series(obs: list[dict], cam: dict) -> list[dict]:
    """A camera's nights (one frame size) as segments of constant centre; see the module."""
    obs = sorted(obs, key=lambda o: o["t0"])
    own = [fit([o], cam) for o in obs]
    segs, cur = [], [0]
    for i in range(1, len(obs)):
        ref = np.median([[own[j]["cx"], own[j]["cy"]] for j in cur], axis=0)
        off = [np.hypot(own[j]["cx"] - ref[0], own[j]["cy"] - ref[1]) for j in (i, i + 1)
               if j < len(obs)]
        if all(d > JUMP_PX for d in off):
            segs.append(cur)
            cur = []
        cur.append(i)
    segs.append(cur)
    out = []
    for seg in segs:
        o = [obs[j] for j in seg]
        n_stars = sum(x["n_stars"] for x in o)
        free, fixed = fit(o, cam), fit(o, cam, centre=False)
        enough = len(o) >= MIN_NIGHTS or n_stars >= MIN_STARS
        out.append({"W": o[0]["W"], "H": o[0]["H"], "from": int(o[0]["t0"]), "to": int(o[-1]["t0"]),
                    "cx": round(free["cx"], 1) if enough else 0.0,
                    "cy": round(free["cy"], 1) if enough else 0.0,
                    "k_ratio": round(free["k_ratio"], 4), "k1": round(free["k1"], 4),
                    "n_nights": len(o), "n_stars": n_stars,
                    "median_px": round(fixed["median_px"], 2),
                    "median_px_centred": round(free["median_px"], 2),
                    "nights": [x["seq"] for x in o],
                    "own": [[round(own[j]["cx"], 1), round(own[j]["cy"], 1)] for j in seg]})
    return out


def build(results: list[dict], night_of, cams: dict, dest: Path | str | None = None) -> list[dict]:
    """Every camera's series from solved `results`; `night_of(seq)` gives a result's Night."""
    by: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("status") != "solved" or not r.get("matches"):
            continue
        o = observations(r, night_of(r["seq"]))
        if o is not None:
            by.setdefault((r["camera"], o["W"], o["H"]), []).append(o)
    rows = []
    for (camera, W, H), obs in sorted(by.items()):
        rows += [{"camera": camera, **s} for s in series(obs, cams[camera])]
    if dest is not None:
        Path(dest).write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def lookup(rows: list[dict], camera: str, W: int, H: int, epoch: float) -> tuple[float, float]:
    """(cx, cy) for a camera's frame size on a date: the segment whose nights bracket it, else
    the nearest one before it, else the first after it; (0, 0) if the camera has none."""
    mine = sorted((r for r in rows if r["camera"] == camera and r["W"] == W and r["H"] == H),
                  key=lambda r: r["from"])
    if not mine:
        return 0.0, 0.0
    before = [r for r in mine if r["from"] <= epoch]
    r = before[-1] if before else mine[0]
    return r["cx"], r["cy"]
