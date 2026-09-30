"""Ablation: how a solve's stars and tracks move its fit, by distance from the optical centre.

Two studies of a solved night, each taking the stored result and the `Night` it came from:

  * `refit_experiments` keeps the solve's star-to-track matches and refits subsets of them,
    starting from the solved pose (`intrinsics.fit` on one night). No search runs and nothing
    is accepted or rejected, so the only thing that changes is which points the fit sees:
    each star left out in turn, only the inner or only the outer stars, random subsets of n
    stars, every track shortened to its middle. Each is run with the optical centre held at
    the camera's value (the parameters a solve fits) and with it free.
  * `resolve_variants` deletes tracks from the night -- by their distance from the optical
    centre, or at random, matched or not -- and runs `solve.solve_wide` again with the
    acceptance tests loosened (`min_stars`, `min_tracks`), to see whether the search still
    finds the same pose.

A star's distance is `theta_deg`, the angle between it and the boresight at the solved pose,
which is comparable across lenses; `r_px`, its track's median distance from the optical
centre in pixels, is kept beside it. Results are JSON-ready dicts; `hpwren.ablate` runs them
over the cache and docs/records.md lists their fields.
"""
from __future__ import annotations

import dataclasses
import zlib

import numpy as np

from . import intrinsics
from .fisheye import _basis, _unit
from .solve import Night, solve_wide

POSE = ("d_az", "d_pitch", "d_roll")
PARAMS = (*POSE, "k_ratio", "k1", "cx", "cy")
SUBSET_SIZES = (4, 6, 8, 12, 16)
SUBSET_DRAWS = 20
RADIAL_FRACTIONS = (1 / 3, 1 / 2, 2 / 3)
SHORTEN = (0.25, 0.5)
MIN_FIT_STARS = 3          # fewer and a 7-parameter fit has too little to hold it


def _seed(seq: str | None) -> int:
    return zlib.crc32((seq or "").encode())


def stars(result: dict, night: Night, obs: dict) -> list[dict]:
    """Each matched star: name, magnitude, point count, theta_deg and r_px (see the module),
    in `obs["names"]` order."""
    c, p = night.cam, result["pose"]
    boresight, _, _ = _basis(c, p["d_az"], p["d_pitch"], p["d_roll"])
    theta = np.degrees(np.arccos(np.clip(_unit(obs["az"], obs["alt"]) @ boresight, -1, 1)))
    x0, y0 = night.W / 2 + c.get("cx", 0.0), night.H / 2 + c.get("cy", 0.0)
    r = np.hypot(obs["x"] - x0, obs["y"] - y0)
    out = []
    for s, name in enumerate(obs["names"]):
        m = obs["star"] == s
        out.append({"name": name, "mag": (result.get("mags") or {}).get(name),
                    "n_points": int(m.sum()), "theta_deg": float(np.median(theta[m])),
                    "r_px": float(np.median(r[m]))})
    return out


def _subset(obs: dict, mask: np.ndarray) -> dict:
    per_point = ("x", "y", "az", "alt", "star")
    return {**obs, **{k: obs[k][mask] for k in per_point},
            "n_stars": int(len(np.unique(obs["star"][mask])))}


def refit(obs: dict, cam: dict, mask: np.ndarray | None = None, centre: bool = False) -> dict:
    """The fit of the points in `mask` (all of them if None): the parameters, the median
    residual, and how many stars and points it saw."""
    sub = obs if mask is None else _subset(obs, mask)
    f = intrinsics.fit([sub], cam, centre=centre)
    cx, cy = (f["cx"], f["cy"]) if centre else (cam.get("cx", 0.0), cam.get("cy", 0.0))
    return {**dict(zip(POSE, f["poses"][0])), "k_ratio": f["k_ratio"], "k1": f["k1"],
            "cx": float(cx), "cy": float(cy), "median_px": f["median_px"],
            "n_stars": sub["n_stars"], "n_points": int(len(sub["x"]))}


def _central(obs: dict, frac: float) -> np.ndarray:
    """The middle `frac` of each star's points (they are in time order)."""
    keep = np.zeros(len(obs["x"]), bool)
    for s in np.unique(obs["star"]):
        i = np.flatnonzero(obs["star"] == s)
        n = max(2, int(round(frac * len(i))))
        a = (len(i) - n) // 2
        keep[i[a:a + n]] = True
    return keep


def refit_experiments(result: dict, night: Night, centre: bool = False,
                      sizes=SUBSET_SIZES, draws: int = SUBSET_DRAWS) -> dict:
    """Every refit experiment on one solved night, with the centre held (or free): see the
    module. Each row is `refit`'s output plus `kind` and what it kept or dropped."""
    obs = intrinsics.observations(result, night)
    cam = night.cam
    table = stars(result, night, obs)
    theta = np.array([s["theta_deg"] for s in table])
    n = len(table)
    rows = [{"kind": "baseline", **refit(obs, cam, centre=centre)}]
    for s, st in enumerate(table):
        rows.append({"kind": "leave_one_out", "drop": st["name"],
                     **refit(obs, cam, obs["star"] != s, centre)})
    order = np.argsort(theta)
    for q in RADIAL_FRACTIONS:
        m = int(round(q * n))
        if m < MIN_FIT_STARS:
            continue
        for side, idx in (("inner", order[:m]), ("outer", order[n - m:])):
            rows.append({"kind": side, "fraction": q,
                         "theta_max": float(theta[idx].max()), "theta_min": float(theta[idx].min()),
                         **refit(obs, cam, np.isin(obs["star"], idx), centre)})
    rng = np.random.default_rng(_seed(result.get("seq")))
    for k in sizes:
        if k >= n:
            continue
        for d in range(draws):
            idx = rng.choice(n, k, replace=False)
            rows.append({"kind": "subset", "size": k, "draw": d,
                         "theta_mean": float(theta[idx].mean()),
                         **refit(obs, cam, np.isin(obs["star"], idx), centre)})
    for frac in SHORTEN:
        rows.append({"kind": "shorten", "fraction": frac,
                     **refit(obs, cam, _central(obs, frac), centre)})
    return {"seq": result["seq"], "camera": result["camera"], "W": night.W, "H": night.H,
            "centre_free": centre, "solved": {**result["pose"], "cx": result.get("cx", 0.0),
                                              "cy": result.get("cy", 0.0),
                                              "median_px": result["median_px"]},
            "stars": table, "rows": rows}


def track_radius(night: Night) -> np.ndarray:
    """Each track's median distance from the optical centre, px."""
    x0, y0 = night.W / 2 + night.cam.get("cx", 0.0), night.H / 2 + night.cam.get("cy", 0.0)
    return np.array([np.median([np.hypot(v[0] - x0, v[1] - y0) for v in t.values()])
                     for t in night.tracks])


def variants(night: Night, seeds: int = 3, fractions=(0.75, 0.5, 0.25)) -> dict[str, list[int]]:
    """Name -> the track indices a re-solve keeps: all of them; all but one radial third
    (by `track_radius`); a random `fraction`, `seeds` times."""
    n = len(night.tracks)
    order = np.argsort(track_radius(night))
    thirds = np.array_split(order, 3)
    out = {"all": list(range(n))}
    for name, drop in zip(("drop_inner_third", "drop_middle_third", "drop_outer_third"), thirds):
        out[name] = sorted(set(range(n)) - set(drop.tolist()))
    rng = np.random.default_rng(_seed(night.label))
    for f in fractions:
        for s in range(seeds):
            out[f"keep_{round(100 * f)}_s{s}"] = sorted(rng.choice(n, round(f * n), replace=False).tolist())
    return out


def resolve_variants(result: dict, night: Night, min_stars: int = 4, min_tracks: int = 4,
                     same_deg: float = 0.2, which: dict[str, list[int]] | None = None) -> dict:
    """Re-solve `night` under each of `variants` (or `which`), loosened to `min_stars` and
    `min_tracks`. Each row says whether it solved, how far its pose moved from `result`'s,
    and whether it is the same pose (every angle within `same_deg`)."""
    ref = result["pose"]
    ref_stars = set(result["matches"])
    rows = []
    for name, keep in (which or variants(night)).items():
        r = solve_wide(dataclasses.replace(night, tracks=[night.tracks[i] for i in keep]),
                       min_stars=min_stars, min_tracks=min_tracks)
        row = {"variant": name, "n_tracks": len(keep), "status": r["status"],
               "reason": r.get("reason"), "found_by": r.get("found_by")}
        if "pose" in r:
            p = r["pose"]
            delta = {k: p[k] - ref[k] for k in (*POSE, "k_ratio", "k1")}
            row.update(n_stars=r["n_stars"], median_px=r["median_px"], pose=p, delta=delta,
                       same_pose=all(abs(delta[k]) <= same_deg for k in POSE),
                       shared_stars=len(ref_stars & set(r["matches"])))
        rows.append(row)
    return {"seq": result["seq"], "camera": result["camera"], "n_tracks": len(night.tracks),
            "n_stars": result["n_stars"], "min_stars": min_stars, "min_tracks": min_tracks,
            "rows": rows}
