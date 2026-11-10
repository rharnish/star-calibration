"""How the lens and centre improve with nights, from disjoint halves (no shared data).

For each camera with >= 2n solved nights, random disjoint n-night subsets A and B are fitted
separately (intrinsics.fit: shared centre and lens, one pose per night). One estimate's error is
the A-B difference over sqrt(2). Reported per n:
  centre: |c_A - c_B| / sqrt(2), px
  shape_top / shape_low: the lens-and-centre difference that no pose can absorb -- each fit's
    pixel directions over a grid (pose zero), the best rotation between them removed (Kabsch),
    median leftover angle in the grid's pixels, / sqrt(2). top = top 60% of the frame (where
    the stars are), low = the bottom third (below them).
Writes ../static/split_half.json.

    .venv/bin/python examples/hpwren_article/scripts/split_half.py
"""
import collections, itertools, json, random, sys
from pathlib import Path
import numpy as np
from star_calibration import intrinsics
from star_calibration.fisheye import initial_k, unproject_fisheye
from star_calibration.hpwren import cameras
from star_calibration.hpwren.calibrate import night, summary

random.seed(3)
R = [r for r in summary() if r.get("status") == "solved" and "source" not in r and "_N_" not in r["seq"]]
cams = cameras()
by = collections.defaultdict(list)
for r in R:
    by[r["camera"]].append(r)
GRIDS = {"top": np.meshgrid(np.linspace(0.05, 0.95, 13), np.linspace(0.05, 0.6, 7)),
         "low": np.meshgrid(np.linspace(0.05, 0.95, 13), np.linspace(0.67, 0.95, 5))}

def dirs(cam, W, H, f, g):
    gx, gy = GRIDS[g]
    c = {**cam, "cx": f["cx"], "cy": f["cy"]}
    az, el = unproject_fisheye(c, gx, gy, W, H, 0, 0, 0, f["k_ratio"] * initial_k(cam, W), f["k1"])
    az, el = np.radians(az).ravel(), np.radians(el).ravel()
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)

def shape_px(cam, W, H, fa, fb, g):
    a, b = dirs(cam, W, H, fa, g), dirs(cam, W, H, fb, g)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok], b[ok]
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ D @ Vt                     # rotation taking a onto b
    ang = np.degrees(np.arccos(np.clip(((a @ Rm.T) * b).sum(1), -1, 1)))
    deg_per_px = np.degrees(1 / (fa["k_ratio"] * initial_k(cam, W)))
    return float(np.median(ang) / deg_per_px)

rows = []
for c, rs in by.items():
    if len(rs) < 2:
        continue
    cam = cams[c]
    obs = [o for o in (intrinsics.observations(r, night(r["seq"])) for r in rs) if o]
    if len(obs) < 2:
        continue
    W, H = obs[0]["W"], obs[0]["H"]
    for n in range(1, len(obs) // 2 + 1):
        seen = set()
        for _ in range(40):
            idx = random.sample(range(len(obs)), 2 * n)
            A, B = tuple(sorted(idx[:n])), tuple(sorted(idx[n:]))
            key = tuple(sorted((A, B)))
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > 8:
                break
            fa = intrinsics.fit([obs[i] for i in A], cam)
            fb = intrinsics.fit([obs[i] for i in B], cam)
            rows.append({"camera": c, "n": n, "N": len(obs),
                         "stars": [sum(obs[i]["n_stars"] for i in A), sum(obs[i]["n_stars"] for i in B)],
                         "centre": float(np.hypot(fa["cx"] - fb["cx"], fa["cy"] - fb["cy"]) / np.sqrt(2)),
                         "dk": abs(fa["k_ratio"] - fb["k_ratio"]) / np.sqrt(2),
                         "shape_top": shape_px(cam, W, H, fa, fb, "top") / np.sqrt(2),
                         "shape_low": shape_px(cam, W, H, fa, fb, "low") / np.sqrt(2),
                         "night_px": [fa["median_px"], fb["median_px"]]})
    print(c, len(obs), flush=True)
json.dump(rows, open(Path(__file__).resolve().parents[1] / "static" / "split_half.json", "w"))
for n in sorted({r["n"] for r in rows}):
    s = [r for r in rows if r["n"] == n]
    f = lambda k: (np.median([r[k] for r in s]), np.percentile([r[k] for r in s], 90))
    print(n, len(s), len({r['camera'] for r in s}), *(f"{k} {f(k)[0]:.2f}/{f(k)[1]:.2f}" for k in ("centre", "shape_top", "shape_low")))
