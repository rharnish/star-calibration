"""plume's far-ridge skyline check (src/figlib/stars/skyline_check.py), re-run on the v0.4.0
cache solves instead of plume's v0.2.1 ledger, for the article's External validation claim.

Per camera: the newest solved CDN Q1 night whose day still has daytime (Q4, 09:00-) frames on
the CDN. The DEM skyline, drawn through that night's pose, lens and centre, is compared per
column with the strongest bright-above/dark-below edge within +-SEARCH_PX, only where the
skyline is a ridge 10-78 km out (as plume). Of a few frames, the one with the most measured
columns is kept. Residual = edge row - predicted row (+ = the line sits above the ridge).

    .venv/bin/python examples/hpwren_article/scripts/skyline_v040.py   # -> ../static/skyline_v040.json

The per-camera rows are printed, not saved: the article quotes only the aggregate.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples"))
import terrain  # noqa: E402

from star_calibration.hpwren import cameras, nights  # noqa: E402
from star_calibration.hpwren.calibrate import summary  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "static"
MIN_KM, MAX_KM, SEARCH_PX, PAD_DEG, MIN_COLS = 10.0, 78.0, 25, 14.0, 100
FIRST_DAY = "20260718"        # the CDN keeps about 90 days
N_FETCH, EVERY = 60, 10       # 09:00-10:00, try every 10th frame


def skyline_km(cam, step_deg=0.05, step_m=60.0, max_km=80.0):
    azs = np.arange(cam["az"] - cam["fov"] / 2 - PAD_DEG, cam["az"] + cam["fov"] / 2 + PAD_DEG, step_deg)
    dist = np.arange(60.0, max_km * 1000, step_m)
    lat, lon = terrain.destination(cam["lat"], cam["lon"], azs[:, None], dist[None, :])
    h = terrain.terrain_height(lat, lon)
    ang = np.degrees(np.arctan2(h - terrain.eye_height(cam) - dist ** 2 * (1 - terrain.K_REFR)
                                / (2 * terrain.R_EARTH), dist))
    j = ang.argmax(axis=1)
    return azs % 360, ang[np.arange(len(azs)), j], dist[j] / 1000


def edges(img, cols, centre):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    drop = g[:-1] - g[1:]
    H = g.shape[0]
    out = np.full(len(cols), np.nan)
    for i, (c, m) in enumerate(zip(cols, centre)):
        a, b = int(m) - SEARCH_PX, int(m) + SEARCH_PX + 1
        if a < 60 or b >= H - 1:
            continue
        col = drop[a:b, c]
        floor = np.median(np.abs(drop[a - 60:a, c]))
        k = int(np.argmax(col))
        if col[k] > 4 * floor + 1:
            out[i] = a + k + 0.5
    return out


def targets():
    by = defaultdict(list)
    for r in summary():
        if ("source" in r or r["status"] != "solved" or "_Q1_" not in r["seq"]
                or r.get("W") != 3072):
            continue
        day = r["seq"].split("_")[1]
        if day >= FIRST_DAY:
            by[r["camera"]].append((day, r))
    return {c: sorted(v, key=lambda t: t[0], reverse=True) for c, v in sorted(by.items())}


def measure(camera, day, r):
    nights.fetch(camera, day, q=4, n_frames=N_FETCH)
    paths = sorted((nights.frames_dir() / camera / f"{day}_Q4").glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        return None
    cam = cameras()[camera]
    cam_c = {**cam, "cx": r["cx"], "cy": r["cy"]}
    az, el, km = skyline_km(cam)
    best = None
    for p in paths[::EVERY]:
        img = cv2.imread(str(p))
        if img is None or img.shape[1] != 3072:
            continue
        H, W = img.shape[:2]
        x, y = terrain.project(cam_c, az, el, W, H, r["pose"])
        ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W)
        if ok.sum() < 10:
            return None
        o = np.argsort(x[ok])
        xs, ys, ks = x[ok][o], y[ok][o], km[ok][o]
        cols = np.arange(int(xs.min()) + 5, int(xs.max()) - 5, 2)
        rows = np.interp(cols, xs, ys)
        rng = np.interp(cols, xs, ks)
        e = edges(img, cols, rows)
        k = np.isfinite(e) & (rng > MIN_KM) & (rng < MAX_KM)
        if best is None or k.sum() > best["n_cols"]:
            res = e[k] - rows[k]
            best = {"camera": camera, "seq": r["seq"], "frame": p.name, "n_cols": int(k.sum()),
                    "median_px": float(np.median(res)) if k.sum() else None,
                    "mean_abs_px": float(np.mean(np.abs(res))) if k.sum() else None,
                    "p90_abs_px": float(np.percentile(np.abs(res), 90)) if k.sum() else None}
    return best


def main():
    rows = []
    for camera, solves in targets().items():
        got = None
        for day, r in solves[:3]:              # newest first; fall back if no daytime frames
            got = measure(camera, day, r)
            if got:
                break
        if not got:
            print(f"{camera:18s} no daytime frames / skyline", flush=True)
            continue
        rows.append(got)
        med = got["median_px"]
        print(f"{camera:18s} {got['seq'][7:15]} {got['n_cols']:5d} cols  "
              f"median {med if med is None else f'{med:+5.1f}'}", flush=True)
    used = [r for r in rows if r["n_cols"] >= MIN_COLS]
    med = np.array([r["median_px"] for r in used])
    agg = {"n_cameras": len(used), "median_of_medians_px": float(np.median(med)),
           "mean_abs_of_medians_px": float(np.mean(np.abs(med))),
           "frac_within_3px": float(np.mean(np.abs(med) <= 3)),
           "median_mean_abs_px": float(np.median([r["mean_abs_px"] for r in used]))}
    print(f"\n{json.dumps(agg, indent=1)}")
    (STATIC / "skyline_v040.json").write_text(json.dumps(
        {"min_km": MIN_KM, "max_km": MAX_KM, "search_px": SEARCH_PX, "min_cols": MIN_COLS,
         "aggregate": agg}, indent=1) + "\n")


if __name__ == "__main__":
    main()
