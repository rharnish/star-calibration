"""Accept a solve because another night agrees with it, not because it named enough stars.

`solve.py` accepts a pose at >= 8 stars and median residual < 3 px. The star count is a
proxy: it stands in for "enough independent evidence that this pose is not a coincidence",
which is a real thing to want, but it is a blunt proxy and it is measured on one night. It
rejected pi-s-mobo-c at 5 stars and 0.96 px -- a fit whose predicted arcs run visibly inside
the observed tracks -- while accepting any 8-star fit that happened to clear the bar.

There is a better test available, and it costs only another night's frames. Two nights of
the same camera are independent measurements of one physical pose. If a 5-star solve on
Tuesday and a 5-star solve on Thursday land on the same angles to a hundredth of a degree,
they are not both coincidences; the sky has moved on between them and noise does not
reproduce. If they disagree, no star count would have saved them.

So this scores a camera's solves by how well its nights agree, and reports what accepting at
a lower star count would actually cost. Agreement is measured on the boresight direction and
on roll, in degrees, between each solve and the nearest other solve of the same camera --
deliberately *not* on d_az alone, since a pitch error trades against an azimuth error.

Cameras do get re-aimed, which is the honest confound: a real disagreement can mean a real
move rather than a bad fit. That is why this reports agreement rather than silently
accepting, and why the window matters -- nights days apart are the useful comparison, and
`pose_ledger.py` is the thing that decides which pose applies to which fire.

    python -m src.figlib.stars.cross_night                 # the committed solves
    python -m src.figlib.stars.cross_night --wide          # solve_wide's results
    python -m src.figlib.stars.cross_night --min-stars 4   # re-solve low, then check
"""
from __future__ import annotations

import functools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import pole as POLE
from . import solve as S

ROOT = Path(__file__).resolve().parents[3]


def boresight(cam: dict, pose: dict) -> np.ndarray:
    return POLE.axes_from_pose(cam, pose["d_az"], pose["d_pitch"], pose["d_roll"])[2]


def disagreement(cam: dict, a: dict, b: dict) -> tuple[float, float]:
    """(boresight separation, roll difference) in degrees between two poses."""
    sep = math.degrees(math.acos(float(np.clip(
        np.dot(boresight(cam, a), boresight(cam, b)), -1, 1))))
    roll = abs((a["d_roll"] - b["d_roll"] + 180) % 360 - 180)
    return sep, roll


def pairs(results: list[dict]) -> list[dict]:
    """For every solve, its closest-in-time sibling solve of the same camera."""
    by_cam = defaultdict(list)
    for r in results:
        if r.get("status") == "solved" and r.get("pose"):
            t = S.SEQS.get(r["seq"], {}).get("t0")
            if t:
                by_cam[r["camera"]].append((t, r))
    out = []
    for camera, rows in by_cam.items():
        if len(rows) < 2:
            continue
        rows.sort()
        c = S.CAMS[camera]
        for i, (t, r) in enumerate(rows):
            j = min((k for k in range(len(rows)) if k != i), key=lambda k: abs(rows[k][0] - t))
            t2, r2 = rows[j]
            sep, roll = disagreement(c, r["pose"], r2["pose"])
            out.append({"camera": camera, "seq": r["seq"], "other": r2["seq"],
                        "days_apart": abs(t2 - t) / 86400.0,
                        "n_stars": r["n_stars"], "other_n_stars": r2["n_stars"],
                        "median_px": r["median_px"], "sep_deg": sep, "roll_deg": roll})
    return out


def report(results: list[dict]) -> None:
    ps = pairs(results)
    if not ps:
        print("no camera has two solves to compare")
        return
    near = [p for p in ps if p["days_apart"] <= 30]
    print(f"{len(ps)} solves with a sibling solve on the same camera "
          f"({len(near)} within 30 days)\n")
    print(f"{'stars':>9s} {'n':>4s} {'boresight sep':>14s} {'roll diff':>11s}")
    for lo, hi in ((4, 5), (6, 7), (8, 11), (12, 19), (20, 999)):
        g = [p for p in near if lo <= min(p["n_stars"], p["other_n_stars"]) <= hi]
        if not g:
            continue
        s = np.array([p["sep_deg"] for p in g])
        r = np.array([p["roll_deg"] for p in g])
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"{label:>9s} {len(g):4d} {np.median(s):9.3f} deg {np.median(r):8.3f} deg")
    print("\nworst disagreements within 30 days (a re-aim looks like this too):")
    for p in sorted(near, key=lambda p: -p["sep_deg"])[:8]:
        print(f"  {p['camera']:16s} {p['seq'][-28:]:28s} vs {p['other'][-28:]:28s} "
              f"{p['days_apart']:5.1f}d  sep {p['sep_deg']:6.3f} deg  "
              f"stars {p['n_stars']}/{p['other_n_stars']}")
    agree = [p for p in near if p["sep_deg"] < 0.25]
    print(f"\n{len(agree)}/{len(near)} agree within 0.25 deg of a sibling night")
    low = [p for p in near if min(p["n_stars"], p["other_n_stars"]) < 8]
    if low:
        ok = sum(1 for p in low if p["sep_deg"] < 0.25)
        print(f"of the {len(low)} pairs where one side has fewer than 8 stars, "
              f"{ok} agree within 0.25 deg")


if __name__ == "__main__":
    args = sys.argv[1:]
    src = S.DATA / ("solve_wide_summary.json" if "--wide" in args else "solve_summary.json")
    if "--min-stars" in args:
        n = int(args[args.index("--min-stars") + 1])
        seqs = sorted(p.name[len("tracks_"):-4] for p in S.DATA.glob("tracks_*.pkl"))
        from multiprocessing import Pool
        with Pool(4) as pool:
            res = list(pool.imap_unordered(functools.partial(S.solve_wide, min_stars=n), seqs))
    else:
        res = json.loads(src.read_text())
    report(res)
