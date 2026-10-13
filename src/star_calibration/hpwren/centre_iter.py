"""Optical centres by iteration: solve every block through the current centres, refit the
centres from those solves, and repeat until they stop moving. Nothing here touches solves/ or
the shipped intrinsics.json.

The input is the `sky` variant's tracks (detect_trial keeps them in detect-trial/tracks/):
the shipped blob sizes, searched over everything above the DEM skyline. Those tracks are made
once. The mask they were found under was drawn through a seed pose solved under the old
centre, and pose and centre together describe the same pixels, so a new centre needs a new
solve but not new tracks.

Round 0 solves under the shipped centres. Round k+1 solves under the centres fitted from
round k by `intrinsics.series` (one shared centre and lens per camera segment, a pose per
night). A solve takes part in the fit only if no sibling night of the same camera disagrees
with it by more than `AGREE_DEG`, so a wrong fit can't drag its camera's centre after it.

Each round is scored three ways:
  * solved blocks, stars and median residual;
  * cross-night agreement, median over every solve with a sibling night;
  * held-out residual: for each camera with 3 or more nights in a segment, each night in turn
    is left out, the centre and lens are fitted from the others, and the left-out night's pose
    alone is fitted through them. Its median residual measures the centre on data that did
    not make it. It should fall from round to round, and is compared with the frame's middle.

The loop stops when no centre of a camera with 2 or more nights moves by more than
`CONVERGED_PX`, or after `MAX_ROUNDS`.

    python -m star_calibration.hpwren.centre_iter run       # rounds into centre-iter/
    python -m star_calibration.hpwren.centre_iter report
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .. import intrinsics
from ..cross_night import pairs
from ..fisheye import initial_k, project_fisheye
from ..solve import Night, solve_wide
from . import cache_dir, camera, cameras, intrinsics_path, nights
from . import detect_trial

AGREE_DEG = 0.1
CONVERGED_PX = 1.0
MAX_ROUNDS = 5


def root() -> Path:
    return cache_dir() / "centre-iter"


def round_dir(k: int) -> Path:
    return root() / f"round{k}"


def blocks() -> list[str]:
    """Every block with sky tracks."""
    d = cache_dir() / "detect-trial" / "tracks"
    return sorted(p.stem[len("tracks_"):-len("_sky")] for p in d.glob("tracks_*_sky.pkl"))


def night(seq: str, intr: Path) -> Night:
    """A block's sky tracks as the solver's input, its centre from `intr`."""
    import star_calibration.hpwren as hp
    hp.intrinsics_path = lambda: intr   # camera() reads its centres through this
    d = pickle.load(open(detect_trial.tracks_path(seq, "sky"), "rb"))
    s = nights.sequences()[seq]
    W, H = d["WH"]
    return Night(camera=s["camera"], cam=camera(s["camera"], W, H, s["t0"]), t0=s["t0"],
                 tracks=d["tracks"], W=W, H=H, label=seq)


def _solve(job):
    seq, intr = job
    try:
        r = solve_wide(night(seq, intr))
    except Exception as exc:   # one bad block shouldn't sink the round
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    r["camera"] = nights.sequences()[seq]["camera"]
    return r


def agreeing(results: list[dict]) -> tuple[list[dict], dict]:
    """The solves no sibling night disagrees with by more than AGREE_DEG, and each solve's
    nearest-sibling pair."""
    t0 = {s: v["t0"] for s, v in nights.sequences().items()}
    near = {p["seq"]: p for p in pairs(results, t0, cameras())}
    keep = [r for r in results if r.get("status") == "solved"
            and (r["seq"] not in near or near[r["seq"]]["sep_deg"] <= AGREE_DEG)]
    return keep, near


def held_out(rows: list[dict], obs_by: dict) -> list[dict]:
    """Leave-one-night-out residuals for every segment with 3+ nights; see the module."""
    cams = cameras()
    out = []
    for row in rows:
        obs = [obs_by[s] for s in row["nights"] if s in obs_by]
        if len(obs) < 3:
            continue
        cam = cams[row["camera"]]
        for i, o in enumerate(obs):
            rest = obs[:i] + obs[i + 1:]
            line = {"camera": row["camera"], "seq": o["seq"]}
            for name, free in (("centred", True), ("middle", False)):
                f = intrinsics.fit(rest, cam, centre=free)
                c = {**cam, "cx": f["cx"], "cy": f["cy"]}
                k = f["k_ratio"] * initial_k(cam, o["W"])

                def r(q):
                    x, y = project_fisheye(c, o["az"], o["alt"], o["W"], o["H"], *q, k, f["k1"])
                    return np.nan_to_num(np.r_[x * o["W"] - o["x"], y * o["H"] - o["y"]], nan=1e3)
                s = least_squares(r, o["pose"], loss="soft_l1", f_scale=4.0)
                line[name] = float(np.median(np.hypot(*np.split(s.fun, 2))))
            out.append(line)
    return out


def fit_round(k: int) -> dict:
    """Centres from round k's solves into round k+1's intrinsics.json, with round k's scores."""
    results = json.loads((round_dir(k) / "results.json").read_text())
    intr = round_dir(k) / "intrinsics.json"
    keep, near = agreeing(results)
    nights_of = {r["seq"]: night(r["seq"], intr) for r in keep}
    round_dir(k + 1).mkdir(parents=True, exist_ok=True)
    rows = intrinsics.build(keep, nights_of.get, cameras(), round_dir(k + 1) / "intrinsics.json")
    obs_by = {r["seq"]: intrinsics.observations(r, nights_of[r["seq"]]) for r in keep}
    loo = held_out(rows, obs_by)
    solved = [r for r in results if r.get("status") == "solved"]
    seps = sorted(p["sep_deg"] for p in near.values())
    old = {(r["camera"], r["W"], r["H"]): r for r in json.loads(intr.read_text())}
    moves = [float(np.hypot(r["cx"] - old[key]["cx"], r["cy"] - old[key]["cy"]))
             for r in rows if r["n_nights"] >= 2
             for key in [(r["camera"], r["W"], r["H"])] if key in old]
    score = {
        "round": k, "blocks": len(results), "solved": len(solved),
        "in_fit": len(keep), "stars": int(sum(r["n_stars"] for r in solved)),
        "median_px": float(np.median([r["median_px"] for r in solved])),
        "agree_median_deg": float(np.median(seps)) if seps else None,
        "agree_p90_deg": float(np.percentile(seps, 90)) if seps else None,
        "n_pairs": len(seps),
        "held_out_centred_px": float(np.median([x["centred"] for x in loo])) if loo else None,
        "held_out_middle_px": float(np.median([x["middle"] for x in loo])) if loo else None,
        "n_held_out": len(loo),
        "segments": len(rows), "multi_night": sum(r["n_nights"] >= 2 for r in rows),
        "with_centre": sum(1 for r in rows if r["cx"] or r["cy"]),
        "max_move_px": max(moves) if moves else None,
        "median_move_px": float(np.median(moves)) if moves else None,
    }
    (round_dir(k) / "held_out.json").write_text(json.dumps(loo, indent=1) + "\n")
    (round_dir(k) / "score.json").write_text(json.dumps(score, indent=1) + "\n")
    return score


def run(start: Path | None = None) -> None:
    seqs = blocks()
    k = 0
    if not (round_dir(0) / "intrinsics.json").exists():
        round_dir(0).mkdir(parents=True, exist_ok=True)
        (round_dir(0) / "intrinsics.json").write_text((start or intrinsics_path()).read_text())
    while k < MAX_ROUNDS:
        d = round_dir(k)
        if not (d / "results.json").exists():
            print(f"round {k}: solving {len(seqs)} blocks", flush=True)
            with ProcessPoolExecutor(4) as pool:
                res = list(pool.map(_solve, [(s, d / "intrinsics.json") for s in seqs]))
            (d / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        score = fit_round(k)
        print(json.dumps(score), flush=True)
        if k > 0 and score["max_move_px"] is not None and score["max_move_px"] <= CONVERGED_PX:
            break
        k += 1
    print("DONE", flush=True)


def report() -> None:
    for d in sorted(root().glob("round*"), key=lambda p: int(p.name[5:])):
        p = d / "score.json"
        if p.exists():
            s = json.loads(p.read_text())
            print(f"round {s['round']}: {s['solved']}/{s['blocks']} solved, {s['stars']} stars, "
                  f"median {s['median_px']:.2f} px; agree {s['agree_median_deg']:.4f} deg "
                  f"(p90 {s['agree_p90_deg']:.3f}, {s['n_pairs']}); held out "
                  f"{s['held_out_centred_px'] or 0:.2f} px vs middle {s['held_out_middle_px'] or 0:.2f} "
                  f"({s['n_held_out']}); centres {s['with_centre']}/{s['segments']}, moved max "
                  f"{s['max_move_px'] or 0:.1f} median {s['median_move_px'] or 0:.1f} px")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "report":
        report()
    else:
        print(__doc__)
