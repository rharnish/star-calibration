"""Trial of two detector changes, on a handful of blocks, without touching solves/.

The colour detector keeps blobs of 1-16 px, and both detectors look only at the top 35% of the
frame. Stars brighter than about magnitude 2.3 make bigger blobs than that on the colour units,
so Betelgeuse, Rigel, Vega and the like never become tracks; and on a camera whose horizon is
lower than 35% of the way down, the sky below that line is never searched.

Each block is tracked and solved four ways:

  base   the detector as it ships
  big    larger blobs allowed: 1-80 px on colour units, 2-120 px on monochrome
  low    the top 60% of the frame searched instead of 35%
  both   big and low together

Results go to the cache's detect-trial/results.json (a block-variant already there is not
re-run), and the report compares each variant with base: tracks, stars, how many of the matched
stars are brighter than magnitude 2.3, and how far the pose moved. Where a block has no base
row, its row in solves/summary.json stands in: base is the detector as it ships, so the two are
the same solve (they agree on all of BLOCKS).

A fifth variant, `sky`, is not part of the default set: it keeps the shipped blob sizes but
replaces the top-35% band with everything above the camera's DEM skyline, drawn through a
seed pose. The seed is the block's `big` solve, else its base solve, else the published pose
(with a wider margin). The skylines come from plume-triangulation's terrain code, computed
once into detect-trial/skylines.json by detect-trial/skylines_from_plume.py (plume's venv has
rasterio; this one does not). The idea: big blobs find the pose, the DEM says where the sky
is, and the fainter, sharper stars over the whole sky fix it precisely.

`--all` runs every CDN block in solves/summary.json except the "no images" failures, and
`--variant=v` (repeatable) limits the run to the variants named. `compare` sums one variant over every block
it ran on: failures rescued, solves lost, how far the poses moved, and whether the rescued
solves agree with another night of the same camera.

    python -m star_calibration.hpwren.detect_trial run [--all] [--variant=v ...] [block ...]
    python -m star_calibration.hpwren.detect_trial report
    python -m star_calibration.hpwren.detect_trial compare [variant]
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from ..cross_night import disagreement, pairs
from ..fisheye import K1, K_RATIO, initial_k, unproject_fisheye
from ..solve import Night, solve_wide
from ..tracks import clean, collect
from . import cache_dir, camera, cameras, nights
from .calibrate import BANNER_PX, link_recipe, summary

BLOCKS = [
    # solved, with bright stars in view
    "hpwren_20261007_Q1_sm-e-mobo-c", "hpwren_20261007_Q1_sm-e-mobo-m",
    "hpwren_20261007_Q1_bi-w-mobo-c", "hpwren_20260911_Q1_hp-w-mobo-c",
    "hpwren_20260911_Q1_vo-e-mobo-c", "hpwren_20260911_Q1_cp-w-mobo-c",
    "hpwren_20260911_Q1_bm-s-mobo-c", "hpwren_20260821_Q1_starr-n-mobo-c",
    # stars seen, too few to accept
    "hpwren_20260911_Q1_pi-s-mobo-c", "hpwren_20260911_Q1_sjh-n-mobo-c",
    "hpwren_20260912_Q1_tdlln-n-mobo-c", "hpwren_20260913_Q1_blkrc-s-mobo-c",
    "hpwren_20260913_Q1_buff-s-mobo-c", "hpwren_20260911_Q1_starr-n-mobo-m",
    "hpwren_20260911_Q1_lp-s-mobo-m",
]
BIG = {False: {"area_range": (1, 80)}, True: {"area_range": (2, 120)}}   # by mono
LOW = {"sky_frac": 0.60}
VARIANTS = ["base", "big", "low", "both"]
BRIGHT_MAG = 2.3
# How far above the DEM skyline the sky starts: trees and masts stand above the DEM's ridge,
# and a solved seed pose is good to a few hundredths of a degree, the published one is not.
SKY_MARGIN_DEG = {"solved": 0.5, "published": 2.0}


def results_path():
    return cache_dir() / "detect-trial" / "results.json"


def tracks_path(seq: str, variant: str):
    """Where a `sky` run keeps its cleaned tracks, for centre_iter to re-solve from."""
    return cache_dir() / "detect-trial" / "tracks" / f"tracks_{seq}_{variant}.pkl"


def detect_kw(variant: str, mono: bool) -> dict:
    kw = {}
    if variant in ("big", "both"):
        kw |= BIG[mono]
    if variant in ("low", "both"):
        kw |= LOW
    return kw


def skylines() -> dict:
    p = cache_dir() / "detect-trial" / "skylines.json"
    return json.loads(p.read_text()) if p.exists() else {}


def sky_mask(cam: dict, pose: dict, W: int, H: int, sky: dict, margin_deg: float,
             step: int = 8) -> np.ndarray:
    """Frame-sized bool mask of the pixels more than `margin_deg` above the DEM skyline."""
    xs = (np.arange(0, W, step) + step / 2) / W
    ys = (np.arange(0, H, step) + step / 2) / H
    X, Y = np.meshgrid(xs, ys)
    az, el = unproject_fisheye(cam, X, Y, W, H, pose["d_az"], pose["d_pitch"], pose["d_roll"],
                               pose["k_ratio"] * initial_k(cam, W), pose["k1"])
    ridge = np.interp(az, sky["az"], sky["elev"], period=360.0)
    m = (el > ridge + margin_deg).astype(np.uint8)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def seed(seq: str) -> tuple[str, dict]:
    """The pose the sky mask is drawn through: big's solve, else base's, else published."""
    for r in (next((r for r in load() if r["seq"] == seq and r["variant"] == "big"), None),
              bases().get(seq)):
        if r and r.get("status") == "solved" and r.get("pose"):
            return ("big" if r.get("variant") == "big" else "base"), r["pose"]
    return "published", {"d_az": 0.0, "d_pitch": 0.0, "d_roll": 0.0, "k_ratio": K_RATIO, "k1": K1}


def run_one(seq: str, variant: str) -> dict:
    s = nights.sequences()[seq]
    link = link_recipe(seq)
    mono = link["linker"] == "predictive"
    frames = nights.read_frames(seq)
    extra = {}
    if variant == "sky":
        H, W = cv2.imdecode(np.frombuffer(frames[0][2], np.uint8), cv2.IMREAD_GRAYSCALE).shape
        src, pose = seed(seq)
        sky = skylines().get(s["camera"])
        if sky is None:
            return {"seq": seq, "variant": variant, "status": "failed", "reason": "no DEM skyline"}
        mask = sky_mask(camera(s["camera"], W, H, s["t0"]), pose, W, H, sky,
                        SKY_MARGIN_DEG["published" if src == "published" else "solved"])
        rows = np.flatnonzero(mask.any(axis=1))
        kw = {"sky_mask": mask}
        extra = {"seed": src, "seed_pose": pose, "mask_frac": float(mask.mean()),
                 "mask_bottom": float((rows[-1] + 1) / H) if len(rows) else 0.0,
                 "site_problem": sky.get("site_problem")}
    else:
        kw = detect_kw(variant, mono)
    linked, decoded = collect(frames, s["lat"], s["lon"], mono=mono,
                              keep_frames=False, max_gap_s=link["max_gap_s"],
                              label=f"{seq} {variant}", detect=kw)
    kw = {k: v for k, v in kw.items() if k != "sky_mask"}
    H, W = next(iter(decoded.values()))[1].shape[:2] if decoded else (2048, 3072)
    tracks, counts = clean(linked, banner_px=BANNER_PX)
    n = Night(camera=s["camera"], cam=camera(s["camera"], W, H, s["t0"]), t0=s["t0"],
              tracks=tracks, W=W, H=H, label=f"{seq} {variant}")
    r = solve_wide(n)
    if variant == "sky":
        p = tracks_path(seq, variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({"tracks": tracks, "WH": (W, H), "seed": extra["seed"],
                     "seed_pose": extra["seed_pose"]}, open(p, "wb"))
    mags = r.get("mags", {})
    return {"seq": seq, "variant": variant, "detect": kw, "n_linked": len(linked),
            "cleaning": counts, "status": r["status"], "reason": r.get("reason"),
            "n_tracks": len(tracks), "n_stars": r.get("n_stars"),
            "median_px": r.get("median_px"), "pose": r.get("pose"),
            "matches": sorted(r.get("matches", {})),
            "bright": sorted(m for m in r.get("matches", {}) if mags.get(m, 99) < BRIGHT_MAG)} | extra


def _run(job):
    try:
        return run_one(*job)
    except Exception as exc:   # one bad block shouldn't sink the batch
        return {"seq": job[0], "variant": job[1], "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}"}


def load() -> list[dict]:
    p = results_path()
    return json.loads(p.read_text()) if p.exists() else []


def all_blocks() -> list[str]:
    """Every CDN block in solves/summary.json that had frames to solve."""
    return [r["seq"] for r in summary() if r.get("source", "cdn") == "cdn"
            and not (r.get("reason") or "").startswith("no images") and "_N_" not in r["seq"]]


def bases() -> dict[str, dict]:
    """Base result by block: the trial's own base row, else the block's row in summary.json."""
    out = {r["seq"]: r for r in summary() if r.get("source", "cdn") == "cdn"}
    out |= {r["seq"]: r for r in load() if r["variant"] == "base"}
    return out


def run(blocks: list[str], variants: list[str] = VARIANTS) -> None:
    rows = load()
    # a sky row from before its tracks were kept is run again
    done = {(r["seq"], r["variant"]) for r in rows
            if r["variant"] != "sky" or tracks_path(r["seq"], "sky").exists()}
    jobs = [(b, v) for b in blocks for v in variants if (b, v) not in done]
    redo = set(jobs)
    rows = [r for r in rows if (r["seq"], r["variant"]) not in redo]
    print(f"{len(jobs)} block-variants to run")
    results_path().parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(_run, jobs):
            rows.append(r)
            # Atomic: a `sky` worker reads this file for its seed while it is being rewritten.
            tmp = results_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, indent=1, default=float) + "\n")
            tmp.replace(results_path())
            print(f"{r['seq']} {r['variant']}: {r['status']} {r.get('n_stars')} stars", flush=True)


def report() -> None:
    rows = load()
    by = {(r["seq"], r["variant"]): r for r in rows}
    base_of = bases()
    cams = cameras()
    print(f"{'block':36s} {'variant':6s} {'tracks':>6s} {'status':8s} {'stars':>5s} {'px':>5s} "
          f"{'bright':>6s} {'moved':>7s}  bright stars matched")
    for seq in dict.fromkeys(r["seq"] for r in rows):
        base = base_of.get(seq)
        for v in VARIANTS + ["sky"]:
            r = by.get((seq, v))
            if not r:
                continue
            moved = ""
            if r.get("pose") and base and base.get("pose") and v != "base":
                sep, droll = disagreement(cams[seq.split("_")[-1]], base["pose"], r["pose"])
                moved = f"{sep:.3f}"
            px = f"{r['median_px']:.2f}" if r.get("median_px") is not None else ""
            print(f"{seq[7:] if v == 'base' else '':36s} {v:6s} {r.get('n_tracks', 0):6d} "
                  f"{r['status']:8s} {r.get('n_stars') or 0:5d} {px:>5s} {len(r.get('bright', [])):6d} "
                  f"{moved:>7s}  {' '.join(r.get('bright', []))}")
            if r["status"] != "solved" and r.get("reason"):
                print(f"{'':44s}{r['reason'][:90]}")


def compare(variant: str = "big") -> None:
    base_of = bases()
    runs = {r["seq"]: r for r in load() if r["variant"] == variant and r["seq"] in base_of}
    cams = cameras()
    rescued = [s for s, r in runs.items() if r["status"] == "solved" and base_of[s]["status"] != "solved"]
    lost = [s for s, r in runs.items() if r["status"] != "solved" and base_of[s]["status"] == "solved"]
    both = [s for s, r in runs.items() if r["status"] == "solved" and base_of[s]["status"] == "solved"]
    print(f"{variant}: {len(runs)} blocks, {sum(base_of[s]['status'] == 'solved' for s in runs)} solved "
          f"at base, {sum(r['status'] == 'solved' for r in runs.values())} with {variant}")

    # Cross-night agreement over the whole cache, with this variant's solves standing in for base
    # wherever it ran; FIgLib rows count as siblings too.
    meta = {r["seq"]: r for r in summary()}
    def as_result(seq, r):
        return {**r, "seq": seq, "camera": meta[seq].get("camera") or seq.split("_")[-1]}
    seqs = nights.sequences()   # CDN rows carry no t0; FIgLib rows do
    t0 = {s: r.get("t0") or seqs.get(s, {}).get("t0") for s, r in meta.items()}
    swapped = [as_result(s, runs.get(s, r)) for s, r in meta.items()]
    near = {p["seq"]: p for p in pairs(swapped, t0, cams)}
    near_base = {p["seq"]: p for p in pairs(list(meta.values()), t0, cams)}

    def line(s, r, extra=""):
        p = near.get(s)
        agree = (f"nearest night {p['other'][7:15]} ({p['days_apart']:.0f} d): sep {p['sep_deg']:.3f} "
                 f"roll {p['roll_deg']:+.3f}") if p else "no other night solved"
        print(f"  {s[7:]:34s} {r.get('n_stars') or 0:3d} stars {r.get('median_px') or 0:5.2f} px  {extra}{agree}")

    print(f"\nrescued ({len(rescued)}):")
    for s in sorted(rescued):
        line(s, runs[s])
        print(f"  {'':34s} base: {(base_of[s].get('reason') or '')[:70]}")
    print(f"\nlost ({len(lost)}):")
    for s in sorted(lost):
        print(f"  {s[7:]:34s} base {base_of[s]['n_stars']} stars -> {(runs[s].get('reason') or '')[:70]}")

    moved = sorted((disagreement(cams[base_of[s].get("camera") or s.split("_")[-1]],
                                 base_of[s]["pose"], runs[s]["pose"])[0], s) for s in both)
    if moved:
        q = lambda f: moved[min(len(moved) - 1, int(f * len(moved)))][0]
        print(f"\nsolved both ways ({len(moved)}): pose moved median {q(0.5):.3f} deg, p90 {q(0.9):.3f}, "
              f"max {moved[-1][0]:.3f}; stars {sum(base_of[s]['n_stars'] for s in both)} -> "
              f"{sum(runs[s]['n_stars'] for s in both)}, bright matched "
              f"{sum(len(runs[s].get('bright', [])) for s in both)}")
        for sep, s in moved[::-1][:8]:
            if sep >= 0.05:
                line(s, runs[s], f"moved {sep:.3f}  ")

    def med(d, keys):
        v = sorted(d[k]["sep_deg"] for k in keys if k in d)
        return f"{v[len(v) // 2]:.3f} deg over {len(v)}" if v else "-"
    print(f"\ncross-night sep, blocks solved both ways: base {med(near_base, both)}, {variant} {med(near, both)}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "run":
        variants = [a.split("=", 1)[1] for a in args if a.startswith("--variant=")]
        blocks = [a for a in args if not a.startswith("--")]
        run(all_blocks() if "--all" in args else blocks or BLOCKS, variants or VARIANTS)
    elif cmd == "report":
        report()
    elif cmd == "compare":
        compare(*args)
    else:
        print(__doc__)
