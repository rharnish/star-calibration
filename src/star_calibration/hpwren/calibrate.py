"""Calibrate HPWREN cameras from their CDN night frames: fetch, track, solve, ledger.

Each step caches under hpwren.cache_dir() ($HPWREN_CACHE, default ~/.cache/hpwren), so a
re-run only does what it hasn't done:

  nights/      frames, one directory per camera-night block (nights.py)
  tracks/      tracks_<block>.pkl, the moving tracks and frame size per block
  solves/      solve_<block>.json per block, and summary.json over all of them

    python -m star_calibration.hpwren.calibrate fetch 20260911 hp-s-mobo-c vo-n-mobo-c
    python -m star_calibration.hpwren.calibrate solve                  # every indexed block
    python -m star_calibration.hpwren.calibrate solve hpwren_20260911_Q1_hp-s-mobo-c
    python -m star_calibration.hpwren.calibrate agree                  # night-to-night check
    python -m star_calibration.hpwren.calibrate ledger [dest.json]     # solved -> pose ledger

`solve` runs `solve.solve_wide`: the published-pose grid, then the pole-based global search
under the shared lens and under the lens the trails measure, best result kept. Whole-night
blocks (<day>_N) are for window studies, not the ledger, and are left out of `solve` unless
named.
"""
from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

from .. import cross_night, ledger
from ..solve import Night, solve_wide
from ..tracks import collect
from . import cache_dir, cameras, nights


def tracks_dir() -> Path:
    return cache_dir() / "tracks"


def solves_dir() -> Path:
    return cache_dir() / "solves"


def load_tracks(seq: str) -> tuple[list[dict], tuple[int, int]]:
    """One block's moving tracks and frame size, extracted once and cached."""
    cache = tracks_dir() / f"tracks_{seq}.pkl"
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        return d["tracks"], tuple(d.get("WH", (3072, 2048)))
    s = nights.sequences()[seq]
    mono = cameras()[s["camera"]].get("imager") == "monochrome"
    # A whole night is ~470 frames: keep only one decoded, and close tracks lost for 5 min so
    # a star behind cloud can't be relinked hours later.
    whole = "_N_" in seq
    tracks, decoded = collect(nights.read_frames(seq), s["lat"], s["lon"], mono=mono,
                              keep_frames=not whole, max_gap_s=300.0 if whole else None,
                              label=seq)
    W, H = 3072, 2048
    if decoded:
        H, W = next(iter(decoded.values()))[1].shape[:2]
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"tracks": tracks, "WH": (W, H)}, open(cache, "wb"))
    return tracks, (W, H)


def night(seq: str) -> Night:
    """An indexed block as the solver's input."""
    s = nights.sequences()[seq]
    tracks, (W, H) = load_tracks(seq)
    return Night(camera=s["camera"], cam=cameras()[s["camera"]], t0=s["t0"], tracks=tracks,
                 W=W, H=H, label=seq)


def _one(seq: str) -> dict:
    try:
        r = solve_wide(night(seq))
    except Exception as exc:   # one bad block shouldn't sink the batch
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    solves_dir().mkdir(parents=True, exist_ok=True)
    (solves_dir() / f"solve_{seq}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


def summary() -> list[dict]:
    p = solves_dir() / "summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def solve_blocks(seqs: list[str], workers: int = 4) -> list[dict]:
    """Solve `seqs` and merge them into solves/summary.json."""
    with Pool(workers) as pool:
        new = {r["seq"]: r for r in pool.imap_unordered(_one, seqs)}
    merged = {r["seq"]: r for r in summary()} | new
    rows = sorted(merged.values(), key=lambda r: (r["status"] != "solved", r["seq"]))
    solves_dir().mkdir(parents=True, exist_ok=True)
    (solves_dir() / "summary.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    return sorted(new.values(), key=lambda r: (r["status"] != "solved", r["seq"]))


def report(results: list[dict]) -> None:
    for r in results:
        if r["status"] == "solved":
            p = r["pose"]
            print(f"SOLVED {r['seq']:42s} {r['n_stars']:2d} stars med {r['median_px']:.2f}px  "
                  f"d_az {p['d_az']:+6.2f} d_pitch {p['d_pitch']:+6.2f} d_roll {p['d_roll']:+6.2f} "
                  f"k {p['k_ratio']:.3f} k1 {p['k1']:+.3f} ({r.get('found_by')})  {r['W']}x{r['H']}")
        else:
            print(f"failed {r['seq']:42s} {r.get('n_tracks', '-')}/{r.get('n_tracks_raw', '-')} "
                  f"tracks  {r['reason']}")
    print(f"{sum(r['status'] == 'solved' for r in results)}/{len(results)} blocks solved")


def main(argv: list[str]) -> None:
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "fetch":
        nights.main(args)
    elif cmd == "solve":
        seqs = args or sorted(s for s in nights.index() if "_N_" not in s)
        print(f"{len(seqs)} blocks", flush=True)
        report(solve_blocks(seqs))
    elif cmd == "agree":
        t0 = {k: v["t0"] for k, v in nights.sequences().items()}
        cross_night.report(summary(), t0, cameras())
    elif cmd == "ledger":
        t0 = {k: v["t0"] for k, v in nights.sequences().items()}
        dest = Path(args[0]) if args else cache_dir() / "pose_ledger.json"
        rows = ledger.build(summary(), t0, dest)
        print(f"wrote {dest} ({len(rows)} solves, {len({r['camera'] for r in rows})} cameras)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
