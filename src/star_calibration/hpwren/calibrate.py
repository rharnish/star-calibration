"""Extract tracks and star-solve every indexed HPWREN night block, then rebuild the ledger.

Tracks are cached per block (data/star_tracks/tracks_hpwren_*.pkl), so a re-run only
decodes blocks it hasn't seen. Solves for FIgLib sequences are left as they are; the
summary is merged, so solve_summary.json keeps covering both sources.
"""
from __future__ import annotations

import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from . import nights as hpwren_nights
from . import solve as S

ROOT = Path(__file__).resolve().parents[3]


def _one(seq: str) -> dict:
    return S._run(seq)


if __name__ == "__main__":
    nights = hpwren_nights.index()
    # whole-night directories (<day>_N) belong to stars.window_ablation, not the Q1 batch
    seqs = sorted(s for s in nights if "_N_" not in s)
    print(f"{len(seqs)} HPWREN night blocks indexed", flush=True)
    with Pool(4) as pool:
        new = {r["seq"]: r for r in pool.imap_unordered(_one, seqs)}
    summary_path = S.DATA / "solve_summary.json"
    old = {r["seq"]: r for r in json.loads(summary_path.read_text())} if summary_path.exists() else {}
    old.update(new)
    merged = sorted(old.values(), key=lambda r: (r["status"] != "solved", r["seq"]))
    summary_path.write_text(json.dumps(merged, indent=1, default=float) + "\n")
    for r in sorted(new.values(), key=lambda r: (r["status"] != "solved", r["seq"])):
        if r["status"] == "solved":
            p = r["pose"]
            print(f"SOLVED {r['seq']:42s} {r['n_stars']:2d} stars med {r['median_px']:.2f}px  "
                  f"d_az {p['d_az']:+6.2f} d_pitch {p['d_pitch']:+6.2f} d_roll {p['d_roll']:+6.2f} "
                  f"k {p['k_ratio']:.3f} k1 {p['k1']:+.3f} runs {r['runs_agreeing']}  {r['W']}x{r['H']}")
        else:
            print(f"failed {r['seq']:42s} {r.get('n_tracks', '-')}/{r.get('n_tracks_raw', '-')} tracks  {r['reason']}")
    print(f"{sum(r['status'] == 'solved' for r in new.values())}/{len(new)} HPWREN blocks solved")
    subprocess.run([sys.executable, "-m", "src.figlib.pose_ledger"], cwd=ROOT, check=True)
