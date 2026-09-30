"""Run `ablation.py` over the cache's solved blocks, and sum it up.

    python -m star_calibration.hpwren.ablate refit [block ...]    # ablation/refit_<block>.json
    python -m star_calibration.hpwren.ablate resolve [block ...]  # ablation/resolve_<block>.json
    python -m star_calibration.hpwren.ablate report               # ablation/summary.json

With no blocks named, every solved CDN block in solves/summary.json (whole-night blocks and
rows filed by `calibrate add` left out: their tracks aren't here). Nothing outside
ablation/ is written: solves, gallery and ledger stay as `calibrate` left them.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import ablation as A
from . import cache_dir, nights
from .calibrate import night, solves_dir, summary

SAME_DEG = 0.2      # a pose is "the same" when every angle is within this
SAME_K = 0.005      # ... and, for a refit, the lens ratio within this


def ablation_dir() -> Path:
    return cache_dir() / "ablation"


def solved_blocks() -> list[str]:
    indexed = nights.sequences()
    return [r["seq"] for r in summary() if r["status"] == "solved" and "source" not in r
            and r["seq"] in indexed and "_N_" not in r["seq"]]


def _result(seq: str) -> dict:
    return json.loads((solves_dir() / f"solve_{seq}.json").read_text())


def _write(kind: str, seq: str, d: dict) -> None:
    ablation_dir().mkdir(parents=True, exist_ok=True)
    (ablation_dir() / f"{kind}_{seq}.json").write_text(json.dumps(d, default=float) + "\n")


def _refit(seq: str) -> str:
    r, n = _result(seq), night(seq)
    _write("refit", seq, {"held": A.refit_experiments(r, n, centre=False),
                          "free": A.refit_experiments(r, n, centre=True)})
    return seq


def _resolve(seq: str) -> str:
    _write("resolve", seq, A.resolve_variants(_result(seq), night(seq), same_deg=SAME_DEG))
    return seq


def run(fn, seqs: list[str], workers: int = 4) -> None:
    with Pool(workers) as pool:
        for i, seq in enumerate(pool.imap_unordered(fn, seqs), 1):
            print(f"{i}/{len(seqs)} {seq}", flush=True)


def _load(kind: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(ablation_dir().glob(f"{kind}_*.json"))]


def _q(v) -> list[float]:
    """Median and the middle half."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [50, 25, 75])] if len(v) else [np.nan] * 3


def refit_summary(blocks: list[dict]) -> dict:
    """One centre mode's refits over every block."""
    params = A.PARAMS if blocks and blocks[0]["centre_free"] else A.PARAMS[:5]
    out: dict = {"n_blocks": len(blocks), "params": list(params)}

    def d(row, base):
        return {k: row[k] - base[k] for k in params}

    # the baseline against the stored solve: a check that the refit is the solve's own fit
    off = [{k: b["rows"][0][k] - b["solved"][k] for k in params} for b in blocks]
    out["baseline_vs_solved"] = {k: _q([abs(o[k]) for o in off]) for k in params}
    out["baseline_median_px"] = _q([b["rows"][0]["median_px"] for b in blocks])

    # leave one out, as jackknife influence (n-1) * delta, by the dropped star's theta
    loo = []
    for b in blocks:
        base, th = b["rows"][0], {s["name"]: s for s in b["stars"]}
        n = len(b["stars"])
        for row in b["rows"]:
            if row["kind"] == "leave_one_out":
                s = th[row["drop"]]
                loo.append({"seq": b["seq"], "theta_deg": s["theta_deg"], "r_px": s["r_px"],
                            "n_points": s["n_points"], "n_stars": n,
                            **{k: (n - 1) * v for k, v in d(row, base).items()}})
    out["loo"] = loo
    edges = np.arange(0, 100, 10)
    out["loo_by_theta"] = []
    for lo in edges:
        sel = [x for x in loo if lo <= x["theta_deg"] < lo + 10]
        if sel:
            out["loo_by_theta"].append({"theta_lo": float(lo), "n": len(sel),
                                        **{k: _q([abs(x[k]) for x in sel]) for k in params}})
    th = np.array([x["theta_deg"] for x in loo])
    out["loo_rank_corr"] = {k: float(np.corrcoef(np.argsort(np.argsort(th)),
                                                 np.argsort(np.argsort(np.abs([x[k] for x in loo]))))[0, 1])
                            for k in params}
    out["theta_range"] = {"stars": _q(th), "min": float(th.min()), "max": float(th.max()),
                          "per_block_span": _q([max(s["theta_deg"] for s in b["stars"]) -
                                                min(s["theta_deg"] for s in b["stars"]) for b in blocks])}

    # inner-only against outer-only, same number of stars each
    out["radial"] = []
    for q in A.RADIAL_FRACTIONS:
        for side in ("inner", "outer"):
            rows = [(r, b["rows"][0]) for b in blocks for r in b["rows"]
                    if r["kind"] == side and abs(r["fraction"] - q) < 1e-9]
            if rows:
                out["radial"].append({"side": side, "fraction": q, "n": len(rows),
                                      "median_px": _q([r["median_px"] for r, _ in rows]),
                                      **{k: _q([r[k] - b[k] for r, b in rows]) for k in params},
                                      "abs": {k: _q([abs(r[k] - b[k]) for r, b in rows]) for k in params}})

    # random subsets: the spread across draws, and how often a draw lands on the same fit
    out["subsets"] = []
    for size in A.SUBSET_SIZES:
        sd, same, n_blocks = {k: [] for k in params}, [], 0
        for b in blocks:
            rows = [r for r in b["rows"] if r["kind"] == "subset" and r["size"] == size]
            if not rows:
                continue
            n_blocks += 1
            base = b["rows"][0]
            for k in params:
                sd[k].append(float(np.std([r[k] for r in rows])))
            same += [all(abs(r[k] - base[k]) <= SAME_DEG for k in A.POSE)
                     and abs(r["k_ratio"] - base["k_ratio"]) <= SAME_K for r in rows]
        out["subsets"].append({"size": size, "n_blocks": n_blocks,
                               "same_fit": float(np.mean(same)) if same else np.nan,
                               "sd": {k: _q(v) for k, v in sd.items()}})

    out["shorten"] = []
    for frac in A.SHORTEN:
        rows = [(r, b["rows"][0]) for b in blocks for r in b["rows"]
                if r["kind"] == "shorten" and r["fraction"] == frac]
        out["shorten"].append({"fraction": frac, "abs": {k: _q([abs(r[k] - b[k]) for r, b in rows])
                                                         for k in params}})
    return out


def centre_summary(pairs: list[tuple[dict, dict]]) -> dict:
    """Centre free against centre held, block by block: how far the centre wants to move
    from the camera's intrinsics value, and what that does to the pose and residual."""
    rows = []
    for h, f in pairs:
        a, b = h["rows"][0], f["rows"][0]
        rows.append({"seq": h["seq"], "n_stars": len(h["stars"]),
                     "dcx": b["cx"] - a["cx"], "dcy": b["cy"] - a["cy"],
                     **{f"d_{k}": b[k] - a[k] for k in (*A.POSE, "k_ratio", "k1")},
                     "median_px_held": a["median_px"], "median_px_free": b["median_px"]})
    shift = [np.hypot(r["dcx"], r["dcy"]) for r in rows]
    return {"rows": rows, "centre_shift_px": _q(shift),
            "abs": {k: _q([abs(r[k]) for r in rows]) for k in ("d_d_az", "d_d_pitch", "d_d_roll",
                                                               "d_k_ratio", "d_k1")},
            "median_px_held": _q([r["median_px_held"] for r in rows]),
            "median_px_free": _q([r["median_px_free"] for r in rows]),
            "shift_vs_n_stars_corr": float(np.corrcoef(shift, [r["n_stars"] for r in rows])[0, 1])}


def resolve_summary(blocks: list[dict]) -> dict:
    names = list(dict.fromkeys(r["variant"] for b in blocks for r in b["rows"]))
    per, wrong = [], []
    for v in names:
        rows = [r for b in blocks for r in b["rows"] if r["variant"] == v]
        solved = [r for r in rows if r["status"] == "solved"]
        per.append({"variant": v, "n": len(rows),
                    "solved": len(solved) / len(rows),
                    "solved_same": sum(r["same_pose"] for r in solved) / len(rows),
                    "solved_wrong": sum(not r["same_pose"] for r in solved) / len(rows),
                    "pass_default": sum(r["n_stars"] >= 8 for r in solved) / len(rows),
                    "pass_default_wrong": sum(r["n_stars"] >= 8 and not r["same_pose"]
                                              for r in solved) / len(rows)})
    # every solve at the loosened threshold, by star count: right or wrong
    by_n: dict[int, list[bool]] = {}
    for b in blocks:
        for r in b["rows"]:
            if r["status"] == "solved":
                by_n.setdefault(r["n_stars"], []).append(r["same_pose"])
                if not r["same_pose"]:
                    wrong.append({"seq": b["seq"], "variant": r["variant"], "n_stars": r["n_stars"],
                                  "median_px": r["median_px"], "delta": r["delta"]})
    return {"n_blocks": len(blocks), "variants": per, "wrong": wrong,
            "by_n_stars": [{"n_stars": n, "solves": len(v), "same": sum(v)}
                           for n, v in sorted(by_n.items())]}


def report() -> dict:
    refits = _load("refit")
    out = {}
    if refits:
        out["held"] = refit_summary([r["held"] for r in refits])
        out["free"] = refit_summary([r["free"] for r in refits])
        out["centre"] = centre_summary([(r["held"], r["free"]) for r in refits])
    resolves = _load("resolve")
    if resolves:
        out["resolve"] = resolve_summary(resolves)
    ablation_dir().mkdir(parents=True, exist_ok=True)
    (ablation_dir() / "summary.json").write_text(json.dumps(out, default=float) + "\n")
    _print(out)
    return out


def _print(out: dict) -> None:
    f = "{:>9.4f}".format
    for mode in ("held", "free"):
        if mode not in out:
            continue
        s = out[mode]
        ps = s["params"]
        print(f"\n== refit, centre {mode}: {s['n_blocks']} blocks")
        print("baseline vs stored |d| median:", {k: round(v[0], 5) for k, v in s["baseline_vs_solved"].items()})
        print(f"star theta: median {s['theta_range']['stars'][0]:.1f} deg, range "
              f"{s['theta_range']['min']:.1f}-{s['theta_range']['max']:.1f}; per-block span "
              f"{s['theta_range']['per_block_span'][0]:.1f} deg")
        print("leave-one-out |influence| median by theta:   " + " ".join(f"{k:>9s}" for k in ps))
        for b in s["loo_by_theta"]:
            print(f"  {b['theta_lo']:3.0f}-{b['theta_lo'] + 10:<3.0f} n={b['n']:4d}"
                  + " " * 21 + " ".join(f(b[k][0]) for k in ps))
        print("  rank corr(theta, |infl|)" + " " * 27 + " ".join(f(s["loo_rank_corr"][k]) for k in ps))
        print("radial (median delta from baseline):")
        for b in s["radial"]:
            print(f"  {b['side']:5s} {b['fraction']:.2f} n={b['n']:3d} med {b['median_px'][0]:.2f}px"
                  + " " * 14 + " ".join(f(b[k][0]) for k in ps))
        print("subsets (median sd across draws):")
        for b in s["subsets"]:
            print(f"  n={b['size']:2d} blocks={b['n_blocks']:3d} same {b['same_fit']:.2f}"
                  + " " * 19 + " ".join(f(b["sd"][k][0]) for k in ps))
        for b in s["shorten"]:
            print(f"  shorten {b['fraction']:.2f}" + " " * 32 + " ".join(f(b["abs"][k][0]) for k in ps))
    if "centre" in out:
        c = out["centre"]
        print(f"\n== centre free vs held: shift median {c['centre_shift_px'][0]:.1f}px "
              f"(IQR {c['centre_shift_px'][1]:.1f}-{c['centre_shift_px'][2]:.1f}); "
              f"median px {c['median_px_held'][0]:.2f} -> {c['median_px_free'][0]:.2f}")
        print("  |d| median:", {k: round(v[0], 4) for k, v in c["abs"].items()})
    if "resolve" in out:
        r = out["resolve"]
        print(f"\n== re-solve, min_stars 4: {r['n_blocks']} blocks")
        for v in r["variants"]:
            print(f"  {v['variant']:18s} solved {v['solved']:.2f}  same {v['solved_same']:.2f}  "
                  f"wrong {v['solved_wrong']:.2f}  | at min_stars 8: pass {v['pass_default']:.2f} "
                  f"wrong {v['pass_default_wrong']:.2f}")
        print("  by n_stars (solves, same):", [(b["n_stars"], b["solves"], b["same"]) for b in r["by_n_stars"]])


def main(argv: list[str]) -> None:
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    if cmd in ("refit", "resolve"):
        seqs = args or solved_blocks()
        print(f"{len(seqs)} blocks", flush=True)
        run(_refit if cmd == "refit" else _resolve, seqs)
    elif cmd == "report":
        report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
