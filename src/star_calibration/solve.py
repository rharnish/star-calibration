"""Star-track calibration with the lens fixed: pose only, from cached moving tracks.

star_track_calibrate.py solved 9 of 34 night sequences by searching pose AND lens scale
together. All 9 agreed on the lens (k 0.882-0.890x nameplate, k1 -0.074..-0.084; NOTES.md,
2026-09-13), so this treats that lens as known and searches only d_az/d_pitch/d_roll. A
3-parameter grid is finer (1 deg, not 2) and cheaper, and it can't wander into a wrong
focal scale that happens to line up a few stars. Only the last two refinement stages free
the lens again, as a check rather than an assumption.

Other changes from star_track_calibrate.py, each aimed at a failure the batch showed:
  * the coincidence score counts predictions inside the band the tracks actually occupy, not
    a fixed top 35% of the frame;
  * monochrome units produced 800-3700 "moving" tracks of mostly noise, so when there are
    more than MAX_TRACKS, keep long tracks and then the brightest by median amplitude;
  * sequences with no or few tracks report a reason instead of crashing;
  * a solve needs >= 8 stars, median residual < 3px and a lens ratio inside 0.85-0.92.

Writes data/star_tracks/solve_<seq>.json per sequence and solve_summary.json.
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

from . import catalog as SG
from . import nights as hpwren_nights
from . import pole as POLE
from .fisheye import initial_k, project_fisheye

ROOT = Path(__file__).resolve().parents[3]
SKY = ROOT / "out" / "sky"
CAMS = json.loads((ROOT / "data/meta/cams.json").read_text())
SEQS = {s["seq"]: s for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
SEQS.update(hpwren_nights.sequences())
DATA = SKY / "data/star_tracks"
K_RATIO, K1 = 0.886, -0.078
MAX_TRACKS = 150


def load_tracks(seq: str):
    cache = DATA / f"tracks_{seq}.pkl"
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        return d["tracks"], tuple(d.get("WH", (3072, 2048)))
    from .tracks import collect   # imported lazily: it decodes whole archives
    tracks, decoded, _ = collect(seq)
    W, H = 3072, 2048
    if decoded:
        H, W = next(iter(decoded.values()))[1].shape[:2]
    pickle.dump({"tracks": tracks, "WH": (W, H)}, open(cache, "wb"))
    return tracks, (W, H)


def prune(tracks: list[dict]) -> list[int]:
    """Indices of the tracks to use: all of them, unless there are too many to be stars."""
    idx = list(range(len(tracks)))
    if len(tracks) <= MAX_TRACKS:
        return idx
    longest = max(len(t) for t in tracks)
    idx = [i for i in idx if len(tracks[i]) >= 0.5 * longest]
    idx.sort(key=lambda i: -np.median([v[2] for v in tracks[i].values()]))
    return idx[:MAX_TRACKS]


def make_coincidence(c, W, H, lens, tree, y_band, az_b, alt_b, coarse_px):
    """How many bright stars a candidate pose puts on top of some track.

    Factored out so `stars.fig_solve_process` can draw the same score the search optimises,
    rather than a reimplementation of it that could quietly drift.
    """
    def coincidence(p):
        x, y = project_fisheye(c, az_b, alt_b, W, H, *p, *lens)
        x, y = x * W, y * H
        ins = (x >= 0) & (x < W) & (y >= 0) & (y < y_band)
        n_pred = int(ins.sum())
        if n_pred < 3:
            return 0.0, 0, n_pred
        d, _ = tree.query(np.c_[x[ins], y[ins]])
        n_in = int((d < coarse_px).sum())
        return n_in / n_pred * min(n_in, 10), n_in, n_pred
    return coincidence


def scan_context(seq: str, k_ratio: float | None = None, wide: bool = True):
    """Everything `solve`'s coarse stage builds, for a figure to draw or a caller to inspect.

    Returns the tracks actually used, the frame size, the lens, the pole fit, the scorer, and
    the psi scan itself (angles, scores, and the pose each angle implies). Same code path as
    the solver: the scorer comes from `make_coincidence` and the family from `pole`.
    """
    s, c = SEQS[seq], CAMS[SEQS[seq]["camera"]]
    raw, (W, H) = load_tracks(seq)
    tracks = [raw[i] for i in prune(raw)]
    k0 = initial_k(c, W)
    lens = ((k_ratio or K_RATIO) * k0, K1)
    offs = sorted({o for t in tracks for o in t})
    ref = max(offs, key=lambda o: sum(o in t for t in tracks))
    vis = SG.visible_stars(c, s["t0"] + ref, mag_limit=4.0,
                           fov_margin_deg=180 if wide else 45, min_alt_deg=-5)
    mags = np.array([v["mag"] for v in vis])
    near = [t for t in tracks if min(abs(o - ref) for o in t) <= 70]
    tree = cKDTree(np.array([t[min(t, key=lambda o: abs(o - ref))][:2] for t in near]))
    y_band = max(v[1] for t in tracks for v in t.values()) + 60
    b = mags <= 3.5
    alt_b = np.array([v["alt"] for v in vis])[b]
    az_b = np.array([v["az"] for v in vis])[b]
    coarse_px = 60.0 if (wide and k_ratio is not None) else 20.0
    coin = make_coincidence(c, W, H, lens, tree, y_band, az_b, alt_b, coarse_px)
    fit = POLE.estimate(tracks, W, H, *lens)
    psi = np.arange(0.0, 360.0, 0.1)
    poses = POLE.pose_from_axes(POLE.cam_from_pole(fit["p_hat"], c["lat"], psi), c) if fit else None
    scores = np.array([coin(tuple(q))[0] for q in poses]) if fit else None
    return {"cam": c, "tracks": tracks, "W": W, "H": H, "lens": lens, "ref": ref,
            "fit": fit, "coincidence": coin, "psi": psi, "poses": poses, "scores": scores,
            "t0": s["t0"]}


def _spread(grid, min_sep_deg: float = 2.0, n: int = 5):
    """Top-scoring poses that are not neighbours of each other.

    A 0.1 deg scan puts hundreds of near-identical poses around one peak, and handing five of
    them to the refinement loop would be five copies of the same start. Keep the best, then
    the best that is at least `min_sep_deg` away from everything kept.
    """
    out = []
    for g in grid:
        v = np.array(g[3], float)
        if all(np.max(np.abs(v - np.array(h[3], float))) >= min_sep_deg for h in out):
            out.append(g)
        if len(out) >= n:
            break
    return out


def solve(seq: str, wide: bool = False, min_stars: int = 8,
          k_ratio: float | None = None) -> dict:
    """One sequence's pose.

    `wide` replaces the +-30/15/10 deg grid around the published pose with `pole.py`'s
    closed-form pole plus a 1-D scan of the one angle the pole cannot see, which makes the
    search global instead of local -- the only way to reach a camera whose published azimuth
    is wrong by more than the grid is wide. `min_stars` is the acceptance cutoff; lowering it
    is only safe if something else vouches for the solve (see `cross_night.py`).
    """
    s = SEQS[seq]
    c = CAMS[s["camera"]]
    out = {"seq": seq, "camera": s["camera"], "imager": c.get("imager")}
    raw, (W, H) = load_tracks(seq)
    keep = prune(raw)
    tracks = [raw[i] for i in keep]
    out.update(W=W, H=H, n_tracks_raw=len(raw), n_tracks=len(tracks))
    if len(tracks) < 8:
        out.update(status="failed", reason=f"only {len(tracks)} moving tracks")
        return out

    k0 = initial_k(c, W)
    lens = ((k_ratio or K_RATIO) * k0, K1)
    offs = sorted({o for t in tracks for o in t})
    # Reference epoch: the frame the most tracks were actually seen in, and only tracks seen
    # within a frame of it enter the coincidence score. A fragment far from the reference
    # would contribute its position at the wrong time, displaced along its own motion --
    # tp-w-mobo-c's steep, fast arcs break into pieces and failed exactly that way.
    ref = max(offs, key=lambda o: sum(o in t for t in tracks))
    out["ref_offset"] = ref
    # A wide search may land the boresight anywhere, so the candidate pool has to be every
    # star above the horizon rather than a window around the published azimuth.
    vis = SG.visible_stars(c, s["t0"] + ref, mag_limit=4.0,
                           fov_margin_deg=180 if wide else 45, min_alt_deg=-5)
    names = [v["name"] for v in vis]
    mags = np.array([v["mag"] for v in vis])

    near = [t for t in tracks if min(abs(o - ref) for o in t) <= 70]
    ref_xy = np.array([t[min(t, key=lambda o: abs(o - ref))][:2] for t in near])
    tree = cKDTree(ref_xy)
    y_band = max(v[1] for t in tracks for v in t.values()) + 60
    bright = mags <= 3.5
    alt_b = np.array([v["alt"] for v in vis])[bright]
    az_b = np.array([v["az"] for v in vis])[bright]

    # How far a coarse candidate can sit from the truth, and so how big a catch radius the
    # scan needs. Under the shared lens a grid point is at most ~1 deg (34 px) out and a
    # pole-based start is tighter still, so 20 px is right and loosening it actively hurts:
    # on a north-facing camera the trails crowd around an in-frame pole, and a fat radius
    # lets a wrong turn about the pole collect enough near-misses to outscore the right one.
    # Under a lens *measured* from the trails the scale itself carries a few percent of
    # error -- 45 px at the edge of the frame for bm-s-mobo-c -- so that attempt needs both a
    # wider radius and a ladder that starts outside it. Keeping the two as separate attempts
    # is what lets each have the tolerance it needs.
    measured_lens = wide and k_ratio is not None
    coarse_px = 60.0 if measured_lens else 20.0
    ladder = (((100, False), (50, False), (25, False), (15, False), (8, False), (6, False),
               (6, True), (5, True)) if measured_lens else
              ((25, False), (15, False), (8, False), (6, False), (6, True), (5, True)))

    coincidence = make_coincidence(c, W, H, lens, tree, y_band, az_b, alt_b, coarse_px)

    grid = []
    if wide:
        fit = POLE.estimate(tracks, W, H, *lens)
        out["pole"] = None if fit is None else {
            "norm": fit["norm"], "inlier_frac": fit["inlier_frac"],
            "median_res_rel": fit["median_res_rel"], "p_cam": list(fit["p_hat"])}
        if fit is None or fit["norm"] < 0.75:
            out.update(status="failed",
                       reason=("no pole fit" if fit is None else
                               f"no coherent sky rotation (|p| {fit['norm']:.2f}, "
                               f"residual {fit['median_res_rel']:.2f} of sidereal)"))
            return out
        # Everything except the turn about the polar axis is now fixed, so scan that one
        # angle. 0.1 deg is finer than the 1 deg the old grid could afford in three.
        psi = np.arange(0.0, 360.0, 0.1)
        for pose in POLE.pose_from_axes(POLE.cam_from_pole(fit["p_hat"], c["lat"], psi), c):
            sc, n_in, n_pred = coincidence(tuple(pose))
            if n_in >= 3:
                grid.append((sc, n_in, n_pred, tuple(float(v) for v in pose)))
        grid.sort(key=lambda g: -g[0])
        grid = _spread(grid, min_sep_deg=2.0, n=5)
    else:
        for p in itertools.product(range(-30, 31), range(-15, 16), range(-10, 11, 2)):
            sc, n_in, n_pred = coincidence(p)
            if n_in >= 3:
                grid.append((sc, n_in, n_pred, p))
        grid.sort(key=lambda g: -g[0])
    pub = coincidence((0, 0, 0))
    out["published_coincidence"] = {"inliers": pub[1], "predicted": pub[2]}
    out["coarse_top"] = [{"score": round(g[0], 2), "inliers": g[1], "predicted": g[2],
                          "pose": list(g[3])} for g in grid[:5]]

    # every catalog star's (alt, az) at every observed offset
    ep = s["t0"] + np.array(offs, float)
    alt_tab = np.zeros((len(vis), len(offs)))
    az_tab = np.zeros_like(alt_tab)
    for i, n in enumerate(names):
        alt_tab[i], az_tab[i] = SG.altaz(*SG.STARS[n], ep, c["lat"], c["lon"])
    oidx = {o: j for j, o in enumerate(offs)}
    tr_idx = [np.array([oidx[o] for o in sorted(t)]) for t in tracks]
    tr_xy = [np.array([t[o][:2] for o in sorted(t)]) for t in tracks]

    def costs(p5):
        x, y = project_fisheye(c, az_tab.ravel(), alt_tab.ravel(), W, H, *p5)
        X, Y = (x * W).reshape(alt_tab.shape), (y * H).reshape(alt_tab.shape)
        C = np.empty((len(vis), len(tracks)))
        for j in range(len(tracks)):   # median, so a linker jump doesn't sink a true pair
            C[:, j] = np.median(np.hypot(X[:, tr_idx[j]] - tr_xy[j][:, 0],
                                         Y[:, tr_idx[j]] - tr_xy[j][:, 1]), axis=1)
        return C

    def refit(p5, pairs, free_lens):
        ia = np.concatenate([np.full(len(tr_idx[j]), i) for i, j in pairs])
        oa = np.concatenate([tr_idx[j] for _, j in pairs])
        obs = np.concatenate([tr_xy[j] for _, j in pairs])
        al, az = alt_tab[ia, oa], az_tab[ia, oa]

        def full(q):
            return q if free_lens else np.r_[q, p5[3:]]

        def r(q):
            x, y = project_fisheye(c, az, al, W, H, *full(q))
            return np.r_[x * W - obs[:, 0], y * H - obs[:, 1]]

        sol = least_squares(r, p5 if free_lens else p5[:3], loss="soft_l1", f_scale=4.0,
                            max_nfev=3000)
        return np.asarray(full(sol.x), float), np.hypot(*sol.fun.reshape(2, -1)), ia

    runs = []
    for g in grid[:5]:
        p5 = np.array([*g[3], *lens], float)
        pairs = []
        for thr, free in ladder:
            C = costs(p5)
            ri, ci = linear_sum_assignment(np.minimum(C, 1e3))
            pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < thr]
            if len(pairs) < 4:
                break
            p5, e, ia = refit(p5, pairs, free)
        if len(pairs) < 4:
            continue
        runs.append({"start": list(g[3]), "pose": p5, "pairs": pairs, "e": e, "ia": ia})

    if not runs:
        out.update(status="failed", reason="no start converged to >= 4 stars")
        return out
    best = max(runs, key=lambda r: (len(r["pairs"]), -float(np.median(r["e"]))))
    stars = {names[i] for i, _ in best["pairs"]}
    agree = sum(1 for r in runs if {names[i] for i, _ in r["pairs"]} == stars)
    p, e = best["pose"], best["e"]
    kr, med = p[3] / k0, float(np.median(e))
    lo_k, hi_k = (0.70, 0.95) if (wide or k_ratio) else (0.85, 0.92)
    ok = len(stars) >= min_stars and med < 3.0 and lo_k <= kr <= hi_k
    out["k_ratio_start"] = k_ratio or K_RATIO
    out.update(
        status="solved" if ok else "failed",
        reason=None if ok else f"best: {len(stars)} stars, median {med:.1f}px, k {kr:.3f}x",
        pose={"d_az": p[0], "d_pitch": p[1], "d_roll": p[2], "k_ratio": kr, "k1": p[4]},
        n_stars=len(stars), median_px=med, rmse_px=float(np.sqrt(np.mean(e ** 2))),
        runs_agreeing=f"{agree}/{len(runs)}",
        matches={names[i]: keep[j] for i, j in best["pairs"]},   # raw track indices
        per_star_px={names[i]: float(np.median(e[best["ia"] == i])) for i, _ in best["pairs"]},
        mags={names[i]: float(mags[i]) for i, _ in best["pairs"]})
    return out


def _run(seq: str) -> dict:
    try:
        r = solve(seq)
    except Exception as exc:   # one bad archive shouldn't sink the batch
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    (DATA / f"solve_{seq}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


if __name__ == "__main__":
    seqs = sys.argv[1:] or sorted(p.name[len("tracks_"):-4] for p in DATA.glob("tracks_*.pkl"))
    with Pool(4) as pool:
        results = list(pool.imap_unordered(_run, seqs))
    results.sort(key=lambda r: (r["status"] != "solved", r["seq"]))
    for r in results:
        if r["status"] == "solved":
            p = r["pose"]
            print(f"SOLVED {r['seq']:55s} {r['n_stars']:2d} stars  med {r['median_px']:.2f}px  "
                  f"d_az {p['d_az']:+6.2f} d_pitch {p['d_pitch']:+6.2f} d_roll {p['d_roll']:+6.2f}  "
                  f"k {p['k_ratio']:.3f}x k1 {p['k1']:+.3f}  runs {r['runs_agreeing']}")
        else:
            print(f"failed {r['seq']:55s} {r.get('n_tracks', '-')}/{r.get('n_tracks_raw', '-')} tracks  "
                  f"{r['reason']}")
    (DATA / "solve_summary.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"{sum(r['status'] == 'solved' for r in results)}/{len(results)} solved")


def solve_wide(seq: str, min_stars: int = 8) -> dict:
    """A global solve, tried under each lens the trails allow, best result kept.

    Two lens hypotheses go in: the shared 0.886x nameplate that `solve` assumes for every
    camera, and whatever `pole.lens_scale` measures from this night's own trails. Each gets
    its own complete search rather than being mixed into one candidate list, because the
    coincidence score is computed *through* the lens and so is not comparable across the two
    -- a wrong lens on a crowded field can outscore a right one. Comparing finished solves on
    star count and residual is comparable, and is what this does.

    Both hypotheses are needed. The pole's lens is what rescues the Big Black Mountain
    cameras, whose trails want ~0.78x and which match nothing at all under 0.886x. The shared
    lens is what rescues every north-facing camera, where the pole sits inside the frame,
    scale is ill-conditioned against pole position, and the measurement comes out 3-4% low.
    """
    raw, (W, H) = load_tracks(seq)
    tracks = [raw[i] for i in prune(raw)]
    cands = []
    if len(tracks) >= 8:
        c = CAMS[SEQS[seq]["camera"]]
        ls = POLE.lens_scale(tracks, W, H, initial_k(c, W), K1)
        if ls is not None and 0.6 <= ls["k_ratio"] <= 1.2 and abs(ls["k_ratio"] - K_RATIO) > 0.005:
            cands.append(ls["k_ratio"])
    cands = [None] + cands
    # The published-pose grid runs too, as a third attempt. The pole is an *extra* way in,
    # not a replacement, and it has one failure mode the grid does not: a night whose trails
    # carry no coherent rotation gets rejected outright. On five sequences that the grid
    # solves with 8+ stars the flow fit returns |p| of 0.47-0.69, and dropping them to gain
    # the Big Black Mountain cameras would be a bad trade. Running all three and keeping the
    # best makes this a strict superset of the old solver.
    attempts = [(False, None)] + [(True, kr) for kr in cands]
    best = None
    for wide, kr in attempts:
        r = solve(seq, wide=wide, min_stars=min_stars, k_ratio=kr)
        r["lens_from_pole"] = cands[1] if len(cands) > 1 else None
        r["found_by"] = "pole" if wide else "grid"
        key = (r["status"] == "solved", r.get("n_stars") or 0, -(r.get("median_px") or 1e9))
        if best is None or key > best[0]:
            best = (key, r)
    return best[1]
