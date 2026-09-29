"""Camera pose (and lens) from one night's moving star tracks.

star_track_calibrate.py solved 9 of 34 night sequences by searching pose AND lens scale
together. All 9 agreed on the lens (k 0.882-0.890x nameplate, k1 -0.074..-0.084;
plume-triangulation's NOTES.md, 2026-09-13), so this treats that lens as known and searches
only d_az/d_pitch/d_roll. A
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

Input is a `Night`: one camera's tracks (tracks.collect) with the frame size, the camera's
published record (lat, lon, elev, az, fov, pitch, roll, imager) and the epoch the track
offsets count from. `solve` and `solve_wide` return a JSON-ready dict; nothing here reads or
writes files. `hpwren.calibrate` is the command-line driver for HPWREN's CDN nights.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree

from . import catalog as SG
from . import solver_id
from . import pole as POLE
from .fisheye import K1, K_RATIO, initial_k, project_fisheye

MAX_TRACKS = 150


@dataclass
class Night:
    """One camera's sky over one block of frames: what a solve needs, and nothing else.

    `tracks` are tracks.collect's output, offsets in seconds from `t0` (an epoch). `cam` is
    the camera's published record; `camera` its name and `label` the block's, both only
    carried into the result."""
    camera: str
    cam: dict
    t0: float
    tracks: list[dict]
    W: int = 3072
    H: int = 2048
    label: str | None = None


def clip(track: dict, window: tuple[float, float] | None,
         min_frames: int = 8, min_span_px: float = 50.0) -> dict:
    """The part of a track inside [window[0], window[1]) seconds of offset, or {} if what is
    left would fail tracks.moving_tracks' test (>= min_frames points, >= min_span_px from
    first to last). No window: the track unchanged."""
    if window is None:
        return track
    t = {o: v for o, v in track.items() if window[0] <= o < window[1]}
    if len(t) < min_frames:
        return {}
    o = sorted(t)
    return t if np.hypot(t[o[-1]][0] - t[o[0]][0], t[o[-1]][1] - t[o[0]][1]) >= min_span_px else {}


def windowed(tracks: list[dict], window: tuple[float, float] | None = None):
    """(raw, keep): the tracks clipped to `window` -- raw keeps its length and indexing, so
    `matches` still name the caller's track indices -- and the indices `prune` keeps."""
    raw = [clip(t, window) for t in tracks]
    ok = [i for i, t in enumerate(raw) if t]
    return raw, [ok[i] for i in prune([raw[i] for i in ok])]


def prune(tracks: list[dict]) -> list[int]:
    """Indices of the tracks to use: all of them, unless there are too many to be stars."""
    idx = list(range(len(tracks)))
    if len(tracks) <= MAX_TRACKS:
        return idx
    # "Half as long as the longest" separates stars from short noise tracks over one block,
    # but over a whole night the longest track spans ~500 frames and the rule would keep only
    # the few stars up all night (hp-s-mobo-c: 377 tracks -> 12, too few to solve). Capping
    # the bar at 45 frames leaves every <= 90-frame block exactly as before.
    longest = max(len(t) for t in tracks)
    idx = [i for i in idx if len(tracks[i]) >= min(0.5 * longest, 45)]
    idx.sort(key=lambda i: -np.median([v[2] for v in tracks[i].values()]))
    return idx[:MAX_TRACKS]


def make_coincidence(c, W, H, lens, tree, y_band, az_b, alt_b, coarse_px):
    """How many bright stars a candidate pose puts on top of some track.

    Factored out so a figure (plume-triangulation's `stars.fig_solve_process`) can draw the
    same score the search optimises, rather than a reimplementation that could quietly drift.
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


def scan_context(night: Night, k_ratio: float | None = None, wide: bool = True,
                 window: tuple[float, float] | None = None):
    """Everything `solve`'s coarse stage builds, for a figure to draw or a caller to inspect.

    Returns the tracks actually used, the frame size, the lens, the pole fit, the scorer, and
    the psi scan itself (angles, scores, and the pose each angle implies). Same code path as
    the solver: the scorer comes from `make_coincidence` and the family from `pole`.
    """
    c, W, H = night.cam, night.W, night.H
    raw, keep = windowed(night.tracks, window)
    tracks = [raw[i] for i in keep]
    k0 = initial_k(c, W)
    lens = ((k_ratio or K_RATIO) * k0, K1)
    offs = sorted({o for t in tracks for o in t})
    ref = max(offs, key=lambda o: sum(o in t for t in tracks))
    vis = SG.visible_stars(c, night.t0 + ref, mag_limit=4.0,
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
            "t0": night.t0}


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


def _level_starts(psi, poses, scores, starts, coincidence, window_deg: float = 8.0):
    """Extra starts from the level prior: cameras are mounted close to level, so of the pole's
    one-angle family of poses the right one is almost always near where d_pitch and d_roll
    are both small. The family has two such places (the other is always >= 24 deg away);
    near each, the best-scoring angle within +-window_deg becomes a start, unless the
    coincidence scan already has one within 2 deg.

    Over the 129 CDN blocks solved on 2026-09-29 the level minimum sat 1.4 deg (median) and
    6.7 deg (worst) from the solved pose. It is a start, not a constraint: solved pitch
    offsets reach 16.8 deg (mpo-n) and roll offsets several degrees."""
    lvl = np.hypot(poses[:, 1], poses[:, 2])
    n, step = len(psi), psi[1] - psi[0]
    w = int(round(window_deg / step))
    idx = np.arange(n)
    minima = [i for i in idx if lvl[i] == lvl[(i + idx[:2 * w + 1] - w) % n].min()]
    out = []
    for m in sorted(minima, key=lambda i: lvl[i])[:2]:
        win = (m + np.arange(-w, w + 1)) % n
        i = int(win[np.argmax(scores[win])]) if scores[win].max() > 0 else m
        pose = tuple(float(v) for v in poses[i])
        if all(np.max(np.abs(np.array(pose) - np.array(g[3], float))) >= 2.0 for g in starts + out):
            sc, n_in, n_pred = coincidence(pose)
            out.append((sc, n_in, n_pred, pose))
    return out


def solve(night: Night, wide: bool = False, min_stars: int = 8,
          k_ratio: float | None = None, window: tuple[float, float] | None = None,
          trace: list | None = None, level_starts: bool = True) -> dict:
    """One night's pose.

    `wide` replaces the +-30/15/10 deg grid around the published pose with `pole.py`'s
    closed-form pole plus a 1-D scan of the one angle the pole cannot see, which makes the
    search global instead of local -- the only way to reach a camera whose published azimuth
    is wrong by more than the grid is wide. `min_stars` is the acceptance cutoff; lowering it
    is only safe if something else vouches for the solve (see `cross_night.py`).

    `trace`, if given, gets one dict per step appended to it -- the attempt, the pole, the
    coarse scan, the starts, every rung of every start's refinement, and the verdict -- for
    `animate.py` to draw. Nothing reads it back, so passing one never changes the result.
    """
    def note(**event):
        if trace is not None:
            trace.append(event)

    c, W, H = night.cam, night.W, night.H
    # every solve says which catalog and corrections it was made under; the ledger checks it
    out = {"seq": night.label, "camera": night.camera, "imager": c.get("imager"),
           "sky_model": SG.model_id(), "solver": solver_id()}
    if window is not None:
        out["window"] = window
    raw, keep = windowed(night.tracks, window)
    tracks = [raw[i] for i in keep]
    out.update(W=W, H=H, n_tracks_raw=len(raw), n_tracks=len(tracks))
    note(stage="attempt", wide=wide, k_ratio=k_ratio, tracks=list(keep))
    if len(tracks) < 8:
        out.update(status="failed", reason=f"only {len(tracks)} moving tracks")
        note(stage="verdict", status="failed", reason=out["reason"])
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
    vis = SG.visible_stars(c, night.t0 + ref, mag_limit=4.0,
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
    note(stage="setup", lens=lens, ref=ref, coarse_px=coarse_px, ladder=ladder)

    grid = []
    if wide:
        fit = POLE.estimate(tracks, W, H, *lens)
        out["pole"] = None if fit is None else {
            "norm": fit["norm"], "inlier_frac": fit["inlier_frac"],
            "median_res_rel": fit["median_res_rel"], "p_cam": list(fit["p_hat"])}
        note(stage="pole", fit=out["pole"], inliers=None if fit is None else fit["inliers"])
        if fit is None or fit["norm"] < 0.75:
            out.update(status="failed",
                       reason=("no pole fit" if fit is None else
                               f"no coherent sky rotation (|p| {fit['norm']:.2f}, "
                               f"residual {fit['median_res_rel']:.2f} of sidereal)"))
            note(stage="verdict", status="failed", reason=out["reason"])
            return out
        # Everything except the turn about the polar axis is now fixed, so scan that one
        # angle. 0.1 deg is finer than the 1 deg the old grid could afford in three.
        psi = np.arange(0.0, 360.0, 0.1)
        poses = POLE.pose_from_axes(POLE.cam_from_pole(fit["p_hat"], c["lat"], psi), c)
        scores = np.zeros(len(psi))
        for i, pose in enumerate(poses):
            sc, n_in, n_pred = coincidence(tuple(pose))
            scores[i] = sc
            if n_in >= 3:
                grid.append((sc, n_in, n_pred, tuple(float(v) for v in pose)))
        note(stage="scan", psi=psi, poses=poses, scores=scores)
        grid.sort(key=lambda g: -g[0])
        grid = _spread(grid, min_sep_deg=2.0, n=5)
        if level_starts:
            grid += _level_starts(psi, poses, scores, grid, coincidence)
    else:
        for p in itertools.product(range(-30, 31), range(-15, 16), range(-10, 11, 2)):
            sc, n_in, n_pred = coincidence(p)
            if n_in >= 3:
                grid.append((sc, n_in, n_pred, p))
        grid.sort(key=lambda g: -g[0])
        note(stage="scan", poses=np.array([g[3] for g in grid[:200]], float).reshape(-1, 3),
             scores=np.array([g[0] for g in grid[:200]], float))
    pub = coincidence((0, 0, 0))
    out["published_coincidence"] = {"inliers": pub[1], "predicted": pub[2]}
    starts = grid[:5] + [g for g in grid[5:] if wide][:2]   # the level starts come after
    out["coarse_top"] = [{"score": round(g[0], 2), "inliers": g[1], "predicted": g[2],
                          "pose": list(g[3])} for g in starts]
    note(stage="starts", poses=[list(g[3]) for g in starts], scores=[g[0] for g in starts])

    # every catalog star's (alt, az) at every observed offset
    ep = night.t0 + np.array(offs, float)
    alt_tab = np.zeros((len(vis), len(offs)))
    az_tab = np.zeros_like(alt_tab)
    for i, n in enumerate(names):
        alt_tab[i], az_tab[i] = SG.star_altaz(n, ep, c["lat"], c["lon"], c.get("elev") or 1600.0)
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
    for s, g in enumerate(starts):
        p5 = np.array([*g[3], *lens], float)
        pairs = []
        for thr, free in ladder:
            C = costs(p5)
            ri, ci = linear_sum_assignment(np.minimum(np.nan_to_num(C, nan=1e3), 1e3))
            pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < thr]
            named = [(names[i], keep[j]) for i, j in pairs]   # raw track indices, as `matches`
            if len(pairs) < 4:
                note(stage="rung", start=s, thr=thr, free_lens=free, before=p5.copy(),
                     after=None, pairs=named, median_px=None)
                break
            before = p5.copy()
            p5, e, ia = refit(p5, pairs, free)
            note(stage="rung", start=s, thr=thr, free_lens=free, before=before, after=p5.copy(),
                 pairs=named, median_px=float(np.median(e)))
        if len(pairs) < 4:
            continue
        runs.append({"s": s, "start": list(g[3]), "pose": p5, "pairs": pairs, "e": e, "ia": ia})

    if not runs:
        out.update(status="failed", reason="no start converged to >= 4 stars")
        note(stage="verdict", status="failed", reason=out["reason"])
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
    note(stage="verdict", status=out["status"], reason=out["reason"],
         start=best["s"],
         tests=[(f"{len(stars)} stars >= {min_stars}", len(stars) >= min_stars),
                (f"median {med:.2f}px < 3.0", med < 3.0),
                (f"k {kr:.3f}x in {lo_k:.2f}-{hi_k:.2f}", lo_k <= kr <= hi_k)])
    return out


def solve_wide(night: Night, min_stars: int = 8, window: tuple[float, float] | None = None,
               trace: list | None = None, level_starts: bool = True,
               grid: str = "fallback", sure_stars: int = 12) -> dict:
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

    `trace` collects every attempt's steps (see `solve`), then a "kept" event naming the
    attempt whose result is returned.
    """
    raw, keep = windowed(night.tracks, window)
    tracks = [raw[i] for i in keep]
    cands = []
    if len(tracks) >= 8:
        ls = POLE.lens_scale(tracks, night.W, night.H, initial_k(night.cam, night.W), K1)
        if ls is not None and 0.6 <= ls["k_ratio"] <= 1.2 and abs(ls["k_ratio"] - K_RATIO) > 0.005:
            cands.append(ls["k_ratio"])
    cands = [None] + cands
    # The published-pose grid is the third attempt, and with grid="fallback" it runs only
    # when the pole attempts haven't already solved with sure_stars stars. It once rescued
    # five blocks whose pole fits came out at |p| 0.47-0.69; those were banner tracks, which
    # tracks.clean now drops, and every solved block's pole fit is within 1% of |p| 1. It is
    # also ~3/4 of the time: over 180 CDN blocks (2026-09-29) skipping it on sure solves cut
    # the solve time five-fold and moved no pose. On a thin solve it still runs, because it
    # can end in a tighter fit of the same stars (mp-n 09-11: 0.90 px against the pole's
    # 1.32). grid="always" runs it every time, as before.
    attempts = [(True, kr) for kr in cands]
    attempts = attempts + [(False, None)] if grid == "fallback" else [(False, None)] + attempts
    best = None
    for a, (wide, kr) in enumerate(attempts):
        if (not wide and grid == "fallback" and best is not None
                and best[1]["status"] == "solved" and best[1]["n_stars"] >= sure_stars):
            continue
        r = solve(night, wide=wide, min_stars=min_stars, k_ratio=kr, window=window, trace=trace,
                  level_starts=level_starts)
        r["lens_from_pole"] = cands[1] if len(cands) > 1 else None
        r["found_by"] = "pole" if wide else "grid"
        key = (r["status"] == "solved", r.get("n_stars") or 0, -(r.get("median_px") or 1e9))
        if best is None or key > best[0]:
            best = (key, r, a)
    if trace is not None:
        trace.append({"stage": "kept", "attempt": best[2]})
    return best[1]
