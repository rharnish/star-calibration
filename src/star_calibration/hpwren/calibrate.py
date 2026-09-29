"""Calibrate HPWREN cameras from their CDN night frames: fetch, track, solve, ledger.

Each step caches under hpwren.cache_dir() ($HPWREN_CACHE, default ~/.cache/hpwren), so a
re-run only does what it hasn't done:

  nights/      frames, one directory per camera-night block (nights.py)
  tracks/      tracks_<block>.pkl, the moving tracks and frame size per block
  solves/      solve_<block>.json per block, and summary.json over all of them
  overlays/    star_solve_<block>.jpg: the solve drawn on its own frame (overlay.py)
  animations/  star_solve_<block>.mp4: the solve played back, step by step (animate.py)
  explore/     <block>.js: the solve as data for gallery/explore.html, the browser replay

    python -m star_calibration.hpwren.calibrate fetch 20260911 hp-s-mobo-c vo-n-mobo-c
    python -m star_calibration.hpwren.calibrate solve                  # every indexed block
    python -m star_calibration.hpwren.calibrate solve hpwren_20260911_Q1_hp-s-mobo-c
    python -m star_calibration.hpwren.calibrate overlay [block ...]    # redraw overlays
    python -m star_calibration.hpwren.calibrate animate [--kept] [--full] [--mark A-B,C] block ...
    python -m star_calibration.hpwren.calibrate explore block ...       # replay in the browser
    python -m star_calibration.hpwren.calibrate agree                  # night-to-night check
    python -m star_calibration.hpwren.calibrate ledger [dest.json]     # solved -> pose ledger

`solve` runs `solve.solve_wide`: the pole-based global search under the shared lens and under
the lens the trails measure, then the published-pose grid unless a pole attempt already
solved with 12 or more stars, best result kept. Whole-night
blocks (<day>_N) are for window studies, not the ledger, and are left out of `solve` unless
named. Each solved (or failed) block's overlay is drawn as it finishes, so the gallery
(gallery.py) has a picture for every card. A block the camera sent nothing for -- the CDN's
"No Images!" placeholder in most of its frames (nights.PLACEHOLDER_WH) -- fails without a
solve, reason "no images: ...", and its card shows one of the placeholders.

A solve made from frames the cache doesn't hold (another project's archive) can be filed
beside these with `add`, so the one gallery shows it too; see `add` for what it must carry.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .. import animate as anim
from .. import cross_night, ledger, overlay
from .. import explore as ex
from ..solve import Night, solve_wide
from ..tracks import clean, collect
from . import cache_dir, cameras, nights


def tracks_dir() -> Path:
    return cache_dir() / "tracks"


def solves_dir() -> Path:
    return cache_dir() / "solves"


def overlays_dir() -> Path:
    return cache_dir() / "overlays"


def animations_dir() -> Path:
    return cache_dir() / "animations"


def explore_dir() -> Path:
    return cache_dir() / "explore"


def explorer_page() -> Path:
    """The replay page, gallery/explore.html, refreshed from the package's template."""
    dest = cache_dir() / "gallery" / "explore.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("explore.html"), dest)
    return dest


# HPWREN burns a line of text, with a clock, into the top ~35 rows of every frame; 60 leaves
# margin for the few pixels a detection's centroid can sit below the glyphs.
BANNER_PX = 60


def link_recipe(seq: str) -> dict:
    """What `collect` is called with for a block. A cached pickle made another way is stale."""
    mono = cameras()[nights.sequences()[seq]["camera"]].get("imager") == "monochrome"
    # A track unseen for 5 min closes, so a later star passing the spot can't inherit it
    # (`tracks.split` cuts what's left). 3 min lost bm-n 09-11 and smarpk-s 09-12: a slow,
    # faint star blinks out for longer than that and its pieces are too short to keep. The
    # predictive linker keeps its own 4 min: its 3 px gate already stops such hand-overs.
    if mono:
        return {"linker": "predictive", "max_gap_s": 240.0}
    return {"linker": "nearest", "max_gap_s": 300.0}


def clean_recipe() -> dict:
    return {"banner_px": BANNER_PX}


def load_tracks(seq: str) -> tuple[list[dict], tuple[int, int]]:
    """One block's tracks and frame size, extracted once and cached.

    The pickle keeps the linker's output (`linked`) and the cleaned tracks the solver uses
    (`tracks.clean`), each with the recipe that made it. A pickle whose linking recipe differs
    from `link_recipe` is rebuilt from the frames; one whose cleaning differs is re-cleaned
    from `linked`. A change to the code that doesn't change a recipe goes unnoticed: delete
    `tracks/` then."""
    cache = tracks_dir() / f"tracks_{seq}.pkl"
    link, cl = link_recipe(seq), clean_recipe()
    d = pickle.load(open(cache, "rb")) if cache.exists() else {}
    if d.get("link") != link:
        s = nights.sequences()[seq]
        whole = "_N_" in seq   # ~470 frames: keep only one decoded
        linked, decoded = collect(nights.read_frames(seq), s["lat"], s["lon"],
                                  mono=link["linker"] == "predictive", keep_frames=not whole,
                                  max_gap_s=link["max_gap_s"], label=seq)
        W, H = 3072, 2048
        if decoded:
            H, W = next(iter(decoded.values()))[1].shape[:2]
        d = {"linked": linked, "link": link, "WH": (W, H)}
    if d.get("clean") != cl:
        d["tracks"], d["cleaning"] = clean(d["linked"], **cl)
        d["clean"] = cl
        print(f"  {seq}: cleaned {d['cleaning']}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(d, open(cache, "wb"))
    return d["tracks"], tuple(d["WH"])


def night(seq: str) -> Night:
    """An indexed block as the solver's input."""
    s = nights.sequences()[seq]
    tracks, (W, H) = load_tracks(seq)
    return Night(camera=s["camera"], cam=cameras()[s["camera"]], t0=s["t0"], tracks=tracks,
                 W=W, H=H, label=seq)


def block(r: dict, indexed: dict | None = None) -> dict:
    """Where a result came from -- its camera, t0, lat and lon: its indexed block, or the
    result itself for one filed with `add`."""
    return (nights.sequences() if indexed is None else indexed).get(r["seq"]) or r


def render(seq: str, result: dict | None = None) -> Path:
    """Draw one block's solve on its reference frame into overlays/."""
    if result is None:
        result = json.loads((solves_dir() / f"solve_{seq}.json").read_text())
    n = night(seq)
    _, _, blob = nights.frame_at(seq, overlay.reference_offset(result, n))
    frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    overlays_dir().mkdir(parents=True, exist_ok=True)
    dest = overlays_dir() / f"star_solve_{seq.replace('#', '_')}.jpg"
    overlay.write(overlay.draw(result, n, frame), dest)
    return dest


def animate(seq: str, only_kept: bool = False, full_frame: bool = False,
            mark: list[str] = ()) -> Path:
    """Re-solve one block with a trace and play the solve back into animations/. The result
    isn't filed: solves/ and the gallery stay as `solve` left them. `full_frame` draws the
    whole frame, not just the sky band, into star_solve_<block>_full.mp4. `mark` names stars
    (or "-"-joined chains of them) to point out at the end; see animate.frames."""
    if seq not in nights.sequences():
        raise SystemExit(f"{seq}: not a block in this cache (a result filed with `add` has "
                         f"no frames here to draw)")
    n = night(seq)
    trace: list[dict] = []
    r = solve_wide(n, trace=trace)
    images = ((o, cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR))
              for _, o, blob in nights.read_frames(seq))
    animations_dir().mkdir(parents=True, exist_ok=True)
    dest = animations_dir() / f"star_solve_{seq.replace('#', '_')}{'_full' if full_frame else ''}.mp4"
    path = anim.write(anim.frames(trace, r, n, images, only_kept=only_kept,
                                  full_frame=full_frame, mark=mark), dest)
    print(f"{seq}: {r['status']}{'' if r['status'] == 'solved' else ', ' + r['reason']}")
    return path


def explore(seq: str) -> Path:
    """Re-solve one block with a trace and write explore/<block>.js, which the gallery's
    explore.html replays. Like `animate`, the result isn't filed. The page shows the block's
    frames from nights/ where they are: nothing is copied."""
    if seq not in nights.sequences():
        raise SystemExit(f"{seq}: not a block in this cache (a result filed with `add` has "
                         f"no frames here to draw)")
    n = night(seq)
    trace: list[dict] = []
    r = solve_wide(n, trace=trace)
    t0 = nights.sequences()[seq]["t0"]
    frames = [(int(p.stem) - t0, "../" + quote(p.relative_to(cache_dir()).as_posix()))
              for p in nights.frame_files(seq)[0]]
    data = ex.export(trace, r, n, frames)
    explore_dir().mkdir(parents=True, exist_ok=True)
    dest = explore_dir() / f"{seq.replace('#', '_')}.js"
    dest.write_text("window.EXPLORE = " + json.dumps(data, separators=(",", ":"), allow_nan=False)
                    + ";\n")
    explorer_page()
    print(f"{seq}: {r['status']}{'' if r['status'] == 'solved' else ', ' + r['reason']}")
    return dest


def _one(seq: str) -> dict:
    placeholders = []
    try:
        real, placeholders = nights.frame_files(seq)
        # 8 = tracks.moving_tracks' min_frames; a block that is mostly placeholders has too
        # few, too scattered real frames to say anything about the sky
        if len(real) < 8 or len(placeholders) > len(real):
            r = {"seq": seq, "camera": nights.sequences()[seq]["camera"], "status": "failed",
                 "reason": f"no images: {len(placeholders)}/{len(real) + len(placeholders)} "
                           f"frames are the CDN's placeholder"}
        else:
            r = solve_wide(night(seq))
    except Exception as exc:   # one bad block shouldn't sink the batch
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    solves_dir().mkdir(parents=True, exist_ok=True)
    (solves_dir() / f"solve_{seq}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    if (r.get("reason") or "").startswith("no images") and placeholders:
        # nothing to draw on: the gallery card shows the card the CDN served instead
        overlays_dir().mkdir(parents=True, exist_ok=True)
        shutil.copyfile(placeholders[len(placeholders) // 2],
                        overlays_dir() / f"star_solve_{seq.replace('#', '_')}.jpg")
    elif "n_tracks" in r:         # it got as far as tracks: there is something to draw
        try:
            render(seq, r)
        except Exception as exc:  # a picture is not worth losing the solve over
            print(f"{seq}: no overlay ({type(exc).__name__}: {exc})")
    return r


def summary() -> list[dict]:
    p = solves_dir() / "summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def _merge(new: list[dict]) -> None:
    """Fold `new` into solves/summary.json, replacing rows with the same seq."""
    merged = {r["seq"]: r for r in summary()} | {r["seq"]: r for r in new}
    rows = sorted(merged.values(), key=lambda r: (r["status"] != "solved", r["seq"]))
    solves_dir().mkdir(parents=True, exist_ok=True)
    (solves_dir() / "summary.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")


def add(result: dict, image: np.ndarray | None, source: str) -> None:
    """File a solve of frames the cache doesn't hold, with its overlay (overlay.draw's
    picture, or None), so weather, gallery and agree see it beside the cache's own.

    With no indexed block to supply them, `result` must carry "camera", "t0", "lat" and "lon",
    and "ref_offset" (the frame the overlay is drawn on; overlay.reference_offset gives it
    for a failure) so the weather can be read at the right hour. It is marked with `source`,
    and `ledger` leaves it out: the shipped ledger is this package's own CDN solves."""
    missing = {"camera", "t0", "lat", "lon", "ref_offset"} - set(result)
    if missing:
        raise ValueError(f"{result.get('seq')}: an added result needs {sorted(missing)}")
    r = result | {"source": source}
    solves_dir().mkdir(parents=True, exist_ok=True)
    (solves_dir() / f"solve_{r['seq']}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    if image is not None:
        overlays_dir().mkdir(parents=True, exist_ok=True)
        overlay.write(image, overlays_dir() / f"star_solve_{r['seq'].replace('#', '_')}.jpg")
    _merge([r])


def solve_blocks(seqs: list[str], workers: int = 4) -> list[dict]:
    """Solve `seqs` and merge them into solves/summary.json."""
    with Pool(workers) as pool:
        new = list(pool.imap_unordered(_one, seqs))
    _merge(new)
    return sorted(new, key=lambda r: (r["status"] != "solved", r["seq"]))


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
    elif cmd == "overlay":
        indexed = nights.sequences()   # an added result's overlay came with it
        for seq in args or [r["seq"] for r in summary() if "n_tracks" in r and r["seq"] in indexed]:
            print(render(seq))
    elif cmd == "animate":
        kept, full = "--kept" in args, "--full" in args
        mark = args[args.index("--mark") + 1].split(",") if "--mark" in args else []
        skip = {"--kept", "--full", "--mark", *([",".join(mark)] if mark else [])}
        for seq in [a for a in args if a not in skip]:
            print(animate(seq, only_kept=kept, full_frame=full, mark=mark))
    elif cmd == "explore":
        for seq in args:
            print(explore(seq))
        print(f"open {explorer_page()}?b=<block>, or the block's card in the gallery")
    elif cmd == "agree":
        indexed = nights.sequences()
        rows = summary()
        cross_night.report(rows, {r["seq"]: block(r, indexed)["t0"] for r in rows}, cameras())
    elif cmd == "ledger":
        t0 = {k: v["t0"] for k, v in nights.sequences().items()}
        dest = Path(args[0]) if args else cache_dir() / "pose_ledger.json"
        rows = ledger.build([r for r in summary() if "source" not in r], t0, dest)
        print(f"wrote {dest} ({len(rows)} solves, {len({r['camera'] for r in rows})} cameras)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
