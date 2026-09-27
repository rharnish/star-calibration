"""Night frame blocks from HPWREN's public CDN, for star-track calibration.

The CDN keeps only the last ~89 days of JPGs public (older ones sit in Glacier Deep Archive
and need an HPWREN staff restore), so recent nights are the source. They measure each
camera's pose *now*; whether that applies to another date is the pose ledger's call
(star_calibration.ledger), not this module's. Moonlight is not a problem: moonlit nights
solve as well as dark ones (plume-triangulation's NOTES.md, 2026-09-15).

Q blocks are local time: Q1 = 00:00-02:59 America/Los_Angeles (verified: the 2026-09-10 Q1
list starts at 07:00:57 UTC = 00:00:57 PDT). Frames are ~1/min.

Layout, under the shared cache (hpwren.cache_dir(), $HPWREN_CACHE):
nights/<cam>/<YYYYMMDD>_Q<n>/<epoch>.jpg, indexed in nights.json as blocks named
"hpwren_<YYYYMMDD>_Q<n>_<cam>", each with its camera, t0 (the median frame's epoch; track
offsets count from it) and its directory relative to nights/. A whole night -- Q7 and Q8 of
the evening's date, Q1 and Q2 of the next -- lives in <YYYYMMDD>_N/ and indexes as
"hpwren_<YYYYMMDD>_N_<cam>" (see `fetch_night`).

Data credit: HPWREN, https://www.hpwren.ucsd.edu/

    python -m star_calibration.hpwren.nights 20260911 hp-s-mobo-c vo-n-mobo-c ...
    python -m star_calibration.hpwren.nights 20260911 @cams.txt   (a JSON list of camera names)
    python -m star_calibration.hpwren.nights --night 20260713 vo-n-mobo-c   (18:00-06:00)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import cache_dir, cameras

CDN = "https://cdn.hpwren.ucsd.edu"


def frames_dir() -> Path:
    return cache_dir() / "nights"


def manifest_path() -> Path:
    return cache_dir() / "nights.json"


def _get(url: str, tries: int = 3) -> bytes:
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()
        except Exception as exc:   # 403 (Glacier), 404 (offline camera), timeouts
            err = exc
            time.sleep(2 * (k + 1))
    raise err


def seq_name(cam: str, day: str, q: int) -> str:
    return f"hpwren_{day}_Q{q}_{cam}"


def fetch(cam: str, day: str, q: int = 1, n_frames: int = 90) -> tuple[str, str, int, int]:
    """The first n_frames of one Q block (90 = 00:00-01:30 for Q1), skipping files on disk."""
    try:
        listing = _get(f"{CDN}/hpwren-cameras/{cam}/{day[:4]}/{day}/{day}_{cam}_Q{q}.txt")
    except Exception:
        return cam, day, 0, 0
    names = [l.strip() for l in listing.decode().splitlines() if l.strip().endswith(".jpg")][:n_frames]
    d = frames_dir() / cam / f"{day}_Q{q}"
    d.mkdir(parents=True, exist_ok=True)
    got = 0
    for n in names:
        p = d / n
        if not p.exists() or p.stat().st_size == 0:
            try:
                p.write_bytes(_get(f"{CDN}/MTA/{cam}/large/{day}/Q{q}/{n}"))
            except Exception:
                continue
        got += 1
    return cam, day, got, len(names)


def fetch_night(cam: str, evening_day: str) -> tuple[str, str, int, int]:
    """Every frame from 18:00 on `evening_day` to 06:00 the next morning (Q7, Q8, Q1, Q2) into
    one <evening_day>_N/ directory, skipping files on disk. Twilight frames come along; the
    track extractor drops them by sun elevation, so the directory holds the whole block."""
    nxt = (datetime.strptime(evening_day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    d = frames_dir() / cam / f"{evening_day}_N"
    d.mkdir(parents=True, exist_ok=True)
    got = listed = 0
    for day, q in ((evening_day, 7), (evening_day, 8), (nxt, 1), (nxt, 2)):
        try:
            listing = _get(f"{CDN}/hpwren-cameras/{cam}/{day[:4]}/{day}/{day}_{cam}_Q{q}.txt")
        except Exception:
            continue
        names = [l.strip() for l in listing.decode().splitlines() if l.strip().endswith(".jpg")]
        listed += len(names)

        def one(n, day=day, q=q):
            p = d / n
            if p.exists() and p.stat().st_size > 0:
                return 1
            try:
                p.write_bytes(_get(f"{CDN}/MTA/{cam}/large/{day}/Q{q}/{n}"))
                return 1
            except Exception:
                return 0
        with ThreadPoolExecutor(8) as pool:
            got += sum(pool.map(one, names))
    return cam, evening_day, got, listed


def index() -> dict:
    """Rebuild the manifest from what's on disk."""
    cams, root = cameras(), frames_dir()
    out = {}
    for d in sorted([*root.glob("*/*_Q*"), *root.glob("*/*_N")]):
        epochs = sorted(int(p.stem) for p in d.glob("*.jpg") if p.stat().st_size > 0)
        if len(epochs) < 8:
            continue
        cam = d.parent.name
        c = cams[cam]
        if d.name.endswith("_N"):
            day, q = d.name[:-2], None
            s = f"hpwren_{day}_N_{cam}"
        else:
            day, q = d.name.split("_Q")
            q = int(q)
            s = seq_name(cam, day, q)
        out[s] = {"seq": s, "camera": cam, "day": day, "q": q,
                  "t0": epochs[len(epochs) // 2], "lat": c["lat"], "lon": c["lon"],
                  "n_frames": len(epochs), "dir": str(d.relative_to(root))}
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(out, indent=1) + "\n")
    return out


def sequences() -> dict:
    """The indexed blocks, by name."""
    m = manifest_path()
    return json.loads(m.read_text()) if m.exists() else {}


def block_dir(s: dict) -> Path:
    """The directory holding one indexed block's frames."""
    return frames_dir() / s["dir"]


def read_frames(seq: str) -> list[tuple[int, int, bytes]]:
    """(epoch, offset from t0, jpeg bytes), in time order: tracks.collect's input."""
    s = sequences()[seq]
    files = sorted(block_dir(s).glob("*.jpg"), key=lambda p: int(p.stem))
    return [(int(p.stem), int(p.stem) - s["t0"], p.read_bytes()) for p in files if p.stat().st_size > 0]


def frame_at(seq: str, offset: float) -> tuple[int, int, bytes]:
    """The one frame nearest `offset` seconds from t0, as read_frames would give it, reading
    only that file (a whole night is ~470 frames of a few MB)."""
    s = sequences()[seq]
    files = sorted((p for p in block_dir(s).glob("*.jpg") if p.stat().st_size > 0),
                   key=lambda p: int(p.stem))
    p = min(files, key=lambda p: abs(int(p.stem) - s["t0"] - offset))
    return int(p.stem), int(p.stem) - s["t0"], p.read_bytes()


def main(argv: list[str]) -> None:
    night = "--night" in argv
    argv = [a for a in argv if a != "--night"]
    day, args = argv[0], argv[1:]
    cams = json.loads(Path(args[0][1:]).read_text()) if args and args[0].startswith("@") else args
    with ThreadPoolExecutor(4) as pool:
        for cam, d, got, listed in pool.map(lambda c: (fetch_night if night else fetch)(c, day), cams):
            print(f"{d} {cam:18s} {got}/{listed} frames", flush=True)
    print(f"indexed {len(index())} night blocks")


if __name__ == "__main__":
    main(sys.argv[1:])
