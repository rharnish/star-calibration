"""Moonless-night frame blocks from HPWREN's public CDN, for star-track calibration.

FIgLib is exhausted for this: all 456 archives are local and none holds an unsolved night
sequence on the 60 cameras the scored fires use (checked 2026-09-13). The CDN keeps only
the last ~89 days of JPGs public (older ones sit in Glacier Deep Archive and need an HPWREN
staff restore), so recent moonless nights are the only source. They measure each camera's
pose *now*; whether that applies to an older fire is the pose ledger's call
(src/figlib/pose_ledger.py), not this module's.

Q blocks are local time: Q1 = 00:00-02:59 America/Los_Angeles (verified: the 2026-09-10 Q1
list starts at 07:00:57 UTC = 00:00:57 PDT). Frames are ~1/min.

Layout: data/hpwren_nights/<cam>/<YYYYMMDD>_Q<n>/<epoch>.jpg (gitignored), indexed in
out/sky/data/hpwren_nights.json as pseudo-sequences "hpwren_<YYYYMMDD>_Q<n>_<cam>" that
stars.tracks.collect and stars.solve treat like FIgLib sequences. A whole night -- Q7 and Q8
of the evening's date, Q1 and Q2 of the next -- lives in <YYYYMMDD>_N/ and indexes as
"hpwren_<YYYYMMDD>_N_<cam>" (see `fetch_night`; used by stars.window_ablation).

Data credit: HPWREN, https://www.hpwren.ucsd.edu/

    python -m src.figlib.stars.nights 20260911 hp-s-mobo-c vo-n-mobo-c ...
    python -m src.figlib.stars.nights 20260911 @cams.json        (a JSON list of camera names)
    python -m src.figlib.stars.nights --night 20260713 vo-n-mobo-c   (18:00-06:00, all frames)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKY = ROOT / "out" / "sky"
FRAMES = ROOT / "data/hpwren_nights"
MANIFEST = SKY / "data/hpwren_nights.json"
CDN = "https://cdn.hpwren.ucsd.edu"


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
    d = FRAMES / cam / f"{day}_Q{q}"
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
    d = FRAMES / cam / f"{evening_day}_N"
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
    cams = json.loads((ROOT / "data/meta/cams.json").read_text())
    out = {}
    for d in sorted([*FRAMES.glob("*/*_Q*"), *FRAMES.glob("*/*_N")]):
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
                  "n_frames": len(epochs), "dir": str(d.relative_to(ROOT))}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=1) + "\n")
    return out


def sequences() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def read_frames(seq: str) -> list[tuple[int, int, bytes]]:
    """(epoch, offset from t0, jpeg bytes), in time order -- detect_yolo.read_frames' shape."""
    s = sequences()[seq]
    files = sorted((ROOT / s["dir"]).glob("*.jpg"), key=lambda p: int(p.stem))
    return [(int(p.stem), int(p.stem) - s["t0"], p.read_bytes()) for p in files if p.stat().st_size > 0]


if __name__ == "__main__":
    night = "--night" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--night"]
    day, args = argv[0], argv[1:]
    cams = json.loads(Path(args[0][1:]).read_text()) if args and args[0].startswith("@") else args
    with ThreadPoolExecutor(4) as pool:
        for cam, d, got, listed in pool.map(lambda c: (fetch_night if night else fetch)(c, day), cams):
            print(f"{d} {cam:18s} {got}/{listed} frames", flush=True)
    print(f"indexed {len(index())} night blocks")
