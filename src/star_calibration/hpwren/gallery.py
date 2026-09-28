"""One page to look at every star solve: solved beside failed, and each against the weather.

Reads the cache's solves/summary.json and solves/solve_weather.json (weather.py), and the
star_solve_<block>.jpg overlays in the cache's overlays/ that `calibrate solve` draws
(overlay.py: tracks and fitted stars on the block's own frame). Writes gallery/index.html in
the cache, with 480 px thumbnails beside it; the page links the full overlays by relative
path, so it is not self-contained. A block with no overlay still gets its card. A block
with a video in animations/ (`calibrate animate`, on request) plays it in its detail view,
and one with a replay in explore/ (`calibrate explore`) links explore.html, the browser replay.

    python -m star_calibration.hpwren.calibrate overlay   # redraw any missing overlays
    python -m star_calibration.hpwren.weather
    python -m star_calibration.hpwren.gallery

Outcome classes (the page colours by these):
  solved  -- status "solved"
  stars   -- failed, but the tracks were there: below the solve cutoff, or no start converged
  dark    -- failed with too few moving tracks to try: cloud, fog, lens, or a dead camera
  noimg   -- failed because the camera sent nothing: the CDN served its "No Images!" card
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from . import cache_dir, nights
from .calibrate import (animations_dir, explore_dir, explorer_page, overlays_dir, solves_dir,
                        summary)

THUMB_W = 480


def gallery_dir() -> Path:
    return cache_dir() / "gallery"


def outcome(r: dict) -> str:
    if r["status"] == "solved":
        return "solved"
    reason = r.get("reason") or ""
    if reason.startswith("no images"):
        return "noimg"
    return "dark" if reason.startswith("only ") else "stars"


def thumb(src: Path) -> str:
    thumbs = gallery_dir() / "thumbs"
    dest = thumbs / src.name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        img = cv2.imread(str(src))
        h = round(img.shape[0] * THUMB_W / img.shape[1])
        cv2.imwrite(str(dest), cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    return f"{thumbs.name}/{dest.name}"


def rows() -> list[dict]:
    wpath = solves_dir() / "solve_weather.json"
    weather = {w["seq"]: w for w in json.loads(wpath.read_text())} if wpath.exists() else {}
    (gallery_dir() / "thumbs").mkdir(parents=True, exist_ok=True)
    blocks = nights.sequences()
    out = []
    for r in summary():
        img = overlays_dir() / f"star_solve_{r['seq'].replace('#', '_')}.jpg"
        video = animations_dir() / f"star_solve_{r['seq'].replace('#', '_')}.mp4"
        replay = explore_dir() / f"{r['seq'].replace('#', '_')}.js"
        w = weather.get(r["seq"], {})
        out.append({
            "seq": r["seq"], "camera": r.get("camera") or blocks[r["seq"]]["camera"],
            "imager": r.get("imager"),
            "status": r["status"], "cls": outcome(r), "reason": r.get("reason"),
            "n_tracks": r.get("n_tracks", 0), "n_tracks_raw": r.get("n_tracks_raw", 0),
            "n_stars": r.get("n_stars"), "median_px": r.get("median_px"),
            "rmse_px": r.get("rmse_px"), "k_ratio": (r.get("pose") or {}).get("k_ratio"),
            "epoch": w.get("epoch"), "lat": w.get("lat"), "lon": w.get("lon"),
            "fc": w.get("forecast"), "ra": w.get("reanalysis"),
            "img": f"../overlays/{img.name}" if img.exists() else None,
            "thumb": thumb(img) if img.exists() else None,
            "video": f"../animations/{video.name}" if video.exists() else None,
            "explore": f"explore.html?b={replay.stem}" if replay.exists() else None,
        })
    return out


def main() -> None:
    data = rows()
    page = (Path(__file__).with_name("gallery.html").read_text()
            .replace("/*__DATA__*/[]", json.dumps(data, default=float)))
    out = gallery_dir() / "index.html"
    out.write_text(page)
    explorer_page()
    missing = [r["seq"] for r in data if not r["img"]]
    print(f"wrote {out} ({len(data)} solves, {len(missing)} without an overlay)")


if __name__ == "__main__":
    main()
