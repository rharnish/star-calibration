"""Forecast cloud cover at each CDN block's reference frame, for the article's moonlight numbers.

Reads the cache's solves (the v0.4.0 network run) and the weather cache, fetching what is
missing from Open-Meteo. Writes ../static/cloud.json: {block: cloud cover %}.

    .venv/bin/python examples/hpwren_article/scripts/weather_fill.py
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from star_calibration.hpwren import weather as Wm, cameras, nights, cache_dir
from star_calibration.hpwren.calibrate import solves_dir
from star_calibration.hpwren.calibrate import summary
R = [r for r in summary() if "source" not in r and "_N_" not in r["seq"] and r["seq"] in nights.sequences()]
seqs, cams = nights.sequences(), cameras()
CACHE = solves_dir() / "weather_cache.json"
cache = json.loads(CACHE.read_text())
out = {}
try:
    for r in R:
        s = seqs[r["seq"]]; c = cams[r["camera"]]
        ep = int(s["t0"] + r.get("ref_offset", 2700))
        lat, lon = round(c["lat"], 4), round(c["lon"], 4)
        day = datetime.fromtimestamp(ep, timezone.utc).strftime("%Y-%m-%d")
        out[r["seq"]] = Wm.at(Wm.fetch("forecast", lat, lon, day, cache), ep)["cloud_cover"]["at"]
finally:
    CACHE.write_text(json.dumps(cache) + "\n")
json.dump(out, open(Path(__file__).resolve().parents[1] / "static" / "cloud.json", "w"))
print(len(out), sum(v is None for v in out.values()))
