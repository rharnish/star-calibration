"""Weather at each star solve, from Open-Meteo, so solve quality can be read against the sky.

Two sources per solve, both hourly at the camera's lat/lon:
  * forecast -- Open-Meteo's historical-forecast archive: what the operational models
    (GFS/HRRR/ECMWF blend) forecast for that hour, as issued at the time. Has visibility.
  * reanalysis -- ERA5 (archive-api), the after-the-fact best estimate. No visibility, but
    a second opinion on cloud that doesn't share the forecast's errors.

The value kept is the hour nearest the solve's reference frame (t0 + ref_offset), plus the
+-6 h profile around it. One request per (site, UTC day) is cached in weather_cache.json, so a
re-run only asks for site-days it hasn't seen. Both files live in the cache's solves/.

    python -m star_calibration.hpwren.weather     # writes solves/solve_weather.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from . import cameras, nights
from .calibrate import load_tracks, solves_dir, summary

HOURLY = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
          "visibility", "relative_humidity_2m", "dew_point_2m", "temperature_2m",
          "precipitation", "wind_speed_10m"]
SOURCES = {
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "reanalysis": "https://archive-api.open-meteo.com/v1/archive",
}
PROFILE_H = 6


def ref_epoch(r: dict) -> int:
    """The solve's reference frame; for rows that stopped before choosing one, the median
    track time, or the block's start if it has no tracks at all."""
    s = nights.sequences()[r["seq"]]
    if "ref_offset" in r:
        return int(s["t0"] + r["ref_offset"])
    tracks, _ = load_tracks(r["seq"])
    offs = sorted(o for t in tracks for o in t)
    return int(s["t0"] + (offs[len(offs) // 2] if offs else 0))


def site_latlon(seq: str) -> tuple[float, float]:
    s = nights.sequences()[seq]
    c = cameras().get(s["camera"], {})
    return round(s.get("lat", c.get("lat")), 4), round(s.get("lon", c.get("lon")), 4)


def fetch(source: str, lat: float, lon: float, day: str, cache: dict) -> dict:
    """Hourly series for the UTC day before through the day after `day`, so a +-6 h profile
    around any hour of `day` is covered."""
    key = f"{source}|{lat}|{lon}|{day}"
    if key in cache:
        return cache[key]
    d = datetime.strptime(day, "%Y-%m-%d")
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY), "timezone": "GMT",
              "start_date": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
              "end_date": (d + timedelta(days=1)).strftime("%Y-%m-%d")}
    if source == "reanalysis":
        params["hourly"] = ",".join(h for h in HOURLY if h != "visibility")
    for attempt in range(4):
        resp = requests.get(SOURCES[source], params=params, timeout=60)
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    cache[key] = resp.json()["hourly"]
    return cache[key]


def at(hourly: dict, epoch: int) -> dict:
    """The hour nearest `epoch`, and each variable's +-PROFILE_H hourly profile around it."""
    ts = np.array([datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp()
                   for t in hourly["time"]])
    i = int(np.argmin(np.abs(ts - epoch)))
    lo, hi = max(0, i - PROFILE_H), min(len(ts), i + PROFILE_H + 1)
    return {v: {"at": hourly[v][i], "profile": hourly[v][lo:hi]}
            for v in hourly if v != "time"} | {"hour_utc": hourly["time"][i]}


def main() -> None:
    rows = summary()
    CACHE, OUT = solves_dir() / "weather_cache.json", solves_dir() / "solve_weather.json"
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out = []
    try:
        for n, r in enumerate(rows):
            ep = ref_epoch(r)
            lat, lon = site_latlon(r["seq"])
            day = datetime.fromtimestamp(ep, timezone.utc).strftime("%Y-%m-%d")
            row = {"seq": r["seq"], "epoch": ep, "lat": lat, "lon": lon}
            for src in SOURCES:
                row[src] = at(fetch(src, lat, lon, day, cache), ep)
            out.append(row)
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {r['seq']}", flush=True)
    finally:
        CACHE.write_text(json.dumps(cache) + "\n")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT} ({len(out)} solves)")


if __name__ == "__main__":
    main()
