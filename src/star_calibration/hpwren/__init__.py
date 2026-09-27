"""HPWREN's cameras, calibrated from the night sky: the library applied to one network.

HPWREN (https://www.hpwren.ucsd.edu/) runs ~500 fixed cameras across southern California,
published with a nameplate azimuth and field of view. This package holds what calibrating
them needs and produces:

  cams.json         the published camera table: site, lat, lon, elev, az, fov, imager,
                    pitch/roll/yaw (zero placeholders almost everywhere) and agl, derived from
                    sites.js, HPWREN's own listing, kept verbatim beside it
  nights            moonless-night frame blocks from HPWREN's public CDN, into a local cache
  calibrate         fetch -> tracks -> solve -> ledger, from the command line
  pose_ledger.json  the solved poses, one entry per camera-night (ledger.py has the rules)
  weather, gallery  why a night did or didn't solve: cloud cover, and a contact sheet

Frames and track caches live outside any repository, under $HPWREN_CACHE (default
~/.cache/hpwren), so every project that reads them shares one copy.

Data credit: HPWREN, https://www.hpwren.ucsd.edu/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cache_dir() -> Path:
    """$HPWREN_CACHE, or ~/.cache/hpwren: frames, tracks and solves, shared across projects."""
    return Path(os.environ.get("HPWREN_CACHE") or Path.home() / ".cache" / "hpwren")


def cameras() -> dict[str, dict]:
    """The published camera table, by camera name (e.g. "hp-s-mobo-c")."""
    return json.loads((HERE / "cams.json").read_text())


def ledger_path() -> Path:
    """The shipped pose ledger: every CDN night this package has solved."""
    return HERE / "pose_ledger.json"
