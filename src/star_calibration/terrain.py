"""Where the sky is: a camera's skyline from a terrain model, and the part of a frame above it.

The solver searches for stars above the skyline (`hpwren.calibrate`'s second pass). The
skyline comes from a ray marched out along every bearing from the lens across the terrain
model, with Earth's curvature and standard refraction: along each bearing it is the highest
sight angle to the ground. `sky_mask` draws it through a pose into a frame.

The terrain model is the Copernicus DEM GLO-30, read from its public bucket on AWS one
1-degree tile at a time and cached under `tiles_dir` (the HPWREN package passes
$HPWREN_CACHE/dem). (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018,
provided under COPERNICUS by the European Union and ESA. Needs OpenCV, which reads the
GeoTIFF tiles.
"""
from __future__ import annotations

import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import fisheye

DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
R_EARTH = 6371000.0
K_REFR = 0.13          # standard refraction coefficient
_tiles: dict[str, np.ndarray] = {}


def eye_height(cam: dict) -> float:
    """The lens's height above sea level: site elevation plus mast."""
    return cam["elev"] + (cam.get("agl") or 0.0)


def az_el(lat1, lon1, h1, lat2, lon2, h2, k_refr=K_REFR):
    """Bearing (deg), apparent elevation (deg) and distance (m) from one point to another,
    with Earth's curvature and refraction."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    az = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                 math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))) % 360
    d = R_EARTH * math.acos(min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))
    el = math.degrees(math.atan2(h2 - h1 - d * d * (1 - k_refr) / (2 * R_EARTH), d))
    return az, el, d


def dem_tile(lat: int, lon: int, tiles_dir: Path | str) -> np.ndarray:
    """One 1-degree tile, its north-west corner at (lat + 1, lon), downloaded once. Open sea
    has no tile: it reads as height 0."""
    import cv2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # GeoTIFF tags OpenCV doesn't know
    name = (f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
    if name not in _tiles:
        path = Path(tiles_dir) / f"{name}.tif"
        sea = path.with_suffix(".sea")          # marks a tile the bucket doesn't have
        if not path.exists() and not sea.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("downloading", name)
            try:
                urllib.request.urlretrieve(DEM_URL.format(name=name), path)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                sea.touch()
        if sea.exists():
            _tiles[name] = np.zeros((2, 2), np.float32)
            return _tiles[name]
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        a[a < -1000] = 0.0                      # sea and no-data
        _tiles[name] = a
    return _tiles[name]


def terrain_height(lat: np.ndarray, lon: np.ndarray, tiles_dir: Path | str) -> np.ndarray:
    """Bilinear height (m) at each point; a pixel's value belongs to its centre."""
    out = np.zeros(lat.shape, np.float32)
    la0, lo0 = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la, lo in set(zip(la0.ravel(), lo0.ravel())):
        m = (la0 == la) & (lo0 == lo)
        a = dem_tile(la, lo, tiles_dir)
        r = np.clip((la + 1 - lat[m]) * a.shape[0] - 0.5, 0, a.shape[0] - 1.001)
        q = np.clip((lon[m] - lo) * a.shape[1] - 0.5, 0, a.shape[1] - 1.001)
        r0, q0 = r.astype(int), q.astype(int)
        fr, fq = r - r0, q - q0
        out[m] = (a[r0, q0] * (1 - fr) * (1 - fq) + a[r0 + 1, q0] * fr * (1 - fq)
                  + a[r0, q0 + 1] * (1 - fr) * fq + a[r0 + 1, q0 + 1] * fr * fq)
    return out


def destination(lat, lon, az, d):
    """The point d metres from (lat, lon) along bearing az, on a sphere."""
    p1, l1, a, dr = math.radians(lat), math.radians(lon), np.radians(az), d / R_EARTH
    p2 = np.arcsin(np.sin(p1) * np.cos(dr) + np.cos(p1) * np.sin(dr) * np.cos(a))
    l2 = l1 + np.arctan2(np.sin(a) * np.sin(dr) * np.cos(p1), np.cos(dr) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


def sight_angles(h, h_eye: float, dist):
    """Apparent elevation (deg) of ground at height `h`, `dist` metres out, from an eye at
    `h_eye`: Earth's curvature less standard refraction."""
    return np.degrees(np.arctan2(h - h_eye - dist ** 2 * (1 - K_REFR) / (2 * R_EARTH), dist))


def skyline(cam: dict, tiles_dir: Path | str, max_km: float = 80.0, step_deg: float = 0.1,
            step_m: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """The skyline all round the camera: (az, el), with el the highest sight angle to the
    ground along each bearing, whether that is a ridge or the end of the march."""
    azs = np.arange(0.0, 360.0, step_deg)
    dist = np.arange(60.0, max_km * 1000, step_m)
    lat, lon = destination(cam["lat"], cam["lon"], azs[:, None], dist[None, :])
    angle = sight_angles(terrain_height(lat, lon, tiles_dir), eye_height(cam), dist)
    return azs, angle.max(axis=1)


def sky_mask(cam: dict, pose: dict, W: int, H: int, sky: tuple, margin_deg: float = 0.5,
             step: int = 8) -> np.ndarray:
    """Frame-sized bool mask of the pixels more than `margin_deg` above the skyline `sky`
    (`skyline`'s (az, el)), through a pose and lens (and the camera's cx, cy). The margin
    clears trees and masts that stand above the terrain model, and how far the pose is off."""
    import cv2
    xs = (np.arange(0, W, step) + step / 2) / W
    ys = (np.arange(0, H, step) + step / 2) / H
    X, Y = np.meshgrid(xs, ys)
    az, el = fisheye.unproject_fisheye(cam, X, Y, W, H, pose["d_az"], pose["d_pitch"], pose["d_roll"],
                                       pose["k_ratio"] * fisheye.initial_k(cam, W), pose["k1"])
    ridge = np.interp(az, sky[0], sky[1], period=360.0)
    m = (el > ridge + margin_deg).astype(np.uint8)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
