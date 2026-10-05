"""Terrain checks for the worked examples: landmarks and DEM crests projected through a pose.

Nothing here is fitted to the image. A solved pose is right where what it predicts lands on
what the daytime frame shows.

The terrain model is the Copernicus DEM GLO-30, read from its public bucket on AWS and cached
under $HPWREN_CACHE/dem: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH
2014-2018, provided under COPERNICUS by the European Union and ESA.
"""
from __future__ import annotations

import math
import urllib.request

import cv2
import numpy as np

from star_calibration import fisheye
from star_calibration.hpwren import cache_dir

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # GeoTIFF tags OpenCV doesn't know

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


def dem_tile(lat: int, lon: int) -> np.ndarray:
    """One 1-degree tile, its north-west corner at (lat + 1, lon), downloaded once."""
    name = (f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
    if name not in _tiles:
        path = cache_dir() / "dem" / f"{name}.tif"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("downloading", name)
            urllib.request.urlretrieve(DEM_URL.format(name=name), path)
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        a[a < -1000] = 0.0                      # sea and no-data
        _tiles[name] = a
    return _tiles[name]


def terrain_height(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear height (m) at each point; a pixel's value belongs to its centre."""
    out = np.zeros(lat.shape, np.float32)
    la0, lo0 = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la, lo in set(zip(la0.ravel(), lo0.ravel())):
        m = (la0 == la) & (lo0 == lo)
        a = dem_tile(la, lo)
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


def crests(cam: dict, max_km: float = 80.0, step_deg: float = 0.05, step_m: float = 30.0,
           min_hidden_m: float = 1000.0, min_km: float = 1.5):
    """Every terrain crest the camera can see: (az, el, km, is_skyline) arrays.

    A ray is marched out along every azimuth a little past the field of view. A crest is the
    last point on a ray before the terrain behind it drops out of sight for at least
    `min_hidden_m`; the outermost is the skyline, unless it is sea or the end of the march."""
    azs = np.arange(cam["az"] - cam["fov"] / 2 - 13, cam["az"] + cam["fov"] / 2 + 13, step_deg)
    dist = np.arange(60.0, max_km * 1000, step_m)
    lat, lon = destination(cam["lat"], cam["lon"], azs[:, None], dist[None, :])
    h = terrain_height(lat, lon)
    angle = np.degrees(np.arctan2(h - eye_height(cam) - dist ** 2 * (1 - K_REFR) / (2 * R_EARTH), dist))
    seen = angle >= np.concatenate([np.full((len(azs), 1), -90.0),
                                    np.maximum.accumulate(angle, axis=1)[:, :-1]], axis=1)
    out = []
    for i in range(len(azs)):
        v = np.flatnonzero(seen[i])
        for a, b in zip(v[:-1], v[1:]):
            if dist[b] - dist[a] >= min_hidden_m and dist[a] >= min_km * 1000 and h[i, a] > 5:
                out.append((azs[i], angle[i, a], dist[a] / 1000, False))
        j = v[-1]
        if h[i, j] > 5 and dist[j] < max_km * 1000 - 2000:
            out.append((azs[i], angle[i, j], dist[j] / 1000, True))
    az, el, km, sky = (np.array(c) for c in zip(*out))
    return az, el, km, sky.astype(bool)


def project(cam: dict, az, el, W: int, H: int, pose: dict | None = None):
    """Pixel coordinates of directions: through a solved pose and lens (and the camera's
    cx, cy), or, with no pose, the published pose, shared lens and the frame's middle."""
    az, el = np.atleast_1d(np.asarray(az, float)), np.atleast_1d(np.asarray(el, float))
    if pose is None:
        cam = {**cam, "cx": 0.0, "cy": 0.0}
        kw = dict(k_scale=fisheye.K_RATIO * fisheye.initial_k(cam, W), k1=fisheye.K1)
    else:
        kw = dict(d_az=pose["d_az"], d_pitch=pose["d_pitch"], d_roll=pose["d_roll"],
                  k_scale=pose["k_ratio"] * fisheye.initial_k(cam, W), k1=pose["k1"])
    x, y = fisheye.project_fisheye(cam, az, el, W, H, **kw)
    return x * W, y * H


def edge_map(day_bgr: np.ndarray) -> np.ndarray:
    """Bright above minus dark below: what terrain against sky or haze looks like."""
    g = cv2.GaussianBlur(cv2.cvtColor(day_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    e = np.zeros_like(g)
    e[1:-1] = g[:-2] - g[2:]
    return e


def edge_score(edge: np.ndarray, x, y, shifts=range(-30, 31)) -> np.ndarray:
    """Mean edge strength under the projected crests, with every crest shifted vertically by
    each of `shifts` px. A right pose peaks near zero."""
    H, W = edge.shape
    ok = np.isfinite(x) & np.isfinite(y)
    xi = np.round(x[ok]).astype(int)
    out = []
    for dy in shifts:
        yi = np.round(y[ok] + dy).astype(int)
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out.append(edge[yi[m], xi[m]].mean())
    return np.array(out)
