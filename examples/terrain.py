"""Terrain checks for the worked examples: landmarks and DEM crests projected through a pose.

Nothing here is fitted to the image. A solved pose is right where what it predicts lands on
what the daytime frame shows.

The terrain model is the Copernicus DEM GLO-30, read from its public bucket on AWS and cached
under $HPWREN_CACHE/dem: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH
2014-2018, provided under COPERNICUS by the European Union and ESA.
"""
from __future__ import annotations

import cv2
import numpy as np

from star_calibration import fisheye
from star_calibration import terrain as T
from star_calibration.hpwren import cache_dir
from star_calibration.terrain import (K_REFR, R_EARTH, az_el, destination,  # noqa: F401
                                      eye_height, sky_mask)


def dem_dir():
    return cache_dir() / "dem"


def dem_tile(lat: int, lon: int) -> np.ndarray:
    return T.dem_tile(lat, lon, dem_dir())


def terrain_height(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return T.terrain_height(lat, lon, dem_dir())


def skyline(cam: dict, **kw):
    """The skyline all round the camera (star_calibration.terrain.skyline), from the cache's tiles."""
    return T.skyline(cam, dem_dir(), **kw)


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
