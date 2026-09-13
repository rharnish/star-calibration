"""Bright-star alt-az geometry, mirroring sun_geometry.py's sun() for point stars.

Same sidereal-time machinery as the sun (and star_pixels.py's copy of it), generalized to
any fixed RA/Dec: Greenwich Mean Sidereal Time gives local sidereal time (+ longitude), local
sidereal time minus RA gives hour angle, and the standard alt-az formulas take it from there.
RA/Dec don't need a per-epoch solar-ephemeris approximation like the sun does -- these are
J2000 catalog positions, accurate to a few arcmin over decades (precession is ~0.014 deg/yr,
negligible next to the few-pixel detection noise this project works at).

The catalog (data/meta/bright_stars.json) is the HYG database (github.com/astronexus/HYG-Database,
via the kiloquad/__HYG-Database mirror, whose master branch resolves; the upstream repo's
current default branch does not serve hygdata_v3.csv at that path) filtered to mag <= 4.0 --
523 stars, plenty dense for pattern-matching an asterism without vastly outrunning what these
cameras can actually resolve as point sources. See star_match.py for what actually picks a
constellation out of a frame; this module only answers "where is star X, and what's up".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CATALOG = json.loads((Path(__file__).resolve().parents[3] / "data" / "meta" / "bright_stars.json").read_text())
STARS = {r["name"]: (r["ra_deg"], r["dec_deg"]) for r in CATALOG}

# Kept for backward compatibility with earlier exploration scripts that named these directly.
ORION = ["Betelgeuse", "Rigel", "Bellatrix", "Saiph", "Mintaka", "Alnilam", "Alnitak"]
SCORPIUS = ["Antares", "Dschubba", "Shaula", "Sargas", "Kaus Australis"]


def gmst_deg(epoch):
    n = np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0
    return ((18.697374558 + 24.06570982441908 * n) % 24) * 15


def altaz(ra_deg: float, dec_deg: float, epoch, lat: float, lon: float):
    """(alt_deg, az_deg) for one catalog star at one or many epochs. az from north, clockwise."""
    lst = (gmst_deg(epoch) + lon) % 360
    ha = np.radians((lst - ra_deg) % 360)
    la, dec = np.radians(lat), np.radians(dec_deg)
    alt = np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(-np.sin(ha) * np.cos(dec),
                     np.cos(la) * np.sin(dec) - np.sin(la) * np.cos(dec) * np.cos(ha))
    return np.degrees(alt), np.degrees(az) % 360


def stars_altaz(names: list[str], epoch: float, lat: float, lon: float):
    """alt, az arrays (one per name) at a single epoch."""
    alt = np.array([altaz(*STARS[n], epoch, lat, lon)[0] for n in names])
    az = np.array([altaz(*STARS[n], epoch, lat, lon)[1] for n in names])
    return alt, az


def _unit_vec(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def local_tangent_coords(az: np.ndarray, el: np.ndarray, ref_idx: int):
    """(dx, dy) in radians for each (az, el), in a tangent plane centered on entry `ref_idx`,
    in pixel-image convention (dx: right/increasing az-ish, dy: DOWN/decreasing elevation --
    already flipped from the natural math y-up convention, since every caller here is about
    to compare these against pixel coordinates). Valid to a fraction of a percent within
    ~10-15 deg -- curvature only matters over larger spans; see fisheye.py's docstring for why
    that matters and callers should keep triangles within that radius.
    """
    ref = _unit_vec(az[ref_idx], el[ref_idx])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(ref, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, ref)
    out = []
    for a, e in zip(az, el):
        t = _unit_vec(a, e)
        theta = np.arccos(np.clip(t @ ref, -1, 1))
        s = np.sin(theta)
        if s < 1e-9:
            out.append((0.0, 0.0))
            continue
        out.append((theta * (t @ right) / s, -theta * (t @ up) / s))
    return np.array(out)


def visible_stars(cam: dict, epoch: float, mag_limit: float = 4.0, fov_margin_deg: float = 20.0,
                   min_alt_deg: float = -3.0):
    """Catalog stars plausibly in frame: above the horizon and within the camera's nominal
    azimuth window, padded by `fov_margin_deg` for however wrong the published pose might be
    (the whole premise of this exercise). Returns a list of dicts with name/ra/dec/mag/alt/az,
    brightest first -- the pool `star_match.py` searches for a matching pixel pattern.
    """
    out = []
    half_fov = cam["fov"] / 2.0 + fov_margin_deg
    for r in CATALOG:
        if r["mag"] > mag_limit:
            continue
        alt, az = altaz(r["ra_deg"], r["dec_deg"], epoch, cam["lat"], cam["lon"])
        if alt < min_alt_deg:
            continue
        d_az = (az - cam["az"] + 180) % 360 - 180
        if abs(d_az) > half_fov:
            continue
        out.append({**r, "alt": float(alt), "az": float(az)})
    out.sort(key=lambda r: r["mag"])
    return out


if __name__ == "__main__":
    # sanity: Orion visible from Palomar (hp-s-mobo-c) predawn Oct 21 2024, not from Boucher
    # Hill (bh-s-mobo-c) at 23:05 in July -- matches what NOTES.md records finding by eye.
    alt, az = stars_altaz(ORION, 1729513594, 33.36302, -116.83622)
    print("hp-s-mobo-c Orion:", list(zip(ORION, alt.round(1), az.round(1))))
    alt, az = stars_altaz(ORION, 1722146709, 33.33462, -116.91938)
    print("bh-s-mobo-c Orion (should be below horizon):", list(zip(ORION, alt.round(1), az.round(1))))

    cam = {"lat": 33.36302, "lon": -116.83622, "az": 180, "fov": 90}
    vis = visible_stars(cam, 1729513594)
    print(f"\n{len(vis)} catalog stars in hp-s-mobo-c's nominal FOV window:")
    for r in vis[:15]:
        print(f"  {r['name']:16s} mag {r['mag']:5.2f}  alt {r['alt']:6.1f}  az {r['az']:6.1f}")
