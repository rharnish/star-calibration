"""Sun position from a low-precision solar ephemeris (~0.01 deg, plenty for a dark-sky gate)."""

from __future__ import annotations

import math

import numpy as np


def sun(epoch, lat: float, lon: float):
    """(elevation, azimuth) in degrees for one or many unix epochs; azimuth from north."""
    n = np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    ha = np.radians(gmst * 15 + lon) - ra
    la = math.radians(lat)
    el = np.degrees(np.arcsin(math.sin(la) * np.sin(dec) + math.cos(la) * np.cos(dec) * np.cos(ha)))
    az = (np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * math.sin(la) - np.tan(dec) * math.cos(la))) + 180) % 360
    return el, az
