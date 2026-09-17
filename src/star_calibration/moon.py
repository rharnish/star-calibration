"""Moon position and illuminated fraction, for testing whether star solves need a dark sky.

`nights.py` only ever fetched moonless blocks, on the assumption that moonlight washes out
the star field. That assumption was never measured. This module supplies the axis to measure
it against: how bright the moon was and whether it was even above the horizon while a night
block was recorded.

Meeus, *Astronomical Algorithms* ch. 47/48, truncated to the largest periodic terms -- a few
arcminutes in position and ~0.01 in illuminated fraction, which is far finer than the
question "was there a fat moon in the sky" needs. Same sidereal-time machinery as
`catalog.altaz` and `sun.sun`, so all three agree on where the observer is pointing.
"""
from __future__ import annotations

import numpy as np

from .sun import sun as _sun_altaz

R_SUN_KM = 149_597_870.0


def _jd_centuries(epoch):
    return (np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0) / 36525.0


def _ecliptic(epoch):
    """(lambda, beta, delta_km): geocentric ecliptic longitude/latitude in degrees, distance."""
    T = _jd_centuries(epoch)
    d2r = np.radians
    Lp = (218.3164477 + 481267.88123421 * T) % 360          # mean longitude
    D = d2r((297.8501921 + 445267.1114034 * T) % 360)       # mean elongation
    M = d2r((357.5291092 + 35999.0502909 * T) % 360)        # sun's mean anomaly
    Mp = d2r((134.9633964 + 477198.8675055 * T) % 360)      # moon's mean anomaly
    F = d2r((93.2720950 + 483202.0175233 * T) % 360)        # argument of latitude
    lam = (Lp + 6.289 * np.sin(Mp) + 1.274 * np.sin(2 * D - Mp) + 0.658 * np.sin(2 * D)
           + 0.214 * np.sin(2 * Mp) - 0.186 * np.sin(M) - 0.114 * np.sin(2 * F)) % 360
    beta = (5.128 * np.sin(F) + 0.280 * np.sin(Mp + F) - 0.278 * np.sin(F - Mp)
            - 0.173 * np.sin(2 * D - F))
    delta = (385001 - 20905 * np.cos(Mp) - 3699 * np.cos(2 * D - Mp)
             - 2956 * np.cos(2 * D) - 570 * np.cos(2 * Mp))
    return lam, beta, delta


def _radec(epoch):
    lam, beta, _ = _ecliptic(epoch)
    T = _jd_centuries(epoch)
    eps = np.radians(23.439291 - 0.0130042 * T)
    l, b = np.radians(lam), np.radians(beta)
    ra = np.arctan2(np.sin(l) * np.cos(eps) - np.tan(b) * np.sin(eps), np.cos(l))
    dec = np.arcsin(np.sin(b) * np.cos(eps) + np.cos(b) * np.sin(eps) * np.sin(l))
    return np.degrees(ra) % 360, np.degrees(dec)


def moon(epoch, lat: float, lon: float):
    """(elevation, azimuth) in degrees, matching `sun.sun`'s signature and conventions."""
    ra, dec = _radec(epoch)
    n = np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    ha = np.radians(gmst * 15 + lon - ra)
    la, d = np.radians(lat), np.radians(dec)
    el = np.degrees(np.arcsin(np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(ha)))
    az = (np.degrees(np.arctan2(np.sin(ha),
                                np.cos(ha) * np.sin(la) - np.tan(d) * np.cos(la))) + 180) % 360
    return el, az


def illuminated_fraction(epoch):
    """0 at new moon, 1 at full. Meeus 48.1 via the sun-moon elongation."""
    lam, beta, delta = _ecliptic(epoch)
    T = _jd_centuries(epoch)
    n = np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0
    Ls = (280.460 + 0.9856474 * n) % 360
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam_s = Ls + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    psi = np.arccos(np.cos(np.radians(beta)) * np.cos(np.radians(lam - lam_s)))
    i = np.arctan2(R_SUN_KM * np.sin(psi), delta - R_SUN_KM * np.cos(psi))
    return (1 + np.cos(i)) / 2


def conditions(epoch, lat: float, lon: float) -> dict:
    """What the moon was doing at one instant, as the night-picking code wants it."""
    el, az = moon(epoch, lat, lon)
    k = illuminated_fraction(epoch)
    sun_el, _ = _sun_altaz(epoch, lat, lon)
    return {"moon_el": float(el), "moon_az": float(az), "illum": float(k),
            "sun_el": float(sun_el), "moon_up": bool(el > 0),
            # a rough "how much light is the moon actually putting in the sky" number:
            # illuminated fraction times how high it is, zero when it is down.
            "moon_light": float(k * max(np.sin(np.radians(el)), 0.0))}


if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone
    for a in sys.argv[1:] or ["1789112704"]:
        e = int(a)
        c = conditions(e, 33.33462, -116.91938)
        t = datetime.fromtimestamp(e, timezone.utc).isoformat()
        print(f"{t}  illum {c['illum']:.2f}  moon el {c['moon_el']:+6.1f}  "
              f"light {c['moon_light']:.3f}  sun el {c['sun_el']:+.1f}")
