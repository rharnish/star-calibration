"""Star positions on the sky at a given moment and place, from a catalog that says what it is.

A catalog position is only meaningful with three facts attached, and this module refuses a
catalog that does not state them:

  frame    -- the reference frame the RA/Dec are in. ICRS and FK5 (J2000) are accepted and
              treated as the same frame: they differ by <= 25 mas, far below a pixel here.
              Anything else (FK4/B1950, ...) needs a conversion this module does not have,
              and raises rather than silently mixing frames.
  equinox  -- for an equinox-based frame, the mean equator and equinox the coordinates refer
              to. Precession runs from here to the equator and equinox of each observation.
              Only 2000.0 is implemented (IAU 1976 precession from J2000).
  epoch    -- the date the positions are *for*. Proper motion carries each star from here to
              the observation date. (A Gaia DR3 catalog, say, is ICRS but epoch 2016.0 --
              frame and epoch are different questions.)

The catalog file carries these in a header -- data/meta/bright_stars.json:

    {"catalog": {"name", "source", "frame", "equinox", "epoch", "pm_units", ...},
     "stars":   [{"name", "ra_deg", "dec_deg", "pmra_mas", "pmdec_mas", "mag", "con"}, ...]}

so a solve that swaps catalogs gets the right corrections from the catalog itself, and every
solve records `model_id()` -- the catalog and the corrections applied -- in its result, where
the pose ledger checks it (pose_ledger.load refuses a ledger that mixes sky models).

The chain for one star, catalog -> pixel, with MODEL switching each step:

  1. proper motion  catalog epoch -> observation date (linear in RA/Dec; mu_alpha* / cos dec)
  2. precession     catalog equinox -> mean equator and equinox of date (IAU 1976, Lieske 1977)
  3. [not modelled] nutation (<= 17") and annual aberration (<= 20.5"): < 1 px on these
                    cameras. A check against Skyfield's full chain (IAU 2000A, aberration)
                    agrees to 0.005 deg (tests/test_window_ablation.py).
  4. alt/az         Greenwich mean sidereal time + longitude -> hour angle -> altitude, azimuth
  5. refraction     true -> apparent altitude (Saemundsson), scaled for the site's elevation

Why precession matters: until 2026-09 the solver used the J2000 positions as if they were of
date. By 2026 the sky has turned ~0.36 deg against them, and every star solve absorbed that
into its pose -- about -0.28 deg of azimuth on every camera, and a residual that drifts across
a night because a pose can absorb a fixed rotation but not one that changes with sidereal
time. NOTES.md, 2026-09-24 (the whole-night window ablation) has the measurement.

The shipped catalog is the HYG database v3 (github.com/astronexus/HYG-Database, via the
kiloquad/__HYG-Database mirror) filtered to mag <= 4.0: 523 stars. HYG v3 gives RA/Dec for
"epoch and equinox 2000.0", derived from Hipparcos (ICRS), and Hipparcos proper motions in
mas/yr with pmra = mu_alpha* (already multiplied by cos dec). `rebuild_catalog` regenerates the
file from the CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "meta" / "bright_stars.json"
SAME_AS_J2000 = ("ICRS", "FK5")                  # frames this module can place without conversion
J2000_UNIX = 946728000.0                         # 2000-01-01 12:00 TT ~ UTC, s
JULIAN_YEAR_S = 365.25 * 86400


def _load(path: Path = CATALOG_PATH):
    doc = json.loads(Path(path).read_text())
    meta, stars = doc["catalog"], doc["stars"]
    missing = [k for k in ("name", "frame", "equinox", "epoch") if k not in meta]
    if missing:
        raise ValueError(f"{path}: catalog header lacks {missing}")
    if meta["frame"] not in SAME_AS_J2000:
        raise NotImplementedError(f"{path}: frame {meta['frame']!r} -- only {SAME_AS_J2000} are supported")
    if float(meta["equinox"]) != 2000.0:
        raise NotImplementedError(f"{path}: equinox {meta['equinox']} -- only 2000.0 is implemented")
    return meta, stars


META, CATALOG = _load()
STARS = {r["name"]: (r["ra_deg"], r["dec_deg"]) for r in CATALOG}
PM = {r["name"]: (r.get("pmra_mas", 0.0), r.get("pmdec_mas", 0.0)) for r in CATALOG}
EPOCH_UNIX = J2000_UNIX + (float(META["epoch"]) - 2000.0) * JULIAN_YEAR_S

# Kept for backward compatibility with earlier exploration scripts that named these directly.
ORION = ["Betelgeuse", "Rigel", "Bellatrix", "Saiph", "Mintaka", "Alnilam", "Alnitak"]
SCORPIUS = ["Antares", "Dschubba", "Shaula", "Sargas", "Kaus Australis"]


def gmst_deg(epoch):
    n = np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0
    return ((18.697374558 + 24.06570982441908 * n) % 24) * 15


# Which steps of the chain run. All on by default since 2026-09-25; switching one off
# reproduces the older solves (precession and refraction off: the pre-2026-09 solver).
MODEL = {"proper_motion": True, "precession": True, "refraction": True}


def set_model(**flags) -> dict:
    for k, v in flags.items():
        if k not in MODEL:
            raise KeyError(k)
        MODEL[k] = bool(v)
    return dict(MODEL)


class using:
    """`with using(precession=False): ...` -- set flags for a block, then put them back."""

    def __init__(self, **flags):
        self.flags, self.saved = flags, None

    def __enter__(self):
        self.saved = dict(MODEL)
        set_model(**self.flags)
        return dict(MODEL)

    def __exit__(self, *exc):
        set_model(**self.saved)
        return False


def model_id() -> str:
    """The catalog and the corrections in force, as recorded in every solve and ledger entry."""
    on = [k for k in ("proper_motion", "precession", "refraction") if MODEL[k]]
    return (f"{META['name']} | {META['frame']} eq {float(META['equinox']):.1f} ep {float(META['epoch']):.2f} | "
            + ("+".join(on) if on else "none"))


def apply_proper_motion(ra_deg, dec_deg, pmra_mas, pmdec_mas, epoch):
    """Carry a catalog position from the catalog's epoch to `epoch` (unix s), linearly.
    pmra is mu_alpha*, so the RA rate is pmra / cos(dec). Linear is exact to < 0.1" over a
    century for these stars."""
    dt = (np.asarray(epoch, float) - EPOCH_UNIX) / JULIAN_YEAR_S
    dec = dec_deg + pmdec_mas * dt / 3.6e6
    ra = ra_deg + pmra_mas * dt / 3.6e6 / np.cos(np.radians(dec_deg))
    return ra, dec


def precess(ra_deg, dec_deg, epoch):
    """J2000 mean (RA, Dec) to the mean equator and equinox of `epoch`: the IAU 1976 rotation
    (zeta, z, theta; Lieske 1977). Nutation and aberration (~20 arcsec) are left out."""
    T = (np.asarray(epoch, float) / 86400 + 2440587.5 - 2451545.0) / 36525
    asec = np.pi / (180 * 3600)
    zeta = (2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3) * asec
    z = (2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3) * asec
    th = (2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3) * asec
    a0, d0 = np.radians(ra_deg), np.radians(dec_deg)
    A = np.cos(d0) * np.sin(a0 + zeta)
    B = np.cos(th) * np.cos(d0) * np.cos(a0 + zeta) - np.sin(th) * np.sin(d0)
    C = np.sin(th) * np.cos(d0) * np.cos(a0 + zeta) + np.cos(th) * np.sin(d0)
    return np.degrees(np.arctan2(A, B) + z) % 360, np.degrees(np.arcsin(np.clip(C, -1, 1)))


def refraction_deg(true_alt_deg, elev_m: float = 1600.0, temp_c: float = 10.0):
    """How far refraction lifts a star at true altitude h (Saemundsson 1986, true -> apparent),
    scaled for a mountain site's thinner air: pressure 1010 hPa * exp(-elev / 8.4 km) --
    ~0.83 of sea level at a typical 1.6 km HPWREN summit. Held at its -1 deg value below that,
    where the formula stops being meaningful and no star is visible anyway."""
    h = np.maximum(np.asarray(true_alt_deg, float), -1.0)
    r_arcmin = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))
    scale = np.exp(-elev_m / 8434.0) * 283.0 / (273.0 + temp_c)
    return r_arcmin * scale / 60


def altaz(ra_deg: float, dec_deg: float, epoch, lat: float, lon: float, elev_m: float = 1600.0,
          pm_mas: tuple[float, float] | None = None):
    """(alt_deg, az_deg) for a catalog position at one or many epochs. az from north, clockwise.

    The position is taken to be in this catalog's frame, equinox and epoch; `pm_mas` is its
    (pmra*, pmdec) if known -- `star_altaz` looks it up by name. elev_m only matters for
    refraction."""
    if pm_mas is not None and MODEL["proper_motion"]:
        ra_deg, dec_deg = apply_proper_motion(ra_deg, dec_deg, pm_mas[0], pm_mas[1], epoch)
    if MODEL["precession"]:
        ra_deg, dec_deg = precess(ra_deg, dec_deg, epoch)
    lst = (gmst_deg(epoch) + lon) % 360
    ha = np.radians((lst - ra_deg) % 360)
    la, dec = np.radians(lat), np.radians(dec_deg)
    alt = np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))
    az = np.arctan2(-np.sin(ha) * np.cos(dec),
                     np.cos(la) * np.sin(dec) - np.sin(la) * np.cos(dec) * np.cos(ha))
    alt = np.degrees(alt)
    if MODEL["refraction"]:
        alt = alt + refraction_deg(alt, elev_m)
    return alt, np.degrees(az) % 360


def star_altaz(name: str, epoch, lat: float, lon: float, elev_m: float = 1600.0):
    """`altaz` for a named catalog star, with its proper motion."""
    return altaz(*STARS[name], epoch, lat, lon, elev_m, pm_mas=PM[name])


def rebuild_catalog(hyg_csv: Path, dest: Path = CATALOG_PATH, mag_limit: float = 4.0) -> int:
    """Regenerate the catalog file from HYG v3's CSV: the same 523 names the solver has always
    used (joined on position, which matches to < 1"), with Hipparcos proper motions added and
    the frame/equinox/epoch header written out."""
    import csv
    old = json.loads(Path(dest).read_text())
    old = old["stars"] if isinstance(old, dict) else old
    rows = [r for r in csv.DictReader(open(hyg_csv)) if r["mag"] and float(r["mag"]) <= mag_limit + 0.3]
    H = np.array([[float(r["ra"]) * 15, float(r["dec"])] for r in rows])
    stars = []
    for c in old:
        d = np.hypot((H[:, 0] - c["ra_deg"]) * np.cos(np.radians(c["dec_deg"])), H[:, 1] - c["dec_deg"]) * 3600
        i = int(d.argmin())
        if d[i] > 1.0:
            raise ValueError(f"{c['name']}: no HYG row within 1 arcsec")
        stars.append({"name": c["name"], "ra_deg": c["ra_deg"], "dec_deg": c["dec_deg"],
                      "pmra_mas": float(rows[i]["pmra"] or 0), "pmdec_mas": float(rows[i]["pmdec"] or 0),
                      "mag": c["mag"], "con": c.get("con")})
    header = {"name": f"HYG v3 mag<={mag_limit:g}",
              "source": "https://github.com/astronexus/HYG-Database (hygdata_v3.csv, kiloquad mirror)",
              "frame": "ICRS", "equinox": 2000.0, "epoch": 2000.0,
              "pm_units": "mas/yr; pmra_mas is mu_alpha* (includes cos dec)",
              "notes": "HYG v3: RA/Dec for epoch and equinox 2000.0, from Hipparcos. See catalog.py."}
    Path(dest).write_text(json.dumps({"catalog": header, "stars": stars}, indent=1) + "\n")
    return len(stars)


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
        alt, az = star_altaz(r["name"], epoch, cam["lat"], cam["lon"], cam.get("elev") or 1600.0)
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
