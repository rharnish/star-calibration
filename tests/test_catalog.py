"""The sky model -- precession, proper motion, refraction, and the catalog's declared frame --
checked against first principles and against Skyfield; plus clipping tracks to a time window
and indexing a whole-night directory, which the whole-night window ablation added."""

from __future__ import annotations

import json

import numpy as np
import pytest

from star_calibration import catalog as SG

J2000 = 946728000            # 2000-01-01 12:00 UTC
CENTURY = 36525 * 86400


@pytest.fixture(autouse=True)
def _model_off():
    with SG.using(precession=False, refraction=False):
        yield


def test_precession_moves_the_pole_by_theta():
    # The J2000 pole, carried a century on, sits theta(T=1) = 2004.3109 - 0.42665 - 0.041833"
    # from the new one.
    _, dec = SG.precess(0.0, 90.0, J2000 + CENTURY)
    assert dec == pytest.approx(90 - (2004.3109 - 0.42665 - 0.041833) / 3600, abs=1e-6)


def test_precession_rate_on_the_equinox():
    # A star at (0, 0) gains zeta + z in RA per century to first order: 4612.4" = 1.281 deg.
    ra, dec = SG.precess(0.0, 0.0, J2000 + CENTURY)
    assert ra == pytest.approx(4612.44 / 3600, abs=0.01)
    assert dec == pytest.approx(2004.31 / 3600, abs=0.01)


def test_precession_is_identity_at_j2000():
    ra, dec = SG.precess(123.4, -45.6, J2000)
    assert (ra, dec) == pytest.approx((123.4, -45.6), abs=1e-9)


def test_polaris_2026_distance_from_pole():
    # 0.736 deg from the pole at J2000, closing at ~0.004 deg/yr: ~0.63 deg in mid-2026.
    _, dec = SG.precess(*SG.STARS["Polaris"], J2000 + 26.5 * 365.25 * 86400)
    assert 90 - dec == pytest.approx(0.63, abs=0.01)


def test_refraction_sea_level_values():
    # Saemundsson at 10 C, sea level: ~29' at the horizon, ~9.7' at 5, ~5.4' at 10, ~1' at 45.
    r = SG.refraction_deg([0.0, 5.0, 10.0, 45.0], elev_m=0.0, temp_c=10.0) * 60
    assert r == pytest.approx([29.0, 9.67, 5.41, 1.01], abs=0.05)


def test_refraction_thins_with_altitude_of_site():
    assert SG.refraction_deg(5.0, elev_m=1600.0) / SG.refraction_deg(5.0, elev_m=0.0) == pytest.approx(
        np.exp(-1600 / 8434), rel=1e-9)


def test_altaz_unchanged_with_flags_off():
    args = (*SG.STARS["Vega"], 1784000000, 33.1, -116.6)
    base = SG.altaz(*args)
    SG.set_model(refraction=True)
    lifted = SG.altaz(*args)
    SG.set_model(refraction=False)
    assert SG.altaz(*args) == base
    assert lifted[0] > base[0] and lifted[1] == base[1]


def test_set_model_rejects_unknown_flags():
    with pytest.raises(KeyError):
        SG.set_model(nutation=True)


def test_clip_to_window():
    from star_calibration.solve import clip
    # 20 points, 10 px apart, one per minute: 190 px end to end.
    track = {60 * i: (100.0 + 10 * i, 200.0, 50) for i in range(20)}
    assert clip(track, None) is track
    assert sorted(clip(track, (0, 600))) == [60 * i for i in range(10)]   # 10 pts, 90 px
    assert clip(track, (0, 420)) == {}                                    # 7 points: too few
    slow = {60 * i: (100.0 + 2 * i, 200.0, 50) for i in range(20)}
    assert clip(slow, (0, 1200)) == {}                                    # 38 px: didn't move


def test_index_parses_whole_night_dirs(tmp_path, monkeypatch):
    from star_calibration.hpwren import nights
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    for sub, epochs in (("20260713_N", range(1000, 1020)), ("20260714_Q1", range(2000, 2010))):
        d = tmp_path / "nights/vo-n-mobo-c" / sub
        d.mkdir(parents=True)
        for e in epochs:
            (d / f"{e}.jpg").write_bytes(b"x")
    out = nights.index()
    assert set(out) == {"hpwren_20260713_N_vo-n-mobo-c", "hpwren_20260714_Q1_vo-n-mobo-c"}
    n = out["hpwren_20260713_N_vo-n-mobo-c"]
    assert (n["q"], n["day"], n["n_frames"], n["t0"]) == (None, "20260713", 20, 1010)
    assert out["hpwren_20260714_Q1_vo-n-mobo-c"]["q"] == 1


# Apparent alt/az from Skyfield 1.55 (IAU 2000A precession-nutation, aberration, DE421; no
# refraction) for Volcan Mountain, 2026-07-14 08:10 UTC, computed once from the same catalog
# RA/Dec. The precessed model should agree to the ~20" that nutation and aberration leave;
# the J2000 model is off by up to a third of a degree.
SKYFIELD_VO_20260714_0810 = {
    "Vega": (73.92753768978055, 295.94005694592676),
    "Polaris": (32.9389951063628, 0.711914381438795),
    "Antares": (13.565499899715727, 225.12824882903325),
    "Dubhe": (12.461004730914842, 338.70235272966374),
    "Altair": (65.79830393167984, 180.17930494698533),
    "Arcturus": (15.2531422407819, 283.07444085084296),
    "Capella": (-3.554068738641196, 25.62528548201078),
}


def _sky_err(name):
    alt_ref, az_ref = SKYFIELD_VO_20260714_0810[name]
    alt, az = SG.altaz(*SG.STARS[name], 1784016600, 33.13826, -116.61702)
    return float(alt) - alt_ref, ((float(az) - az_ref + 180) % 360 - 180) * np.cos(np.radians(alt_ref))


@pytest.mark.parametrize("name", sorted(SKYFIELD_VO_20260714_0810))
def test_precessed_positions_match_skyfield(name):
    SG.set_model(precession=True)
    da, dz = _sky_err(name)
    assert abs(da) < 0.01 and abs(dz) < 0.01


def test_j2000_positions_are_off_by_a_third_of_a_degree():
    errs = [np.hypot(*_sky_err(n)) for n in SKYFIELD_VO_20260714_0810]
    assert max(errs) > 0.25


def test_model_id_names_the_catalog_and_its_frame():
    mid = SG.model_id()
    assert SG.META["name"] in mid and "ICRS" in mid and "eq 2000.0" in mid


def test_proper_motion_moves_arcturus():
    # Arcturus: mu_alpha* -1093.45, mu_delta -1999.40 mas/yr -> 26.5 yr carries it ~60"
    ra0, dec0 = SG.STARS["Arcturus"]
    ra, dec = SG.apply_proper_motion(ra0, dec0, *SG.PM["Arcturus"], J2000 + 26.5 * 365.25 * 86400)
    moved = np.hypot((ra - ra0) * np.cos(np.radians(dec0)), dec - dec0) * 3600
    assert moved == pytest.approx(26.5 * np.hypot(1.09345, 1.99940), rel=1e-3)


def test_using_restores_flags():
    before = dict(SG.MODEL)
    with SG.using(proper_motion=False, precession=True):
        assert SG.MODEL["precession"] and not SG.MODEL["proper_motion"]
    assert SG.MODEL == before


def test_catalog_header_is_required(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"catalog": {"name": "x", "frame": "FK4", "equinox": 1950.0, "epoch": 1950.0},
                               "stars": []}))
    with pytest.raises(NotImplementedError):
        SG._load(bad)
    bad.write_text(json.dumps({"catalog": {"name": "x"}, "stars": []}))
    with pytest.raises(ValueError):
        SG._load(bad)
