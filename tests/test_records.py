"""docs/records.md names every field the records hold.

Each record is produced the way the code produces it -- a solve of a synthetic night, a
ledger built from it, an index of a scratch cache -- and every key in it, nested ones
included, must appear in the page as `code`. A new field fails here until it's documented.
Meaning and units can't be checked this way; this only catches a field nobody wrote down.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from star_calibration import ledger  # noqa: E402
from star_calibration.hpwren import cameras, ledger_path, nights  # noqa: E402
from star_calibration.solve import solve, solve_wide  # noqa: E402
from test_solve import synthetic_night  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "docs" / "records.md"


def documented() -> set[str]:
    text = re.sub(r"```.*?```", "", PAGE.read_text(), flags=re.S)   # fences aren't names
    spans = re.findall(r"`([^`\n]+)`", text)
    return {w for s in spans for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)}


def keys(record, skip: tuple[str, ...] = ()) -> set[str]:
    """Every key in a record, and in the dicts and lists of dicts inside it, except the
    values of `skip` (maps keyed by data, like star name -> index)."""
    out = set()
    if isinstance(record, dict):
        for k, v in record.items():
            out.add(k)
            if k not in skip:
                out |= keys(v, skip)
    elif isinstance(record, list):
        for v in record:
            out |= keys(v, skip)
    return out


def missing(record, skip=()) -> list[str]:
    return sorted(keys(record, skip) - documented())


DATA_KEYED = ("matches", "per_star_px", "mags")


def test_solve_results_are_documented():
    night = synthetic_night("hp-s-mobo-c", (40.0, 1.0, -0.5))
    rows = [solve_wide(night), solve(night, window=(0.0, 3600.0)), solve(night, wide=True)]
    few = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    few.tracks = few.tracks[:5]
    rows.append(solve(few))
    # what calibrate.add and calibrate's no-images path add to a row
    rows.append({"source": "figlib", "t0": 0, "lat": 0.0, "lon": 0.0})
    for r in rows:
        assert not missing(r, DATA_KEYED), missing(r, DATA_KEYED)


def test_ledger_entries_and_lookups_are_documented():
    night = synthetic_night("hp-s-mobo-c", (2.4, -0.7, 0.9))
    r = solve(night)
    assert r["status"] == "solved"
    entries = ledger.build([r], {r["seq"]: night.t0})
    assert not missing(entries)
    assert not missing(ledger.load(ledger_path()))


def test_index_entries_and_camera_records_are_documented(monkeypatch, tmp_path):
    monkeypatch.setenv("HPWREN_CACHE", str(tmp_path))
    for sub in ("20260911_Q1", "20260911_N"):
        d = tmp_path / "nights" / "hp-s-mobo-c" / sub
        d.mkdir(parents=True)
        for i in range(10):
            (d / f"{1789113600 + 60 * i}.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    assert not missing(list(nights.index().values()))
    assert not missing(list(cameras().values()))


def test_replay_data_is_documented():
    from star_calibration import explore

    night = synthetic_night("hp-s-mobo-c", (40.0, 1.0, -0.5))
    trace = []
    r = solve_wide(night, trace=trace)
    d = explore.export(trace, r, night, [(0, "f.jpg")])
    by_star = (*DATA_KEYED, "arcs", "fitted_arcs", "published_arcs", "bright_arcs")
    assert not missing(d, by_star)
    few = synthetic_night("hp-s-mobo-c", (0.0, 0.0, 0.0))
    few.tracks = few.tracks[:5]
    trace = []
    assert not missing(explore.export(trace, solve(few, trace=trace), few, []), by_star)
