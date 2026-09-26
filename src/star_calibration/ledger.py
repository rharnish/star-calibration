"""Per-camera, per-date pose corrections, and the rule for when one applies.

A published HPWREN azimuth is a nameplate number, and cameras get re-aimed: om-s-mobo-c
solved at -10.5 deg off in October 2019 (its color and monochrome units agree) and at
-0.4 deg in June 2024. So a correction is a measurement of one camera on one date, and
applying it to a detection on another date is a claim that the camera didn't move in
between. This module makes that claim explicit, and conservative.

Entries come from star-track solves (src/figlib/stars/solve.py) and live in
data/meta/pose_ledger.json, one per solve:
    {"camera", "epoch", "frame_w", "d_az", "d_pitch", "d_roll", "k_ratio", "k1",
     "n_stars", "median_px", "source", "sky_model"}

`sky_model` is stars.catalog.model_id() at solve time: the catalog and which corrections
(proper motion, precession, refraction) were applied. Poses solved under different models
differ by up to ~0.3 deg for reasons that have nothing to do with the camera, so `load`
refuses a ledger that mixes them. Entries from before the field existed read as
"legacy: J2000, uncorrected".

Lookup for (camera, epoch), first rule that fires:
  1. same-night: solves within SAME_DAYS -> their median d_az;
  2. bracketed: the nearest solve on each side agree within AGREE_DEG -> their mean (the
     camera held still across the gap as far as two measurements can say). If they
     disagree, it moved somewhere in between -> no correction;
  3. one-sided: the nearest solve, if within MAX_DAYS -> it; otherwise no correction.
A solve never carries across a change of frame format: a 2048x1536 unit replaced by a
3072x2048 one under the same camera name is a new installation, whatever cams.json says.
Only d_az reaches a bearing (geom.offset_bearing_deg reads along the horizon row), so it
is the one `corrected_cam` applies; pitch, roll and lens are kept for the record.

Opt in with FIGLIB_POSE_LEDGER=1 (FIGLIB_POSE_LEDGER_PATH to point at another file).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import settings

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "meta" / "pose_ledger.json"
SAME_DAYS = 3.0
AGREE_DEG = 1.0
MAX_DAYS = 365.0
DAY_S = 86400.0

_cache: dict[str, list[dict]] = {}


def enabled() -> bool:
    return settings.flag("FIGLIB_POSE_LEDGER")


def full_enabled() -> bool:
    """FIGLIB_POSE_FULL=1: bearings go through the whole solved camera -- its own lens, pitch
    and roll, read at the box's foot -- instead of the shared lens along the middle row."""
    return settings.flag("FIGLIB_POSE_FULL")


def path() -> Path:
    return Path(settings.get("FIGLIB_POSE_LEDGER_PATH") or LEDGER)


LEGACY_MODEL = "legacy: J2000, uncorrected"


def load(p: Path | None = None) -> list[dict]:
    p = Path(p or path())
    if str(p) not in _cache:
        rows = json.loads(p.read_text()) if p.exists() else []
        models = {r.get("sky_model", LEGACY_MODEL) for r in rows}
        if len(models) > 1:
            raise ValueError(f"{p} mixes sky models {sorted(models)}: re-solve so every entry "
                             "shares one (stars.resolve_ledger)")
        _cache[str(p)] = rows
    return _cache[str(p)]


LENS_KEYS = ("d_pitch", "d_roll", "k_ratio", "k1")


def lookup(entries: list[dict], camera: str, epoch: float,
           frame_w: int | None = None) -> dict | None:
    """{"d_az", "rule", "sources", "d_pitch", "d_roll", "k_ratio", "k1"} for `camera` at
    `epoch`, or None when no rule applies. The rule picks the solves and sets d_az; the rest
    of the solved camera is the mean over those same solves.

    With `frame_w`, only solves recorded in that frame width count.
    """
    hit = _rule(entries, camera, epoch, frame_w)
    if hit is not None:
        src = [e for e in entries if e["source"] in hit["sources"]]
        hit.update({k: statistics.fmean(e[k] for e in src) for k in LENS_KEYS
                    if all(k in e for e in src)})
    return hit


def _rule(entries: list[dict], camera: str, epoch: float, frame_w: int | None) -> dict | None:
    mine = sorted((e for e in entries if e["camera"] == camera
                   and (frame_w is None or e.get("frame_w") == frame_w)),
                  key=lambda e: e["epoch"])
    if not mine:
        return None
    near = [e for e in mine if abs(e["epoch"] - epoch) <= SAME_DAYS * DAY_S]
    if near:
        return {"d_az": statistics.median(e["d_az"] for e in near), "rule": "same-night",
                "sources": [e["source"] for e in near]}
    before = [e for e in mine if e["epoch"] < epoch]
    after = [e for e in mine if e["epoch"] > epoch]
    if before and after:
        b, a = before[-1], after[0]
        if abs(b["d_az"] - a["d_az"]) <= AGREE_DEG:
            return {"d_az": (b["d_az"] + a["d_az"]) / 2, "rule": "bracketed",
                    "sources": [b["source"], a["source"]]}
        return None
    e = before[-1] if before else after[0]
    if abs(e["epoch"] - epoch) <= MAX_DAYS * DAY_S:
        return {"d_az": e["d_az"], "rule": "one-sided", "sources": [e["source"]]}
    return None


def corrected_cam(camera: str, cam: dict, epoch: float,
                  entries: list[dict] | None = None) -> tuple[dict, dict | None]:
    """`cam` with the ledger's d_az folded into its azimuth, and the lookup that did it.

    Unchanged (and None) when the ledger is off or no rule applies, so callers can pass
    every camera through without special cases.
    """
    if not enabled():
        return cam, None
    hit = lookup(load() if entries is None else entries, camera, epoch, cam.get("frame_w"))
    if hit is None:
        return cam, None
    out = {**cam, "az": cam["az"] + hit["d_az"]}
    if full_enabled() and all(k in hit for k in LENS_KEYS):
        # the rest of the solved camera, for geom.offset_bearing_deg to read a pixel through
        out["solved"] = {k: hit[k] for k in LENS_KEYS}
    return out, hit


def build(solve_summary: list[dict], t0_by_seq: dict[str, float],
          dest: Path = LEDGER) -> list[dict]:
    """One ledger entry per solved star-track sequence."""
    out = []
    for r in solve_summary:
        if r.get("status") != "solved":
            continue
        p = r["pose"]
        out.append({"camera": r["camera"], "epoch": int(t0_by_seq[r["seq"]]), "frame_w": r.get("W"),
                    "d_az": round(p["d_az"], 3), "d_pitch": round(p["d_pitch"], 3),
                    "d_roll": round(p["d_roll"], 3), "k_ratio": round(p["k_ratio"], 4),
                    "k1": round(p["k1"], 4), "n_stars": r["n_stars"],
                    "median_px": round(r["median_px"], 2), "source": f"star:{r['seq']}",
                    "sky_model": r.get("sky_model", LEGACY_MODEL)})
    out.sort(key=lambda e: (e["camera"], e["epoch"]))
    Path(dest).write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    sky = ROOT / "out" / "sky" / "data"
    t0 = {s["seq"]: s["t0"] for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
    nights = sky / "hpwren_nights.json"
    if nights.exists():
        t0.update({k: v["t0"] for k, v in json.loads(nights.read_text()).items()})
    rows = build(json.loads((sky / "star_tracks" / "solve_summary.json").read_text()), t0)
    print(f"wrote {LEDGER} ({len(rows)} solves, {len({r['camera'] for r in rows})} cameras)")
