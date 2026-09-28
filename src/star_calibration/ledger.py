"""Per-camera, per-date pose corrections, and the rule for when one applies.

A published HPWREN azimuth is a nameplate number, and cameras get re-aimed: om-s-mobo-c
solved at -10.5 deg off in October 2019 (its color and monochrome units agree) and at
-0.4 deg in June 2024. So a correction is a measurement of one camera on one date, and
applying it to a detection on another date is a claim that the camera didn't move in
between. This module makes that claim explicit, and conservative.

Entries come from star-track solves (solve.py), one per solve (`build`):
    {"camera", "epoch", "frame_w", "d_az", "d_pitch", "d_roll", "k_ratio", "k1",
     "n_stars", "median_px", "source", "sky_model"}

`sky_model` is catalog.model_id() at solve time: the catalog and which corrections
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
3072x2048 one under the same camera name is a new installation, whatever the camera table
says. `lookup` returns the rule's d_az and the mean pitch, roll and lens of the same solves;
which of them a caller applies is the caller's choice.

HPWREN's ledger ships as `hpwren/pose_ledger.json` (`hpwren.ledger_path()`).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

SAME_DAYS = 3.0
AGREE_DEG = 1.0
MAX_DAYS = 365.0
DAY_S = 86400.0

_cache: dict[str, list[dict]] = {}


LEGACY_MODEL = "legacy: J2000, uncorrected"


def load(p: Path | str) -> list[dict]:
    """A ledger file's entries, refused if they mix sky models. Cached per path."""
    p = Path(p)
    if str(p) not in _cache:
        rows = json.loads(p.read_text()) if p.exists() else []
        models = {r.get("sky_model", LEGACY_MODEL) for r in rows}
        if len(models) > 1:
            raise ValueError(f"{p} mixes sky models {sorted(models)}: re-solve so every entry "
                             "shares one")
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


def build(solve_summary: list[dict], t0_by_seq: dict[str, float],
          dest: Path | str | None = None) -> list[dict]:
    """One ledger entry per solved result in `solve_summary`, dated by `t0_by_seq[r["seq"]]`;
    written to `dest` when given."""
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
                    "sky_model": r.get("sky_model", LEGACY_MODEL)}
                   | ({"solver": r["solver"]} if r.get("solver") else {}))
    out.sort(key=lambda e: (e["camera"], e["epoch"]))
    if dest is not None:
        Path(dest).write_text(json.dumps(out, indent=1) + "\n")
    return out
