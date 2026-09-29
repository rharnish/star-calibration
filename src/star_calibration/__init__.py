"""Camera pose and lens calibration from the night sky.

Point sources that drift at the sidereal rate are stars; matching their tracks to a catalog
under one shared pose measures a fixed camera's azimuth, pitch, roll and lens in a single fit,
with no site visit and no surveyed landmark.

  catalog      bright-star catalog (HYG, mag <= 4), and apparent alt/az for any epoch and site
  fisheye      the equidistant-plus-one-term lens model, and pixel <-> direction
  tracks       point-source detection in each frame, linking into moving tracks, cleaning
  pole         the celestial pole from the trails alone, and the pose family it implies
  solve        pose (and lens) from one night's tracks: `solve`, `solve_wide`, `Night`
  cross_night  whether two nights' solves of one camera agree
  overlay      a solve drawn on its own frame, fitted stars against tracks
  animate      a solve played back as video, step by step, from its trace
  explore      the same trace as data for the browser replay (hpwren's explore.html)
  intrinsics   each camera's optical centre, fitted jointly from all its solved nights
  ledger       per-camera, per-date poses, and the rule for when one applies to another date
  sun, moon    their positions, for dark-frame selection and moonlit nights

  hpwren       the library applied to HPWREN's camera network: camera table, CDN nights,
               a command-line calibrator, and the solved pose ledger
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "0.2.1"

_solver_id: str | None = None


def solver_id() -> str:
    """Which solver made a result: the package version, and when it runs from a git checkout
    of this repository, "+g<commit>", with ".dirty" if the package's own files have
    uncommitted changes. An installed copy (e.g. in plume-triangulation's venv) is just the
    version, which its pin already names."""
    global _solver_id
    if _solver_id is None:
        _solver_id = __version__
        here = Path(__file__).resolve().parent
        try:
            def git(*args):
                return subprocess.run(["git", "-C", str(here), *args], capture_output=True,
                                      text=True, timeout=5)
            # only if this file is tracked by the repository around it: a copy installed
            # inside some other project's checkout must not take that project's commit
            if git("ls-files", "--error-unmatch", "__init__.py").returncode == 0:
                sha = git("rev-parse", "--short=10", "HEAD").stdout.strip()
                dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".").stdout.strip()
                if sha:
                    _solver_id = f"{__version__}+g{sha}" + (".dirty" if dirty else "")
        except (OSError, subprocess.SubprocessError):
            pass
    return _solver_id
