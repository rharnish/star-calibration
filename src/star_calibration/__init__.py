"""Camera pose and lens calibration from the night sky.

Point sources that drift at the sidereal rate are stars; matching their tracks to a catalog
under one shared pose measures a fixed camera's azimuth, pitch, roll and lens in a single fit,
with no site visit and no surveyed landmark.

  catalog      bright-star catalog (HYG, mag <= 4), and apparent alt/az for any epoch and site
  fisheye      the equidistant-plus-one-term lens model, and pixel <-> direction
  tracks       point-source detection in each frame, and linking into moving tracks
  pole         the celestial pole from the trails alone, and the pose family it implies
  solve        pose (and lens) from one night's tracks: `solve`, `solve_wide`, `Night`
  cross_night  whether two nights' solves of one camera agree
  overlay      a solve drawn on its own frame, fitted stars against tracks
  ledger       per-camera, per-date poses, and the rule for when one applies to another date
  sun, moon    their positions, for dark-frame selection and moonlit nights

  hpwren       the library applied to HPWREN's camera network: camera table, CDN nights,
               a command-line calibrator, and the solved pose ledger
"""

__version__ = "0.2.1"
