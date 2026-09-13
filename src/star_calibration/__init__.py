"""Camera pose and lens calibration from the night sky.

Point sources that drift at the sidereal rate are stars; matching their tracks to a catalog
under one shared pose measures a camera's azimuth, pitch, roll and lens in a single fit.
See NOTES.md (2026-09-13) for how this was built and what it established.

  sun        sun position (el, az) for an epoch and site
  catalog    bright-star catalog (HYG, mag <= 4) and alt/az for any epoch
  fisheye    equidistant fisheye projection, the model these lenses follow
  tracks     point-source detection and linking into moving tracks, per sequence
  solve      label-free pose search and star<->track fit with the lens held fixed
  nights     moonless-night frame blocks from HPWREN's public CDN
  run_nights track and solve every fetched night block, then rebuild the pose ledger
"""
