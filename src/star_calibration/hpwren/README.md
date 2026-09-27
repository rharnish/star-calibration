# Calibrating HPWREN's cameras

This folder applies the library to [HPWREN](https://www.hpwren.ucsd.edu/)'s camera network.
It carries the published camera table, fetches night frames from HPWREN's public CDN, and
solves each camera's pose into a ledger. It uses only the library's public API, so it is also
a worked example for any other fixed-camera network.

## Calibrate a camera

```sh
pip install -e ".[hpwren,opencv]"   # from the repository root; drop opencv if you have a cv2 build

# 1. Frames: the first 90 minutes after local midnight (Q1), about one frame a minute
python -m star_calibration.hpwren.calibrate fetch 20260911 hp-s-mobo-c vo-n-mobo-c

# 2. Tracks, a pose and an overlay for every fetched block (cached; a re-run skips what's done)
python -m star_calibration.hpwren.calibrate solve

# 3. Is it right? A second night of the same camera should land on the same pose
python -m star_calibration.hpwren.calibrate fetch 20260912 hp-s-mobo-c vo-n-mobo-c
python -m star_calibration.hpwren.calibrate solve
python -m star_calibration.hpwren.calibrate agree

# 4. Solved blocks -> a pose ledger (defaults to the cache; name a file to write elsewhere)
python -m star_calibration.hpwren.calibrate ledger
```

A solved block gives its pose relative to the published table. The two cameras above, on
2026-09-11, as they stand in the shipped ledger:

| block | stars | median | d_az | d_pitch | d_roll | k_ratio |
|---|---|---|---|---|---|---|
| hpwren_20260911_Q1_hp-s-mobo-c | 14 | 1.69 px | +0.79° | −0.07° | −0.45° | 0.888 |
| hpwren_20260911_Q1_vo-n-mobo-c | 9 | 1.09 px | **+11.31°** | +1.46° | −0.38° | 0.892 |

`d_az` is how far the camera actually points from its published azimuth, in degrees
(positive is clockwise). vo-n-mobo-c points 11° east of its nameplate north. `k_ratio` is the
lens scale as a fraction of the nameplate one.

Each solve is also drawn on its own frame, in the cache's `overlays/`. Matched tracks are
green with the fitted stars in magenta on top of them. The same stars under the published
pose are orange, with a yellow arrow from published to fitted, so a camera that points 11° off
its nameplate shows eleven degrees of arrow. A failed block shows its tracks in cyan against
where the published pose says the bright stars should be.

**Choosing nights.** The CDN serves roughly the last 89 days; older frames are in Glacier
Deep Archive and need a staff restore. Clear nights work, and so do moonlit ones. A night
that fails is usually cloud or fog: `weather` and `gallery` show which.

```sh
python -m star_calibration.hpwren.weather    # Open-Meteo cloud cover at each solve
python -m star_calibration.hpwren.gallery    # one page: solved beside failed, with weather
```

## Where things live

Frames and intermediate results go in a cache outside any repository: `$HPWREN_CACHE`, or
`~/.cache/hpwren`. Every project that works with these frames can share one copy.
A project that solves frames the cache doesn't hold can file its results there too, with
`calibrate.add` (plume-triangulation files its FIgLib archive solves this way). Those results
reach the gallery and the weather, but not the ledger.

```
nights/<cam>/<YYYYMMDD>_Q<n>/<epoch>.jpg    frames (nights.py)
nights.json                                 the index: block name -> camera, t0, directory
tracks/tracks_<block>.pkl                   moving tracks per block
solves/solve_<block>.json, summary.json     per-block results
overlays/star_solve_<block>.jpg             each solve drawn on its frame (overlay.py)
gallery/index.html                          every block on one page (gallery.py)
```

## What ships here

- **`cams.json`** — 505 cameras across 82 sites, derived from HPWREN's own listing,
  [`sites.js`](sites.js), which is kept verbatim for provenance. Per site: latitude,
  longitude, elevation. Per camera: azimuth, horizontal field of view, roll/pitch/yaw, height
  above ground, and imager type (199 colour, 187 monochrome, 109 PTZ, and VNIR, SWIR and
  thermal singles).
  - **Read the orientation fields carefully.** Only position is a survey. `az` is exactly
    0/90/180/270 on 482 cameras, and `fov` exactly 90 or 60 on 483. `pitch`, `roll` and
    `yaw` are non-zero on only 9, 14 and 3 cameras; everywhere else they are `0.0`
    placeholders.
  - That is a cardinal heading and a spec sheet, not a calibration. It is entirely adequate
    for what the network was built for, which is giving people pictures. It is what this
    package measures against.
- **`pose_ledger.json`** — 106 solves on 73 cameras, from ten nights between 2026-07-14 and
  2026-09-25.
  - Each entry is one camera-night: the pose offsets, lens, star count, residual, source
    block and sky model.
  - It was assembled in plume-triangulation (tag `results-2026-09-26`, from its recent-corpus
    ledger) before the split. New solves from `calibrate ledger` extend it.
  - [`ledger.py`](../ledger.py) has the rules for when a pose applies to another date.

## Data credit

Frames and camera metadata: HPWREN, <https://www.hpwren.ucsd.edu/>. Use of HPWREN data
requires this credit reference.
