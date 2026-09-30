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

# 4. Each camera's optical centre from all its solved nights, then solve again through it
python -m star_calibration.hpwren.calibrate intrinsics   # rewrites hpwren/intrinsics.json
python -m star_calibration.hpwren.calibrate solve

# 5. Solved blocks -> a pose ledger (defaults to the cache; name a file to write elsewhere)
python -m star_calibration.hpwren.calibrate ledger
```

A solved block gives its pose relative to the published table. The two cameras above, on
2026-09-11, as they stand in the shipped ledger:

| block | stars | median | d_az | d_pitch | d_roll | k_ratio |
|---|---|---|---|---|---|---|
| hpwren_20260911_Q1_hp-s-mobo-c | 15 | 1.23 px | +1.77° | +0.49° | −0.48° | 0.889 |
| hpwren_20260911_Q1_vo-n-mobo-c | 16 | 0.90 px | +12.07° | +0.47° | −0.39° | 0.891 |

`d_az`, `d_pitch` and `d_roll` are the measured pose as corrections to the published one, in
degrees (`d_az` positive is clockwise). `k_ratio` is the lens scale as a fraction of the
nameplate one.

Each solve is also drawn on its own frame, in the cache's `overlays/`. Matched tracks are
green with the fitted stars in magenta on top of them. The same stars under the published
pose are orange, with a yellow arrow from published to fitted: the correction, drawn. A
failed block shows its tracks in cyan against where the published pose says the bright stars
should be.

To see how a block got there, `calibrate animate <block>` re-solves it and plays the solve
back as video, in the cache's `animations/`: the frames, the pole the trails imply, the
coarse search, each start's refinement tightening onto the tracks, and the acceptance tests
passed or failed. It is most useful on a failure, where it shows the step that gave out.
`--kept` leaves out the attempts `solve_wide` didn't keep. `--full` draws the whole frame, not
just the band of sky, and `--mark` ends on the frame with chosen stars named where the solved
pose puts them, e.g. `--mark Alkaid-Mizar-Alioth-Megrez-Dubhe-Merak-Phad-Megrez,Polaris` for
the Big Dipper and the pole star. It needs ffmpeg on the PATH for
H.264; without it the video is MPEG-4, which VLC plays but a browser won't.

`calibrate explore <block>` does the same in the browser: it re-solves the block and writes
the solve as data, and the gallery's card links a replay page (`gallery/explore.html`). There
you can scrub back and forth through the steps, zoom into the frame, toggle layers, hover a
track to see its star and residual, and click a rung of the tolerance ladder or a point of the
coarse scan to see that pose. Every star position on it is projected in Python by the
solver's own model, so it shows only poses the solver actually tried.

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
reach the gallery and the weather, but not the ledger. [docs/records.md](../../../docs/records.md)
describes every record below, field by field.

```
nights/<cam>/<YYYYMMDD>_Q<n>/<epoch>.jpg    frames (nights.py)
nights.json                                 the index: block name -> camera, t0, directory
tracks/tracks_<block>.pkl                   linked and cleaned tracks per block
solves/solve_<block>.json, summary.json     per-block results
overlays/star_solve_<block>.jpg             each solve drawn on its frame (overlay.py)
animations/star_solve_<block>.mp4           a solve played back, on request (animate.py)
explore/<block>.js                          a solve as data for the replay page, on request (explore.py)
gallery/index.html                          every block on one page (gallery.py)
gallery/explore.html                        the replay page, one block at a time
```

## What ships here

- **`cams.json`** — 505 cameras across 82 sites, derived from HPWREN's own listing,
  [`sites.js`](sites.js), which is kept verbatim for provenance. Per site: latitude,
  longitude, elevation. Per camera: azimuth, horizontal field of view, roll/pitch/yaw, height
  above ground, and imager type (199 colour, 187 monochrome, 109 PTZ, and VNIR, SWIR and
  thermal singles).
  - **What the orientation fields are.** Position is surveyed. `az` gives the direction a
    camera watches, 0/90/180/270 on 482 cameras, and `fov` the lens's rating, 90 or 60 on
    483. `pitch`, `roll` and `yaw` are set on 9, 14 and 3 cameras and `0.0` elsewhere.
  - That describes the view each camera gives, which is what the network was built for. It
    is the reference this package measures against, and the solved poses are corrections
    to it.
- **`pose_ledger.json`** — 137 solves on 76 cameras, from 18 nights between 2026-07-14 and
  2026-09-27.
  - Each entry is one camera-night: the pose offsets, lens, optical centre, star count,
    residual, source block, sky model and solver.
  - It is `calibrate ledger` over this package's own CDN solves, made with v0.3.0. It replaces
    the 106-solve ledger assembled in plume-triangulation (tag `results-2026-09-26`) before
    the split; every camera-night of that one is still in it.
  - [`ledger.py`](../ledger.py) has the rules for when a pose applies to another date.
- **`intrinsics.json`** — each camera's optical centre, fitted from all its solved nights at
  once ([`intrinsics.py`](../intrinsics.py)). The lens's centre sits a median 32 px from the
  frame's middle on these units (47 of 76 cameras fitted), stable night to night. The solver reads it, and each solve
  and ledger entry records the centre it used.

## Data credit

Frames and camera metadata: HPWREN, <https://www.hpwren.ucsd.edu/>. Use of HPWREN data
requires this credit reference.
