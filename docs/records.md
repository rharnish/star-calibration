# Records

This package and its HPWREN pipeline produce a set of records: the camera table, the pose
ledger, and a local cache of frames, tracks, solves, weather and pictures. This page lists
them, says what produces each one from what, gives every field with its units, and sets out
the rules they obey. The READMEs explain how to run things. This page describes what the runs
leave behind.

The frames are HPWREN's (<https://www.hpwren.ucsd.edu/>), fetched from its public CDN. They
and everything derived from them in the cache stay on the machine that fetched them. None of
it is committed to this repository.

## At a glance

| Record | Where | Written by | Read by | Can it be rebuilt? |
|---|---|---|---|---|
| `cams.json` | package | derived from `sites.js` | everything | yes, from `sites.js` |
| `sites.js` | package | copied verbatim from HPWREN | people deriving `cams.json` | from HPWREN |
| `pose_ledger.json` | package | `calibrate ledger`, copied in on purpose | `ledger.lookup`, plume-triangulation | yes, from `summary.json` |
| `intrinsics.json` | package | `calibrate intrinsics` | the solver, through `hpwren.camera` | yes, from solves and tracks |
| `nights/…/<epoch>.jpg` | cache | `calibrate fetch` / `nights` | tracks, overlays, videos | **only for ~89 days** (see below) |
| `nights.json` | cache | `nights.index` | every step after fetch | yes, from `nights/` |
| `tracks/tracks_<block>.pkl` | cache | `calibrate solve` (first time) | solve, overlay, animate, weather | yes, from frames |
| `solves/solve_<block>.json` | cache | `calibrate solve`, `calibrate add` | `calibrate overlay`, people | yes, from tracks (not FIgLib rows) |
| `solves/summary.json` | cache | `calibrate solve`, `calibrate add` | gallery, weather, ledger, `agree`, `overlay` | yes, from the per-block files |
| `solves/solve_weather.json` | cache | `weather` | gallery | yes, from Open-Meteo |
| `solves/weather_cache.json` | cache | `weather` | `weather` | yes, from Open-Meteo |
| `overlays/star_solve_<block>.jpg` | cache | `calibrate solve`, `overlay`, `add` | gallery | yes, from frames and solve |
| `animations/star_solve_<block>.mp4`, `…_full.mp4` | cache | `calibrate animate`, on request | gallery | yes, from frames and tracks |
| `explore/<block>.js` | cache | `calibrate explore`, on request | `gallery/explore.html` | yes, from frames and tracks |
| `gallery/index.html`, `gallery/thumbs/`, `gallery/explore.html` | cache | `gallery`, `calibrate explore` | people | yes |

The cache is `$HPWREN_CACHE`, `~/.cache/hpwren` by default. plume-triangulation uses the same
one, so both projects read one copy of the frames, and there is one gallery.

## How they depend on each other

```
HPWREN CDN ──fetch──▶ nights/ ──index──▶ nights.json
                         │
                         ├──tracks.collect──▶ tracks/*.pkl ──solve_wide──▶ solves/solve_<block>.json
                         │                                                        │
                         │                        plume (calibrate.add) ─────────▶│
                         │                                                        ▼
                         │                                              solves/summary.json
                         │                                               │      │       │
                         ├──overlay.draw◀─────────────────────────────────┘      │       │
                         │      ▼                                                │       │
                         │  overlays/*.jpg ──────────────────▶ gallery ◀── weather ◀─────┤
                         │                                       ▲                       │
                         ├──animate (re-solves)──▶ animations/ ──┤       ledger.build ◀──┘
                         └──explore (re-solves)──▶ explore/ ─────┘
                                                                              ▼
                                                        cache pose_ledger.json ──(on purpose)──▶ package
```

A change upstream makes everything downstream stale, and nothing records that it is:

- **Changing the detector, linker or cleaning (`tracks.py`)** leaves the old `tracks/*.pkl`
  in place unless it changes a recipe. Each pickle records the recipe `load_tracks` linked
  and cleaned it with (`link`, `clean`), and a pickle made with another recipe is rebuilt.
  A code change that keeps the recipe goes unnoticed: delete `tracks/` to rebuild.
- **Changing the solver** leaves `solves/` as it was until `calibrate solve` runs again.
- **Changing the catalog or its corrections** changes `sky_model`, and `ledger.load` then
  refuses a ledger that mixes the old and new models.

## What can't be rebuilt

- **Frames older than about 89 days.** The CDN keeps recent JPGs public; older ones move to
  Glacier Deep Archive, and only HPWREN staff can restore them. After that, the cache's
  `nights/` may be the only copy of a block's frames anywhere outside HPWREN. Back up
  `nights/` if you need to re-track old blocks.
- **FIgLib rows** (`"source": "figlib"`). Their frames are FIgLib archives that only
  plume-triangulation reads. There are no frames or tracks for them here. Re-solve them in
  plume and publish again with `python -m src.figlib.stars.publish`.

## Conventions

These hold for every record below unless it says otherwise.

| Quantity | Convention |
|---|---|
| Angles | degrees |
| `d_az` | azimuth correction to the published pose, positive clockwise (toward east from north) |
| `d_pitch` | elevation correction, positive up |
| `d_roll` | roll correction, positive turns the image's right edge upward |
| Pixels | full-frame pixels, origin at the top-left corner, x right, y down |
| Frame size | `W` × `H`; 3072 × 2048 for today's CDN frames |
| `k`, `k_ratio` | lens scale: `k` in px per radian; `k_ratio` = `k` / nameplate scale, where the nameplate scale is `(W / 2) / (fov / 2 in radians)` |
| `k1` | radial term of `r = k·θ·(1 + k1·θ²)`, θ in radians |
| `cx`, `cy` | the optical centre: where the boresight lands, in pixels right and down from the frame's middle `(W/2, H/2)` |
| Epochs | Unix seconds, UTC |
| Offsets | seconds from the block's `t0` (the epoch of its median frame) |
| Block names | `hpwren_<YYYYMMDD>_Q<n>_<camera>` (a 3-hour block, Q1 = 00:00–02:59 Pacific), `hpwren_<YYYYMMDD>_N_<camera>` (a whole night), or plume's FIgLib names |

The shared lens is `K_RATIO` 0.886 and `K1` −0.078 (`fisheye.py`). A camera's optical
centre comes from `intrinsics.json`, and is the frame's middle where that has nothing. A
pose is only right together with the centre it was solved under, so every solve and ledger
entry records its `cx`, `cy`.

## Shipped records

These are in the package (`src/star_calibration/hpwren/`), versioned with it, and public.

### `cams.json`

HPWREN's published camera table, one record per camera name (505 as of 2026-09), derived
from `sites.js` and kept beside it verbatim.

| Field | Meaning |
|---|---|
| `site` | site code, the camera name's first part |
| `lat`, `lon` | degrees |
| `elev` | site elevation, metres |
| `az` | published (nameplate) azimuth of the boresight, degrees |
| `fov` | published horizontal field of view, degrees |
| `imager` | `color`, `monochrome`, `ptz`, `experimental`, `VNIR`, `SWIR`, `ir-color`, `ir-thermal` |
| `pitch`, `roll`, `yaw` | degrees; zero placeholders almost everywhere |
| `agl` | camera height above ground, metres |

It is the *published* pose. What a camera actually does is what the solves measure, as
corrections relative to this table.

### `pose_ledger.json`

The solved poses: one entry per solved CDN block, 106 solves of 73 cameras as of
2026-09-26. `calibrate ledger` writes a fresh one to the cache; the package's copy changes
only when someone copies it in on purpose.

| Field | Meaning |
|---|---|
| `camera` | camera name |
| `epoch` | the block's `t0` |
| `frame_w` | frame width the solve was made in |
| `d_az`, `d_pitch`, `d_roll` | the solved pose, as corrections to `cams.json` |
| `k_ratio`, `k1` | the solved lens |
| `cx`, `cy` | the optical centre the solve assumed. Entries without them (all before 2026-09-29) mean (0, 0), and `lookup` averages them that way |
| `n_stars`, `median_px` | how many catalog stars matched, and their median residual |
| `source` | `star:<block>` |
| `sky_model` | the catalog and corrections the solve used (`catalog.model_id()`) |
| `solver` | the solver that made the solve, as in solve results; only on entries built from rows that have it |

Rules (`ledger.py`):

- **A pose is a measurement on one date.** `lookup(camera, epoch)` applies one only if the
  first of these rules fires: *same night* (solves within 3 days: their median), *bracketed*
  (the nearest solve each side agree within 1°: their mean; if they disagree, the camera
  moved in between and no correction applies), or *one-sided* (the nearest solve within
  365 days).
- **A solve never crosses a change of frame format.** A 2048×1536 unit replaced by a
  3072×2048 one under the same name is a new installation.
- **One sky model per ledger.** `load` refuses a file whose entries mix `sky_model`s.
- **Only this package's own CDN solves.** Rows with a `source` (FIgLib) are left out.

`lookup` returns `cx` and `cy` with the rest of the solved camera. A caller that turns
pixels into directions with a ledger pose must use its centre too.

This file is an interface: plume-triangulation reads it through the package, at a pinned
release. Adding a field is safe. Renaming, removing or changing the meaning of one needs a
release and a coordinated change in plume.

### `intrinsics.json`

Each camera's optical centre, fitted from all its solved nights at once
(`intrinsics.py`). A list of segments; a camera's nights in one frame size form one or more
segments, a new one starting where the centre jumps (two consecutive nights each more than
25 px from the segment so far).

| Field | Meaning |
|---|---|
| `camera`, `W`, `H` | the camera and frame size |
| `from`, `to` | the `t0` of the segment's first and last night |
| `cx`, `cy` | the centre the solver uses; (0, 0) when the segment has fewer than 2 nights and fewer than 20 matched stars |
| `k_ratio`, `k1` | the joint fit's lens, for reference: the solver still fits its own |
| `n_nights`, `n_stars` | nights, and matched stars summed over them |
| `median_px`, `median_px_centred` | the joint fit's median residual with the centre held at the middle, and fitted |
| `nights` | the blocks it was fitted from |
| `own` | each night's centre fitted alone, `[cx, cy]`: the scatter the segment hides |

`hpwren.camera(name, W, H, epoch)` gives the solver a camera's record with the centre of
the segment that starts at or before the date (the first segment for earlier dates).
`calibrate intrinsics` rewrites the file from the current solves. Re-solve afterwards.

The first fit (2026-09-29, from 137 solved CDN blocks) gave 47 of 76 cameras a centre, a
median 32 px from the middle (max 75, bm-e-mobo-c). A camera's single-night centres sit a
median 2.5 px from its joint centre (worst 10 px). Solving through the centres dropped the
solves' median residual from 1.23 to 0.78 px, added bm-s 09-11 and (with the level-prior
starts) bi-s 09-20, and moved poses a median 0.76° (max 3.1°, bm-e). Two solves of one
camera within 30 days now agree to 0.015° median and 0.05° worst, against 0.026° and 0.75°
before.

## Cache records

### `nights/` and `nights.json`

Frames live at `nights/<camera>/<YYYYMMDD>_Q<n>/<epoch>.jpg` (a whole night:
`<YYYYMMDD>_N/`), about one a minute. A 320×240 frame is the CDN's "No Images!" placeholder,
served when the camera sent nothing (`nights.PLACEHOLDER_WH`).

`nights.json` indexes them by block name. `nights.index()` rebuilds it from disk, and a
directory with fewer than 8 frames is left out.

| Field | Meaning |
|---|---|
| `seq` | block name |
| `camera` | camera name |
| `day` | `YYYYMMDD`, local date of the block |
| `q` | 1–8 (3-hour block, local time), or `null` for a whole night |
| `t0` | epoch of the median frame; track offsets count from it |
| `lat`, `lon` | the camera's, from `cams.json` |
| `n_frames` | frames on disk |
| `dir` | directory relative to `nights/` |

### `tracks/tracks_<block>.pkl`

A pickle of one block's tracks, written by `calibrate.load_tracks`. Each track is a dict from
offset (seconds) to `(x, y, amp)`: pixel position and peak amplitude of the point in that
frame. The pickle is a cache, not an interface: it can change with `tracks.py`.

| Field | Meaning |
|---|---|
| `linked` | the moving tracks `tracks.collect` linked across frames, before cleaning |
| `link` | the linking recipe: `linker` (`nearest` for colour cameras, `predictive` for monochrome) and `max_gap_s` (a track unseen that long closes) |
| `tracks` | the tracks the solver uses: `linked` after `tracks.clean`. A solve's `matches` index this list |
| `clean` | the cleaning recipe: `banner_px`, the rows of burned-in text at the top of the frame |
| `cleaning` | counts from `tracks.clean`: `linked`, `squiggly` (dropped), `split` (tracks split or trimmed), `banner` (dropped), `duplicate` (dropped), `kept` |
| `WH` | frame width and height |

Pickles written before 2026-09-28 hold only `tracks` (uncleaned, linked with no gap limit
within a block) and `WH`. `load_tracks` rebuilds them.

#### Track cleaning

Nearest-neighbour linking (`tracks.link_tracks`) makes three kinds of tracks that aren't one
star. They were measured on the 3,105 tracks matched to a star in the 130 solved CDN blocks
of 2026-09-28, against each star's path under its solved pose:

- **Hand-overs.** 628 of those tracks leave their star for at least two points. About half
  of the switches come after a gap: a track left open is taken over by a later star passing
  its last position (Markab, then Algenib 70 min later on the same pixels, hp-e 07-14). The
  rest are hops between close stars (the Pleiades), and faint, bloated stars whose centroid
  wanders and fragments. 39% of the off-path points sit on another catalog star.
  `load_tracks` now closes a colour camera's track after 300 s unseen (180 s lost two
  blocks), and `tracks.split` cuts at gaps over 600 s and separates each star's points by
  fitting a cubic path in time (RANSAC, 3 px). On those tracks it kept 98% of each star's
  points and dropped 91% of the off-path points. It left the tracks that stay on their star
  unchanged.
- **Squiggles.** Cloud texture, haze and noise wander: `tracks.squiggly` flags a median turn
  over 40 degrees between steps of at least 10 px, or more than 30% of steps reversing. A
  squiggly track keeps only its largest smooth part, and only if that part holds half its
  points (57% of flagged star tracks qualify, 15% of the others).
- **Banner tracks.** HPWREN burns a line of text into the top ~35 rows. Its clock digits
  change every frame and link into slow tracks, which dragged the pole fit toward |p| 0.5
  on five solved blocks. `clean` drops tracks with a median row under `banner_px` (60) that
  move less than 60 px/h. Real stars cross the banner too, and move far faster.

Over those blocks' old tracks, `clean` kept a track of 8 or more points on 3,065 of the
3,105 matched stars.

### `solves/solve_<block>.json` and `solves/summary.json`

One solve result per block, and `summary.json`, the list of all of them sorted solved-first,
one row per `seq` (a re-solve replaces the row). The fields a row has depend on how far the
solve got.

**Always present**

| Field | Meaning |
|---|---|
| `seq` | block name |
| `camera` | camera name |
| `status` | `solved` or `failed` |
| `reason` | `null` when solved; otherwise why (below) |

**Once tracks exist**

| Field | Meaning |
|---|---|
| `imager` | from `cams.json` |
| `cx`, `cy` | the optical centre the solve assumed (from `intrinsics.json` for this package's blocks) |
| `sky_model` | `catalog.model_id()`. May be missing on older FIgLib failure rows. |
| `solver` | which solver made the row (`star_calibration.solver_id()`): the package version, plus `+g<commit>` when run from a git checkout of this repository, and `.dirty` if the package's files had uncommitted changes. Rows solved before 2026-09-27 don't have it. |
| `W`, `H` | frame size |
| `n_tracks_raw` | tracks the detector produced |
| `n_tracks` | tracks the solve used, after `window` clipping and `prune` |
| `window` | only if the solve was given one: `[start, end)` offsets, seconds |

**Once the coarse search ran**

| Field | Meaning |
|---|---|
| `ref_offset` | the reference frame: the offset where the most tracks were seen |
| `pole` | pole-search attempts only: `norm` (\|p\|, 1 for a coherent sky under the right lens), `inlier_frac`, `median_res_rel` (fit residual as a fraction of the sidereal rate), `p_cam` (the pole in camera coordinates) |
| `published_coincidence` | `inliers` and `predicted`: bright stars that land on a track under the published pose, and how many were predicted in the sky band |
| `coarse_top` | the starts handed to refinement: up to 5 from the coarse score and, in a pole search, up to 2 near where pitch and roll are both small (the level prior): `score`, `inliers`, `predicted`, `pose` (`[d_az, d_pitch, d_roll]`) |

**Once refinement found a fit**, even one that failed the acceptance test:

| Field | Meaning |
|---|---|
| `pose` | `d_az`, `d_pitch`, `d_roll`, `k_ratio`, `k1` |
| `k_ratio_start` | the lens scale the attempt started from |
| `n_stars` | catalog stars matched to tracks |
| `median_px`, `rmse_px` | residual over every matched point, pixels |
| `runs_agreeing` | `"a/b"`: how many of the starts that converged matched the same set of stars |
| `matches` | star name → index into the block's raw track list |
| `per_star_px` | star name → its median residual, pixels |
| `mags` | star name → catalog magnitude |

**Added by `solve_wide`**

| Field | Meaning |
|---|---|
| `found_by` | `grid` (around the published pose) or `pole` (global, from the pole) |
| `lens_from_pole` | the lens scale the trails measure (`pole.lens_scale`), or `null` |

**Added by `calibrate add`**, for results from frames the cache doesn't hold:

| Field | Meaning |
|---|---|
| `source` | the project that filed it, e.g. `figlib` |
| `t0`, `lat`, `lon` | what the index would otherwise supply, for the weather |

`add` refuses a result without `camera`, `t0`, `lat`, `lon` and `ref_offset`. This contract
is an interface: plume-triangulation's `stars.publish` writes to it.

**Acceptance test.** A fit is `solved` when it has at least 8 stars (`min_stars`), a median
under 3.0 px, and a lens scale within 0.70–0.95 (pole search, or a given lens) or 0.85–0.92
(grid). `solve_wide` runs up to three attempts (grid; pole under the shared lens; pole under
the trails' lens) and keeps the best: solved first, then most stars, then lowest median.

**Reasons and outcomes.** The gallery sorts failures by `gallery.outcome`:

| Outcome | Reason starts with | Meaning |
|---|---|---|
| `solved` | — | passed |
| `noimg` | `no images:` | most frames were the CDN placeholder; not solved |
| `dark` | `only N moving tracks` | fewer than 8 usable tracks: cloud, fog, dew, or a dead camera |
| `stars` | `no start converged`, `best: N stars, …`, `no pole fit`, `no coherent sky rotation`, or an exception | the tracks were there, and the fit or the search failed |

### `solves/solve_weather.json` and `solves/weather_cache.json`

`weather` reads Open-Meteo at each solve's reference hour and site.
`solve_weather.json` has one row per solve:

| Field | Meaning |
|---|---|
| `seq` | block name |
| `epoch` | the reference frame's epoch (`t0 + ref_offset`) |
| `lat`, `lon` | the site, rounded to 4 places |
| `forecast` | historical-forecast archive (what the models forecast at the time) |
| `reanalysis` | ERA5, the after-the-fact estimate; no `visibility` |

Each of `forecast` and `reanalysis` maps a variable (`cloud_cover`, `cloud_cover_low`,
`cloud_cover_mid`, `cloud_cover_high`, `visibility`, `relative_humidity_2m`,
`dew_point_2m`, `temperature_2m`, `precipitation`, `wind_speed_10m`) to
`{"at": value at the nearest hour, "profile": hourly values from −6 h to +6 h}`, plus
`hour_utc`. Open-Meteo's units apply: % for cloud and humidity, metres for visibility, °C,
mm, km/h.

`weather_cache.json` keeps Open-Meteo's raw hourly response per
`"<source>|<lat>|<lon>|<UTC day>"`, so a re-run only asks about new site-days.

### `overlays/`, `animations/`, `explore/`, `gallery/`

- **`overlays/star_solve_<block>.jpg`**: the solve drawn on its reference frame
  (`overlay.py`, whose docstring gives the colours). A `noimg` block's overlay is a copy of
  one of its placeholder frames.
- **`animations/star_solve_<block>.mp4`**: the solve played back (`animate.py`), made only
  on request by `calibrate animate`. H.264 when ffmpeg is on the PATH, MPEG-4 otherwise.
  `--full` draws the whole frame instead of the sky band and writes `…_full.mp4`, which the
  gallery doesn't link; `--mark` adds a closing scene naming chosen stars.
- **`explore/<block>.js`**: the solve as data for the browser replay, made only on request by
  `calibrate explore`, which re-solves the block with a trace and doesn't file the result.
  The file sets `window.EXPLORE` (a script, not JSON, so the page works from `file://`,
  where browsers refuse to fetch local files). Its fields are below.
- **`gallery/explore.html`**: the replay page, copied from the package's `hpwren/explore.html`
  by `calibrate explore` and `gallery`. It shows one block, `explore.html?b=<block>`, with
  `&t=<seconds>` to open at a moment. It reads frames from `nights/` where they are, so
  nothing is copied, and a block whose frames have expired can't be replayed.
- **`gallery/index.html`**: every row of `summary.json`, with weather and overlays and, when
  they exist, the video and a link to the replay. It links overlays, videos and replays by
  relative path, so it is not self-contained. There is one gallery; don't make another.

#### The replay's data (`explore.export`)

Pixels are the frame's own (`W` by `H`), rounded to 0.1 px; `null` marks a point off the
sky's side of the lens (more than 100° from the boresight, where the lens polynomial folds
back). Everything projected is projected in Python by the solver's model, so the page only
draws. Internal: the page and the export change together, and `format` says which shape.

| Field | Meaning |
|---|---|
| `format` | the shape's version, 1 |
| `seq`, `camera`, `imager`, `W`, `H`, `t0`, `ref` | the block, its frame size, and the reference offset (s from `t0`) |
| `result` | the re-solve's `status`, `reason`, `pose`, `n_stars`, `median_px`, `rmse_px`, `matches`, `per_star_px`, `mags`, `n_tracks`, `n_tracks_raw`, `solver`, `found_by` (those present) |
| `frames` | `[offset, url]` per frame, in time order; the URL is relative to the page |
| `tracks` | per raw track, as the solver's window clipped it: `[offset, x, y]` per point |
| `used` | the raw indices of the tracks the solver used |
| `kept` | which attempt `solve_wide` returned |
| `attempts` | one per attempt, below |
| `final` | the returned result drawn, below |

Each of `attempts`:

| Field | Meaning |
|---|---|
| `wide`, `k_ratio`, `title`, `kept` | the search, the lens it was given (`null`: the shared lens), a description, and whether it was the one kept |
| `lens`, `lens_ratio` | `[k, k1]` it started from, and `k` over the camera's `initial_k` |
| `ladder` | `[thr, free_lens]` per rung |
| `pole` | pole attempts: `fit` (as the solve's `pole`), `steps` (`[x, y, ux, uy, track]` per velocity sample: midpoint and unit direction), `kept` (a string of `1` kept and `0` trimmed per step, or `null` when it can't be matched to the steps), `xy` (the celestial pole in pixels) |
| `published` | `names` and `xy` of the bright stars (mag ≤ 3.5) in frame at the published pose, under this attempt's lens, at the reference frame |
| `scan` | `label`, `x` (psi, or grid rank), `scores`, `poses`, in the order the page sweeps them; `starts` (indices into those of the poses handed to refinement); `samples` (`i`, `xy`: the bright stars in frame under the pose at index `i`, for ~160 poses and every start) |
| `starts` | `pose`, `score` of each start |
| `rungs` | per rung of every start: `start`, `thr`, `free_lens`, `dropped`, `median_px`, `before`, `after` (the five pose parameters), `pairs` (`[star, raw track]`), `arcs` (per paired star, its path over its track's offsets `before` and `after` the refit, thinned to 24 points) |
| `verdict` | `status`, `reason`, `start`, `tests` (`[text, passed]`) |

`final` holds `fitted_arcs` and `published_arcs` (each matched star's path under the fitted
pose, and under the published pose with the fitted lens), `arrows` (`[star, x0, y0, x1, y1]`
from published to fitted at the reference frame) and, for a failure with no fit,
`bright_arcs` (mag ≤ 3 stars at the published pose and shared lens). `sky` is
`[name, mag, x, y]` for every catalog star in frame at the reference frame under the fitted
pose, or the published one, for labels.

## In memory only

- **`solve.Night`**: one camera's tracks, its `cams.json` record, `t0`, frame size and label.
  Everything a solve reads.
- **The trace**: pass `trace=[]` to `solve` or `solve_wide` to get one dict per step, for
  `animate.py` and `explore.py`. Nothing reads it back, so it never changes a result. It is
  not saved (`explore/` keeps a drawn version of it), and its shape can change freely.

| `stage` | Fields |
|---|---|
| `attempt` | `wide`, `k_ratio`, `tracks` (raw indices used) |
| `setup` | `lens` (`k`, `k1`), `ref`, `coarse_px`, `ladder` |
| `pole` | `fit` (as `pole` above), `inliers` (per velocity sample, in `pole.samples` order) |
| `scan` | `poses`, `scores`, and `psi` for a pole attempt |
| `starts` | `poses`, `scores` |
| `rung` | `start`, `thr`, `free_lens`, `before`, `after` (`null` if the start was dropped), `pairs` (star, raw track index), `median_px` |
| `verdict` | `status`, `reason`, and for a fit `start` and `tests` (each acceptance test, passed or not) |
| `kept` | `attempt`: which attempt `solve_wide` returned (last event) |

## What is an interface

| Stable: change only with a release and plume in step | Internal: change freely |
|---|---|
| `pose_ledger.json` fields (now with `cx`, `cy`) and `ledger.lookup`'s rules | `tracks/*.pkl` |
| `cams.json` fields | the trace |
| the `calibrate add` contract | gallery rows and the pages; the replay's data |
| `pose` and `status` in solve results | overlays and videos |
| `sky_model` strings | `weather_cache.json` |

The other solve-result fields are read by people and by this package's own commands. Keep
them compatible where it's cheap, and list any change in the release notes.

## Known gaps

- **Older results don't say which solver made them.** Rows solved before `solver` was
  added (2026-09-27), including every entry in the shipped `pose_ledger.json`, carry no
  version or commit. A `.dirty` solver means the code differed from the named commit, so
  the row can't be reproduced from the repository alone.
- **Track caches record their recipe, not their code** (see [How they depend on each other](#how-they-depend-on-each-other)).
- **Older FIgLib failure rows lack `sky_model`.**

## Keeping this page right

When a change adds, removes or renames a field in any record above, update this page in the
same commit. `tests/test_records.py` fails when a solve result, summary row, ledger entry,
index entry or camera record has a field this page doesn't mention. It can't check meaning
or units, so read the table you're touching.
