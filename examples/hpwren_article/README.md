# The HPWREN article

[`article.ipynb`](article.ipynb) is the source of *Calibrating HPWREN Cameras Using the
Position of the Stars*, written for HPWREN and reviewed by its team. The prose is in the
markdown cells. Each code cell is a one-line call into [`article_figs.py`](article_figs.py),
and the notebook is committed with its outputs, so GitHub shows the figures.

Cells are tagged:

- `figure`: the output is part of the article and the code is hidden.
- `notes`: the cell checks the numbers quoted in the prose. It is left out of the export.

Unless the article says otherwise, every calibration and number comes from star-calibration
[v0.4.0](https://github.com/rharnish/star-calibration/tree/v0.4.0) and its run on 316 night
blocks.

## Building

```sh
.venv/bin/python examples/hpwren_article/build.py            # export the notebook as it stands
.venv/bin/python examples/hpwren_article/build.py --execute  # re-run every cell first
```

Both write `build/article.html` and `build/article.docx` (not committed). Re-running needs the
HPWREN cache (`$HPWREN_CACHE`, default `~/.cache/hpwren`) with the v0.4.0 solves. Frames the
cache lacks are fetched from HPWREN's CDN, which keeps them for about 90 days.

## Files

| | |
|---|---|
| `article_figs.py` | every figure and number; frames are drawn with OpenCV and plots with matplotlib |
| `build.py` | exports the notebook without code or `notes` cells (nbconvert, then pandoc for .docx) |
| `plume_figs.py` | the two fire figures, drawn from [plume-triangulation](https://github.com/rharnish/plume-triangulation)'s data on FIgLib frames; run it from plume. Its images are output only. |
| `static/cloud.json` | forecast cloud at each of the 316 blocks (Open-Meteo), from `scripts/weather_fill.py`; its keys also pin the network numbers to those blocks |
| `static/split_half.json` | the split-half study of lens and centre against the number of nights, from `scripts/split_half.py` |
| `static/skyline_v040.json` | the network skyline check against the terrain, from `scripts/skyline_v040.py` |
| `static/peaks_wd.json` | peak names and positions from Wikidata |

## Credits

Camera images: [HPWREN](https://www.hpwren.ucsd.edu/), University of California San Diego.
Archived fire sequences: the FIgLib collection, from HPWREN. Terrain: Copernicus DEM GLO-30,
© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under
COPERNICUS by the European Union and ESA. Star positions: the HYG star database.
