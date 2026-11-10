"""Article figures drawn from plume-triangulation's data: run with plume's venv, from plume.

    cd plume-triangulation
    FIGLIB_CORPUS=all .venv/bin/python <star-calibration>/examples/hpwren_article/plume_figs.py steele
    FIGLIB_CORPUS=all .venv/bin/python <star-calibration>/examples/hpwren_article/plume_figs.py rainbow

Writes static/*.jpg next to this file, which article.ipynb shows with F.static. They are drawn
on FIgLib frames, which only plume reads, so they are not committed: the notebook carries them
as outputs. The ledger they use is plume's (v0.2.1-era poses).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PLUME = Path.cwd()   # run from plume-triangulation
sys.path.insert(0, str(PLUME))


def steele(fire="20250709_SteeleFire", camera="om-n-mobo-c", offset=300,
           wide=(0, 900, 1980, 500), zoom=(680, 930, 450, 270),
           lines=((10, "#e879f9"), (None, "#ff8c1a"), (25, "#fde047"), (45, "#7dd3fc"))):
    """A young plume whose foot is behind one ridge and in front of the next, with terrain
    silhouettes drawn through the star pose: for each distance X, the top edge of all the
    terrain nearer than X (None in `lines` is the fire's own distance). A line is drawn only
    where it stands clear of the nearer ones, since elsewhere nearer terrain hides it.
    Top: a wide strip of the frame; bottom: the column, as recorded and with the lines.
    `wide` and `zoom` are x, y, w, h in the full frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from src.figlib import ridge_feet as R
    from src.figlib import terrain as T
    row = next(r for r in json.loads((R.OUT / "feet.json").read_text())
               if r["fire"] == fire and r["camera"] == camera)
    s = R.SEQS[row["seq"]]; cam = R.CAMS[camera]
    blob = next(b for _e, o, b in R.frames(row["seq"]) if o == offset)
    img = cv2.cvtColor(R.decode(blob), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    pose = R.pose_for(camera, s["t0"], W)
    shifted = {**cam, "az": cam["az"] + pose["d_az"]}
    kms = [row["km"] if k is None else k for k, _c in lines]
    az, d_m, _h, ang = T._march(shifted, R.DEM, 14.0, 0.02, max(kms) + 1, 30.0)
    drawn, prev = [], None
    for km, (k, colour) in zip(kms, lines):
        el = ang[:, d_m < km * 1000].max(axis=1)
        xs, ys = R.proj(cam, pose, az, el, W, H)
        ok = np.ones(len(az), bool) if prev is None else el > prev + 0.03
        label = f"{km:.0f} km" + (" (the fire)" if k is None else "")
        drawn.append((xs, np.where(ok, ys, np.nan), colour, label))
        prev = el if prev is None else np.maximum(prev, el)

    halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    x0, y0, w, h = wide
    zx, zy, zw, zh = zoom
    fig = plt.figure(figsize=(12, 12 * (h / w) + 12 / 2 * (zh / zw) + 0.15), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[h / w, (zh / zw) / 2], hspace=0.03, wspace=0.015,
                          left=0, right=1, top=1, bottom=0)
    top = fig.add_subplot(gs[0, :]); za = fig.add_subplot(gs[1, 0]); zb = fig.add_subplot(gs[1, 1])

    def show(ax, box, with_lines, lw, labels):
        bx, by, bw, bh = box
        ax.imshow(img, interpolation="lanczos")
        ax.set_xlim(bx, bx + bw); ax.set_ylim(by + bh, by); ax.axis("off")
        if not with_lines:
            return
        for xs, ys, colour, label in drawn:
            ax.plot(xs, ys, color=colour, lw=lw, solid_capstyle="round")
            if labels:
                inb = np.flatnonzero((xs > bx + 15) & (xs < bx + bw) & np.isfinite(ys))
                if inb.size:
                    i = inb[0]
                    ax.text(xs[i], ys[i] - 5, label, color=colour, fontsize=9, ha="left",
                            va="bottom", path_effects=halo)
        ax.plot(row["truth_x"], row["truth_y"], marker="+", ms=10 if labels else 16, mew=1.6,
                color="white", path_effects=halo)

    show(top, wide, True, 1.2, True)
    top.add_patch(plt.Rectangle((zx, zy), zw, zh, fill=False, ec="white", lw=1.0, ls=(0, (3, 2))))
    show(za, zoom, False, 0, False)
    show(zb, zoom, True, 1.8, False)
    dest = HERE / "static" / "between_ridges.jpg"
    fig.savefig(dest, dpi=200, facecolor="white", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(pose["rule"], row["km"], row["occ_km"], row["hidden_deg"], "->", dest)


def rainbow(fire_id="20260722_RainbowFire", names=None, band_px=30.0):
    """Two cameras facing each other: bearings alone, then with the distance the terrain allows
    along each, on a light relief map. The numbers are terrain_range.before_after's."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.figlib import settings
    settings.default_profile("calibrated")
    from src.figlib import terrain_range as TR
    from src.figlib.fig_triangulate import _hillshade
    from src.figlib.geolocate import refine, refine_step
    from src.figlib.geom import angdiff_deg, bearing_deg, haversine_km

    names = names or {"rm-e-mobo-c": "Red Mountain East", "bh-w-mobo-c": "Boucher Hill West"}
    colours = {"rm-e-mobo-c": "#c2410c", "bh-w-mobo-c": "#1d4ed8"}
    INK, MUTED = "#2b2b2b", "#77766f"

    cams, seqs, fires, resolved = TR._load()
    r = next(x for x in resolved if x["fire_id"] == fire_id)
    fire, t = fires[fire_id], r["truth"]
    bs = sorted(TR.bearings_for_fire(fire, seqs, cams, use_wind=False), key=lambda b: -b.conf)
    center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    lats, lons, ll, e0lat, e0lon = TR.solve(bs, center)
    err0 = haversine_km(e0lat, e0lon, t["lat"], t["lon"]); area0 = TR.credible_area_km2(lats, lons, ll)
    terr = [x for x in (TR.bearing_terrain(b, fire, seqs, cams) for b in bs) if x is not None]
    ll1 = ll + TR.terrain_loglik(lats, lons, terr, band_px)
    i, j = np.unravel_index(np.argmax(ll1), ll1.shape)
    e1lat, e1lon = float(lats[i]), float(lons[j])
    if refine_step():
        e1lat, e1lon = refine(bs, e1lat, e1lon, TR.STEP_KM,
                              extra=lambda la, lo: TR.terrain_loglik(la, lo, terr, band_px))
    err1 = haversine_km(e1lat, e1lon, t["lat"], t["lon"]); area1 = TR.credible_area_km2(lats, lons, ll1)
    info = {}
    for bt in terr:
        b, cam = bt["b"], bt["cam"]
        ok = TR.allowed(bt["prof"], bt["y1"], band_px)
        miss = angdiff_deg(bearing_deg(cam["lat"], cam["lon"], t["lat"], t["lon"]), b.bearing_deg)
        along = haversine_km(cam["lat"], cam["lon"], t["lat"], t["lon"]) * math.cos(math.radians(miss))
        info[b.camera] = {"along": along, **TR.summarise_mask(bt["prof"]["d"], ok)}
    print(f"error {err0:.2f} -> {err1:.2f} km, area {area0:.1f} -> {area1:.1f} km2")
    for c, s in info.items():
        print(c, f"{s['lo_km']:.1f}-{s['hi_km']:.1f} km, official {s['along']:.1f}")

    pts = [(b.lat, b.lon) for b in bs] + [(t["lat"], t["lon"]), (e0lat, e0lon), (e1lat, e1lon)]
    la = [q[0] for q in pts]; lo = [q[1] for q in pts]
    clat, clon = (max(la) + min(la)) / 2, (max(lo) + min(lo)) / 2
    cos_lat = math.cos(math.radians(clat))
    half_km = 0.58 * max((max(la) - min(la)) * 111.32, (max(lo) - min(lo)) * 111.32 * cos_lat)
    dlat, dlon = 0.62 * half_km / 111.32, half_km / (111.32 * cos_lat)
    hs, extent = _hillshade(clat, clon, half_km)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9.4), dpi=150)
    fig.patch.set_facecolor("white")

    def draw(ax, surface, est, area, title, with_terrain):
        ax.set_xlim(clon - dlon, clon + dlon); ax.set_ylim(clat - dlat, clat + dlat)
        if hs is not None:
            ax.imshow(hs, origin="lower", extent=extent, cmap="gray", vmin=-0.2, vmax=1.2,
                      alpha=0.75, aspect="auto", zorder=0)
        rel = surface - surface.max()
        a = np.clip((rel + 9.0) / 9.0, 0.0, 1.0)
        rgba = np.zeros(rel.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb("#a21caf")
        rgba[..., 3] = 0.55 * a ** 1.6
        ax.imshow(rgba, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                  aspect="auto", zorder=2, interpolation="bilinear")
        ax.contour(lons, lats, rel, levels=[-3.0], colors=["#701a75"], linewidths=1.2,
                   linestyles="solid", zorder=3)
        for b in bs:
            col = colours.get(b.camera, INK)
            end = TR.along_ray(b.lat, b.lon, b.bearing_deg, half_km * 3)
            s = info.get(b.camera)
            if with_terrain and s and s["lo_km"] is not None:
                p0 = TR.along_ray(b.lat, b.lon, b.bearing_deg, s["lo_km"])
                p1 = TR.along_ray(b.lat, b.lon, b.bearing_deg, s["hi_km"])
                dashed = dict(color=col, lw=1.1, ls=(0, (4, 4)), alpha=0.8, zorder=4)
                ax.plot([b.lon, p0[1]], [b.lat, p0[0]], **dashed)
                ax.plot([p1[1], end[1]], [p1[0], end[0]], **dashed)
                ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color=col, lw=4.5, alpha=0.85, zorder=5,
                        solid_capstyle="butt")
            else:
                ax.plot([b.lon, end[1]], [b.lat, end[0]], color=col, lw=1.6, zorder=4)
            ax.plot(b.lon, b.lat, marker="^", color=col, ms=11, mec="white", mew=1.2, zorder=6)
            west = b.lon < clon
            ax.annotate(names.get(b.camera, b.camera), (b.lon, b.lat), textcoords="offset points",
                        xytext=(0, 12 if west else -20), ha="left" if west else "right", fontsize=10, color=col, weight="bold", zorder=6)
        ax.plot(t["lon"], t["lat"], "o", mfc="none", mec=INK, ms=16, mew=2.0, zorder=8)
        ax.plot(est[1], est[0], "x", color=INK, ms=11, mew=2.6, zorder=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9c7bf")
        ax.set_title(f"{title}: likely area {area:.0f} km\u00b2", color=INK, fontsize=12, loc="left")
        bar_km = 5
        bx0, by = clon + dlon * 0.62, clat - dlat * 0.88
        ax.plot([bx0, bx0 + bar_km / (111.32 * cos_lat)], [by, by], color=INK, lw=3, zorder=9)
        ax.text(bx0, by + dlat * 0.04, f"{bar_km} km", color=INK, fontsize=9)

    draw(axes[0], ll, (e0lat, e0lon), area0, "Bearings alone", False)
    draw(axes[1], ll1, (e1lat, e1lon), area1, "Bearings and terrain", True)
    axes[1].legend(handles=[
        Line2D([], [], color=MUTED, lw=1.6, label="bearing"),
        Line2D([], [], color=MUTED, lw=4.5, label="distance the terrain allows"),
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 4)), label="ruled out by the terrain"),
        Line2D([], [], color="#701a75", lw=1.2, label="95% of the likelihood"),
        Line2D([], [], color=INK, marker="x", ls="none", ms=9, mew=2.4, label="estimate"),
        Line2D([], [], color=INK, marker="o", mfc="none", ls="none", ms=11, mew=2, label="official ignition point"),
    ], loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#c9c7bf",
       framealpha=0.92, labelcolor=INK)
    fig.tight_layout(h_pad=1.5)
    dest = HERE / "static" / "terrain_range.jpg"
    fig.savefig(dest, dpi=240, facecolor="white", pil_kwargs={"quality": 90})
    plt.close(fig)
    print("->", dest)


if __name__ == "__main__":
    {"steele": steele, "rainbow": rainbow}[sys.argv[1]]()
