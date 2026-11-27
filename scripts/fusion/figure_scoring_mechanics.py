"""Figure for docs/methods.md: how one gold map and one product become hits, misses and false alarms.

Synthetic and illustrative, but every number comes from the pipeline's own functions:
gold.burn_fraction (gold label rule) and sources.modis.tile_counts / values_from_counts (a product at
MODIS resolution). Writes docs/figures/scoring_mechanics.png.

Run: uv run python scripts/fusion/figure_scoring_mechanics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402

from ds_flood_gfm.fusion import gold, grid  # noqa: E402
from ds_flood_gfm.fusion.config import load_config  # noqa: E402
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA, modis  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WATER, LAND, CLOUD = "#2a78d6", "#e9e4d8", "#b8b6b0"
OUT = {"hit (TP)": "#2a78d6", "false alarm (FP)": "#eb6834", "miss (FN)": "#eda100", "correct dry (TN)": "#d6d4ce", "not scored": SURFACE}


def main() -> None:
    cfg = load_config()
    g, s = cfg["gold"], cfg["sources"]["modis"]
    # a 4 x 4 block of 0.01-degree cells inside MODIS tile h18v08 (lon 0..10, lat 10..0)
    x0, y1 = 5.00, 5.04
    win = grid.window_from_bounds(x0, y1 - 0.04, x0 + 0.04, y1)
    # gold map: examined area (all but the left column) and a flood polygon
    valid = shapely.box(x0 + 0.0095, y1 - 0.04, x0 + 0.04, y1)
    flood = shapely.Polygon([(5.006, 5.040), (5.040, 5.040), (5.040, 5.027), (5.024, 5.0215), (5.015, 5.0125), (5.006, 5.016)])
    vf = gold.burn_fraction(valid, win, g["burn_res_deg"])
    ff = gold.burn_fraction(flood, win, g["burn_res_deg"])
    with np.errstate(invalid="ignore", divide="ignore"):
        fov = np.where(vf > 0, np.minimum(ff, vf) / vf, 0.0)
    labelled = vf >= g["valid_min_frac"]
    truth = np.where(labelled, np.where(fov >= g["flood_threshold"], 1, 0), 255)

    # product at MODIS resolution: water where the flood polygon is, but shifted east, plus a cloud bank
    lon = (np.arange(modis.TILE_PX) + 0.5) * modis.PIX
    lat = 10 - (np.arange(modis.TILE_PX) + 0.5) * modis.PIX
    c0, c1 = int((x0 - 0.002) / modis.PIX), int((x0 + 0.042) / modis.PIX)
    r0, r1 = int((10 - y1 - 0.002) / modis.PIX), int((10 - y1 + 0.042) / modis.PIX)
    codes = np.zeros((modis.TILE_PX, modis.TILE_PX), "uint8")
    # the product sees the flood shifted east, misses most of one flooded cell, and is clouded over two places
    seen_water = shapely.affinity.translate(flood, xoff=0.004).difference(shapely.box(5.020, 5.020, 5.030, 5.0285))
    for i in range(r0, r1):
        for j in range(c0, c1):
            if shapely.contains_xy(seen_water, lon[j], lat[i]):
                codes[i, j] = 3
            if (lat[i] < 5.0085 and lon[j] < 5.022) or (5.010 < lat[i] < 5.020 and 5.010 < lon[j] < 5.020):
                codes[i, j] = 255   # cloud
    counts = modis.tile_counts(codes, 18, 8, win)
    val, frac = modis.values_from_counts(counts, "flood", s["observed_min_frac"], s["flood_threshold"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), facecolor=SURFACE)
    ext = (x0, x0 + 0.04, y1 - 0.04, y1)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xlim(ext[0] - 0.001, ext[1] + 0.001)
        ax.set_ylim(ext[2] - 0.001, ext[3] + 0.001)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def cells(ax, lw=1.6):
        for k in range(5):
            ax.plot([x0 + k * 0.01] * 2, [ext[2], ext[3]], color=INK2, lw=lw, zorder=5)
            ax.plot([ext[0], ext[1]], [ext[2] + k * 0.01] * 2, color=INK2, lw=lw, zorder=5)

    # panel 1: gold
    ax = axes[0]
    for k in range(41):  # 0.001-degree sub-pixels
        ax.plot([x0 + k * 0.001] * 2, [ext[2], ext[3]], color=GRID, lw=0.4, zorder=1)
        ax.plot([ext[0], ext[1]], [ext[2] + k * 0.001] * 2, color=GRID, lw=0.4, zorder=1)
    vx, vy = valid.exterior.xy
    ax.fill(vx, vy, color=LAND, zorder=0)
    fx, fy = flood.exterior.xy
    ax.fill(fx, fy, color=WATER, alpha=0.55, zorder=2)
    cells(ax)
    for r in range(4):
        for c in range(4):
            cx, cy = x0 + (c + 0.5) * 0.01, y1 - (r + 0.5) * 0.01
            lab = {1: "FLOOD", 0: "dry", 255: "no label"}[int(truth[r, c])]
            txt = f"{100 * fov[r, c]:.0f}%\n{lab}" if labelled[r, c] else "not\nexamined"
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_title("1. Gold map -> cell label\nflood if >= 30% of the examined part is flooded", loc="left", fontsize=9.5, color=INK)

    # panel 2: product pixels
    ax = axes[1]
    sub = codes[r0:r1, c0:c1]
    rgb = np.zeros((*sub.shape, 3))
    for code, col in ((0, LAND), (3, WATER), (255, CLOUD)):
        rgb[sub == code] = matplotlib.colors.to_rgb(col)
    ax.imshow(rgb, extent=(lon[c0] - modis.PIX / 2, lon[c1 - 1] + modis.PIX / 2, lat[r1 - 1] - modis.PIX / 2, lat[r0] + modis.PIX / 2),
              interpolation="nearest", zorder=1)
    cells(ax)
    for r in range(4):
        for c in range(4):
            cx, cy = x0 + (c + 0.5) * 0.01, y1 - (r + 0.5) * 0.01
            n_all, n_obs = counts["n_all"][r, c], counts["n_obs"][r, c]
            lab = {FLOOD: "FLOOD", DRY: "dry", NODATA: "can't see"}[int(val[r, c])]
            txt = f"seen {100 * n_obs / n_all:.0f}%\n" + (f"water {100 * frac[r, c]:.0f}%\n" if val[r, c] != NODATA else "") + lab
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=7.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_title("2. Product pixels (here ~230 m, 4.8 per cell edge) -> cell value\nseen if >= 50% of pixels observed; flood if >= 30% of seen pixels",
                 loc="left", fontsize=9.5, color=INK)

    # panel 3: outcome
    ax = axes[2]
    tally = {k: 0 for k in OUT}
    for r in range(4):
        for c in range(4):
            t, v = int(truth[r, c]), int(val[r, c])
            if t == 255:
                o = "not scored"
            elif t == 1:
                o = "hit (TP)" if v == FLOOD else "miss (FN)"
            else:
                o = "false alarm (FP)" if v == FLOOD else "correct dry (TN)"
            tally[o] += 1
            note = "\ncouldn't see" if (v == NODATA and t != 255) else ""
            ax.add_patch(plt.Rectangle((x0 + c * 0.01, y1 - (r + 1) * 0.01), 0.01, 0.01, color=OUT[o], zorder=1))
            ax.text(x0 + (c + 0.5) * 0.01, y1 - (r + 0.5) * 0.01, o.replace(" (", "\n(") + note, ha="center", va="center", fontsize=8,
                    color="white" if o in ("hit (TP)", "false alarm (FP)") else INK, zorder=6)
    cells(ax)
    tp, fp, fn = tally["hit (TP)"], tally["false alarm (FP)"], tally["miss (FN)"]
    ax.set_title(f"3. One cell, one day, one count (\"all cells\" scoring)\nhere: recall {tp}/{tp + fn}, precision {tp}/{tp + fp}",
                 loc="left", fontsize=9.5, color=INK)
    fig.suptitle("How a gold map and a product become hits, misses and false alarms (synthetic example; numbers from the pipeline's own functions)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.015, "Couldn't see counts as a miss on gold-flooded cells and as correct dry on gold-dry cells in \"all cells\" scoring; "
             "\"observed cells\" scoring drops those cells. Not-examined cells are never scored.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "scoring_mechanics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}; tally {tally}")


if __name__ == "__main__":
    main()
