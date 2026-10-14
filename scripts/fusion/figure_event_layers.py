"""Figures for the methods docs: how a real event's cells are classified, and how each source sees its water.

Both use cached pipeline layers only (no downloads): the static layers (static.static_for_window),
the GSW monthly "normally wet" share (baseline.normally_wet) and the event cubes.

  docs/figures/strata_example.png         TC20240905VNM window 0 (Typhoon Yagi, Viet Nam, 2024):
                                          HAND, permanent water, WorldCover land type, and the result.
  docs/figures/water_vs_flood_example.png EMSR838, Punjab, Pakistan, 2025-08-28: gold map, normally-wet
                                          water in August, and GFM, VIIRS, FloodScan and GFDS that day.

Run: uv run python scripts/fusion/figure_event_layers.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ds_flood_gfm.fusion import baseline, grid, static  # noqa: E402
from ds_flood_gfm.fusion.config import load_config  # noqa: E402
from ds_flood_gfm.fusion.gold import DRY, FLOOD, NOTRUTH  # noqa: E402
from ds_flood_gfm.fusion.sources import NODATA  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
WATER, LAND, CLOUD, DEEP = "#2a78d6", "#e9e4d8", "#b8b6b0", "#123f80"
OPEN, VEG, URBAN, OUTSIDE = "#e3cf9a", "#4f9a5a", "#8a5ca8", "#d9d7d2"
WET = "#2fa4a0"
OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"


def cube_for(cfg, code: str, key: str) -> xr.Dataset:
    for p in sorted(cfg.local("cube", "events").glob(f"{code}_w*.nc")):
        ds = xr.load_dataset(p, engine="netcdf4")
        w = json.loads(ds.attrs["window"])
        if f"r{w[0]}-{w[1]}_c{w[2]}-{w[3]}" == key:
            return ds
    raise FileNotFoundError(f"no cube of {code} with window {key}")


def frame(ax, ds, title: str) -> tuple:
    lat, lon = ds.lat.values, ds.lon.values
    ext = (lon.min() - 0.005, lon.max() + 0.005, lat.min() - 0.005, lat.max() + 0.005)
    ax.set_facecolor(SURFACE)
    ax.set_aspect(1 / math.cos(math.radians(float(lat.mean()))))
    ax.tick_params(labelsize=7, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color("#c9c7c1")
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    return ext


def categorical(ax, arr, colours: dict[int, str], ext) -> None:
    codes = sorted(colours)
    lut = np.full(256, len(codes), "int16")
    for i, c in enumerate(codes):
        lut[c] = i
    ax.imshow(lut[arr.astype("uint8")], cmap=ListedColormap([colours[c] for c in codes] + [SURFACE]), vmin=0, vmax=len(codes),
              extent=ext, interpolation="nearest")


def strata_figure(cfg) -> None:
    ds = cube_for(cfg, "TC20240905VNM", "r6818-6989_c28476-28708")
    w = json.loads(ds.attrs["window"])
    st = static.static_for_window(cfg, grid.Window(*w))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor=SURFACE)
    ext = frame(axes[0], ds, "1. Height above nearest stream (GLO-30 HAND)\nshare of each cell under 15 m")
    im = axes[0].imshow(st.hand_low_frac.values, cmap="Purples", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)
    frame(axes[1], ds, "2. Permanent water (JRC GSW occurrence)\nshare of each cell water 90%+ of the time")
    im = axes[1].imshow(st.perm_water_frac.values, cmap="Blues", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)
    frame(axes[2], ds, "3. Land type (ESA WorldCover 2021)\nurban if built-up 20%+, else vegetated if trees 30%+")
    categorical(axes[2], st.landcover.values, {0: OPEN, 1: VEG, 2: URBAN}, ext)
    frame(axes[3], ds, "4. Result: domain and stratum\nfloodable (30%+ under 15 m) and not permanent water")
    res = np.where(st.perm_water.values, 4, np.where(~st.floodable.values, 3, st.landcover.values))
    categorical(axes[3], res, {0: OPEN, 1: VEG, 2: URBAN, 3: OUTSIDE, 4: DEEP}, ext)
    dom = st.domain.values
    n = {k: int(((st.landcover.values == k) & dom).sum()) for k in (0, 1, 2)}
    axes[3].legend(handles=[Patch(color=OPEN, label=f"open ({n[0]:,} cells)"), Patch(color=VEG, label=f"vegetated ({n[1]:,})"),
                            Patch(color=URBAN, label=f"urban ({n[2]:,})"), Patch(color=OUTSIDE, label="not floodable"),
                            Patch(color=DEEP, label="permanent water")],
                   loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5, frameon=False)
    fig.suptitle("How each cell is classified before any satellite data is read: Typhoon Yagi, northern Viet Nam, 2024 (one event window)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.02, "Cells are 0.01° (about 1.1 km). Only cells in the domain (panel 4, coloured) are predicted and scored; "
             "the model's hit and false-alarm rates are fitted separately for open, vegetated and urban cells.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT / "strata_example.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"strata_example.png: domain cells {n}")


def water_figure(cfg) -> None:
    key, day = "r5731-5810_c25311-25452", pd.Timestamp("2025-08-28")
    ds = cube_for(cfg, "EMSR838", key)
    i = int(np.where(pd.to_datetime(ds.day.values) == day)[0][0])
    win = grid.Window(*json.loads(ds.attrs["window"]))
    st = static.static_for_window(cfg, win)
    wet = baseline.normally_wet(cfg, win, day.month)
    dom = ds.domain.values.astype(bool)
    t = np.where(dom, ds.truth.values[i], NOTRUTH)
    lab = (t == FLOOD) | (t == DRY)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=SURFACE)
    ax = axes[0, 0]
    ext = frame(ax, ds, "Gold map (Copernicus EMS), 28 Aug 2025")
    categorical(ax, t, {FLOOD: WATER, DRY: LAND, NOTRUTH: SURFACE}, ext)
    on_wet = float((wet[t == FLOOD]).mean())
    ax.text(0.01, 0.01, f"{int((t == FLOOD).sum()):,} flooded cells; {100 * on_wet:.0f}% of them normally wet in August",
            transform=ax.transAxes, fontsize=7.5, color=INK, bbox=dict(fc=SURFACE, ec="none", alpha=0.85))
    ax = axes[0, 1]
    frame(ax, ds, "Normally wet in August (JRC GSW monthly recurrence)\nand permanent water (removed from the domain)")
    categorical(ax, np.where(st.perm_water.values, 2, np.where(wet, 1, 0)), {0: LAND, 1: WET, 2: DEEP}, ext)
    axes[0, 2].axis("off")
    axes[0, 2].legend(handles=[Patch(color=WATER, label="flood"), Patch(color=LAND, label="dry"), Patch(color=CLOUD, label="couldn't see"),
                               Patch(color=SURFACE, ec="#c9c7c1", label="no gold label (not examined)"),
                               Patch(color=WET, label="normally wet this month"), Patch(color=DEEP, label="permanent water")],
                      loc="upper left", fontsize=9, frameon=False, title="legend", title_fontsize=9)
    axes[0, 2].text(0, 0.35, "Scores in each source's panel are for this one day,\non gold-labelled cells, with \"couldn't see\" counted\n"
                    "as a miss (all-cells scoring).", fontsize=8.5, color=INK2, transform=axes[0, 2].transAxes, va="top")
    for ax, name, title in ((axes[1, 0], "gfm", "GFM (Sentinel-1 radar): unusual water only"),
                            (axes[1, 1], "viirs", "VIIRS (optical, 375 m)"),
                            (axes[1, 2], "floodscan", "FloodScan (microwave, 9 km cells)")):
        v = np.where(dom, ds[name].values[i], NODATA)
        frame(ax, ds, title)
        categorical(ax, np.where(dom, v, 254), {FLOOD: WATER, DRY: LAND, NODATA: CLOUD, 254: SURFACE}, ext)
        y, p = t[lab] == FLOOD, v[lab] == FLOOD
        tp = int((p & y).sum())
        rec, prec = tp / max(int(y.sum()), 1), tp / max(int(p.sum()), 1)
        ax.text(0.01, 0.01, f"recall {rec:.2f} · precision {prec:.2f}; saw {100 * float((v[lab] != NODATA).mean()):.0f}% of labelled cells",
                transform=ax.transAxes, fontsize=7.5, color=INK, bbox=dict(fc=SURFACE, ec="none", alpha=0.85))
    fig.suptitle("Water versus flood on one day: Punjab, Pakistan, 28 August 2025 (EMSR838, one window)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "water_vs_flood_example.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"water_vs_flood_example.png: gold flood on normally-wet cells {on_wet:.2f}")


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    strata_figure(cfg)
    water_figure(cfg)


if __name__ == "__main__":
    main()
