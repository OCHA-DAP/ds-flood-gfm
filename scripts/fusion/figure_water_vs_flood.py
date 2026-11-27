"""Water versus flood, drawn for a reader new to it: one event window on one map day (cross-validated fusion).

Step 1, `--count`: for every event window and map day, how fusion's calls (P >= 0.5) fall against gold and
the JRC normally-wet mask (D-38 rule, that day's month). Writes data/fusion/metrics/water_vs_flood_cells.parquet
and prints the totals: of fusion's flood calls on gold-dry cells (scored as false alarms), how many sit on
normally-wet cells (water the analyst left out of the flood, or water that is not there).
Step 2, `--draw CODE DAY`: the four-panel map for one window-day, docs/figures/water_vs_flood_cells.png.

Run: nice -n 19 uv run python scripts/fusion/figure_water_vs_flood.py --count
     nice -n 19 uv run python scripts/fusion/figure_water_vs_flood.py --draw FL20170815BGD 2017-08-15
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES

CATS = ["hit", "flood call on dry, normally wet", "flood call on dry, not normally wet", "miss"]


def setup(cfg):
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code").reset_index(drop=True)
    f = cfg["fusion"]
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == cfg["gold"]["dry_window_rule"]]
    prior = pd.read_csv(cfg.results_dir / "cv" / "prior.csv")
    models = {k: engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k]), prior, MODEL_SOURCES,
                                          f["decay"], tuple(f["sens_spec_clip"])) for k in sorted(ev.fold.unique())}
    return ev, models


def window_day_arrays(cfg, ds, model, i):
    """Per-cell arrays for map day i: truth, P, normally-wet mask, domain."""
    days = pd.to_datetime(ds.day.values)
    p = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, ds.domain.values.astype(bool), model)["p_flood"]
    win = grid.Window(*json.loads(ds.attrs["window"]))
    return ds.truth.values[i], p[i], baseline.normally_wet(cfg, win, days[i].month), ds.domain.values.astype(bool), p


def count(cfg) -> None:
    ev, models = setup(cfg)
    thr = cfg["fusion"]["prob_threshold"]
    rows = []
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            gd = crossval.gold_day_mask(ds)
            p = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, ds.domain.values.astype(bool),
                                    models[r.fold])["p_flood"]
            for i in np.where(gd)[0]:
                t = ds.truth.values[i]
                lab = ((t == gold.DRY) | (t == gold.FLOOD)) & np.isfinite(p[i])
                if not lab.any():
                    continue
                nw = baseline.normally_wet(cfg, ew.window, days[i].month)
                call = p[i] >= thr
                fl, dr = lab & (t == gold.FLOOD), lab & (t == gold.DRY)
                rows.append({"code": r.code, "window": ew.key, "day": days[i], "cells": int(lab.size), "labelled": int(lab.sum()),
                             "nw_labelled": int((lab & nw).sum()), "nw_flood": int((fl & nw).sum()), "nw_dry": int((dr & nw).sum()),
                             "hit_nw": int((fl & call & nw).sum()), "hit_other": int((fl & call & ~nw).sum()),
                             "fa_nw": int((dr & call & nw).sum()), "fa_other": int((dr & call & ~nw).sum()),
                             "miss_nw": int((fl & ~call & nw).sum()), "miss_other": int((fl & ~call & ~nw).sum())})
    df = pd.DataFrame(rows)
    df.to_parquet(cfg.local("metrics", "water_vs_flood_cells.parquet"))
    s = df.sum(numeric_only=True)
    print(f"normally-wet labelled cells: {s.nw_labelled:,}; gold flood {s.nw_flood:,} ({s.nw_flood / s.nw_labelled:.0%}), dry {s.nw_dry:,}")
    print(f"fusion flood calls on gold-dry cells: {s.fa_nw + s.fa_other:,}; on normally-wet cells {s.fa_nw:,} "
          f"({s.fa_nw / (s.fa_nw + s.fa_other):.0%}); on other cells {s.fa_other:,}")
    print(f"fusion hits: normally wet {s.hit_nw:,}, other {s.hit_other:,}; misses: normally wet {s.miss_nw:,}, other {s.miss_other:,}")
    print(f"precision as scored {(s.hit_nw + s.hit_other) / (s.hit_nw + s.hit_other + s.fa_nw + s.fa_other):.3f}; "
          f"if flood calls on normally-wet gold-dry cells were not counted against it {(s.hit_nw + s.hit_other) / (s.hit_nw + s.hit_other + s.fa_other):.3f}")
    # candidates for the figure: mid-sized windows with all four categories well represented
    c = df[(df.cells.between(8000, 90000)) & (df.hit_nw + df.hit_other >= 300) & (df.fa_nw >= 150) & (df.fa_other >= 100) & (df.nw_flood >= 150)]
    c = c.assign(score=np.minimum(c.fa_nw, c.fa_other) + np.minimum(c.hit_nw, c.hit_other)).sort_values("score", ascending=False)
    print(c.head(12)[["code", "window", "day", "cells", "labelled", "hit_nw", "hit_other", "fa_nw", "fa_other", "miss_nw", "miss_other"]].to_string(index=False))


def draw(cfg, code: str, day: str, window_key: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from ds_flood_gfm.fusion.plots import INK, INK2, SURFACE

    ev, models = setup(cfg)
    r = ev[ev.code == code].iloc[0]
    ews = [ew for ew in gold.event_windows(r) if window_key in (None, ew.key)]
    d = pd.Timestamp(day)
    for ew in ews:
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        days = pd.to_datetime(ds.day.values)
        if d in set(days):
            break
    i = int(np.where(days == d)[0][0])
    t, p, nw, dom, _ = window_day_arrays(cfg, ds, models[r.fold], i)
    thr = cfg["fusion"]["prob_threshold"]
    lab = ((t == gold.DRY) | (t == gold.FLOOD)) & np.isfinite(p)
    call = np.where(np.isfinite(p), p >= thr, False)
    fl, dr = lab & (t == gold.FLOOD), lab & (t == gold.DRY)
    win = grid.Window(*json.loads(ds.attrs["window"]))
    minx, miny, maxx, maxy = win.bounds
    ext = [minx, maxx, miny, maxy]
    grey, blue, sand, white = "#d9d8d4", "#2a78d6", "#e9dcc0", SURFACE
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0), facecolor=SURFACE)
    def show(ax, img, colors, title, legend):
        ax.imshow(img, cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1, extent=ext, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.legend(handles=[Patch(facecolor=c, edgecolor=INK2, linewidth=0.4, label=lbl) for c, lbl in legend], loc="lower left",
                  fontsize=7.5, frameon=True, framealpha=0.9)
        ax.tick_params(labelsize=7, colors=INK2)
    # A: normally wet this month (outside the domain: grey)
    a = np.where(~dom, 0, np.where(nw, 2, 1))
    show(axes[0, 0], a, [grey, white, blue], f"A. Normally wet in {d:%B} (JRC, 1984-2021)",
         [(blue, "normally wet this month"), (white, "normally dry"), (grey, "not floodable / permanent water")])
    # B: gold labels
    b = np.where(~dom, 0, np.where(fl, 3, np.where(dr, 2, 1)))
    show(axes[0, 1], b, [grey, white, sand, blue], f"B. Gold map, {d:%d %B %Y}",
         [(blue, "flooded"), (sand, "dry"), (white, "not examined (no label)"), (grey, "not floodable")])
    # C: fusion's calls
    c = np.where(~dom, 0, np.where(call, 2, 1))
    show(axes[1, 0], c, [grey, white, blue], "C. Fusion says flood (P ≥ 0.5)",
         [(blue, "flood"), (white, "no flood"), (grey, "not floodable")])
    # D: how it is scored
    green, orange, red, lgrey = "#1baf7a", "#eda100", "#d6455a", "#b8b7b2"
    dd = np.ones(t.shape, int)  # white: correct dry, or not examined
    dd[fl & call] = 2
    dd[dr & call & nw] = 3
    dd[dr & call & ~nw] = 4
    dd[fl & ~call] = 5
    dd[~dom] = 0
    n = {k: int(v.sum()) for k, v in {"hit": fl & call, "nw": dr & call & nw, "other": dr & call & ~nw, "miss": fl & ~call}.items()}
    show(axes[1, 1], dd, [grey, white, green, orange, red, lgrey], "D. How fusion is scored against gold",
         [(green, f"hit: flood on a gold-flooded cell ({n['hit']:,})"),
          (orange, f"flood on gold-dry, normally-wet cell ({n['nw']:,})"),
          (red, f"flood on gold-dry, normally-dry cell ({n['other']:,})"),
          (lgrey, f"miss: gold flooded, fusion no ({n['miss']:,})"),
          (white, "correct dry, or not examined")])
    fig.suptitle(f"{code}, {d:%d %B %Y}: some of fusion's false alarms are normal water the analyst left out (orange);\n"
                 "most here are not (red: the square blocks are the microwave products' 9 km cells)", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = cfg.local_root.parent.parent / "docs" / "figures" / "water_vs_flood_cells.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--draw", nargs=2, metavar=("CODE", "DAY"))
    ap.add_argument("--window", default=None, help="window key when an event has several")
    a = ap.parse_args()
    cfg = load_config()
    if a.count:
        count(cfg)
    if a.draw:
        draw(cfg, *a.draw, a.window)


if __name__ == "__main__":
    main()
