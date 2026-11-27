"""Experiment (D-38): does masking "normally wet" cells cut the coinciding false alarms and raise
precision without losing recall?

Baseline (fusion/baseline.py): per event window and calendar month, the cells that JRC Global
Surface Water monthly recurrence marks as normally wet. Masking: a source's flood call in a
normally-wet cell becomes NoData, because normal water is not evidence of flood. Dry calls are
left unchanged. The rule and both thresholds were fixed before this was run (configs/fusion.yaml
`baseline`). Nothing is fitted, so training and held-out events are reported separately.

On the selected events' cubes:
  1. gold: how many gold-flood and gold-dry cells on gold days are normally wet. Masking can cost
     recall only where gold maps flood on normally-wet cells.
  2. per source, recall and precision on gold days, over all cells and over observed cells,
     before vs after.
  3. on gold-dry cells (gold days and dry windows), how often two sources' false alarms coincide
     (lift, as phase3_agreement.py), before vs after.

Outputs (results/): experiment_seasonal_baseline_{split}.csv (scores),
experiment_seasonal_baseline_gold_{split}.csv, experiment_seasonal_baseline_dependence_{split}.csv.

Run: nice -n 19 uv run python scripts/fusion/experiment_seasonal_baseline.py [--split train|holdout] [--plan] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import accuracy, agreement, baseline, cube, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("seasonal_baseline")
SOURCES = ["gfm", "microwave", "viirs", "floodscan", "gfds"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "holdout"], default="train")
    ap.add_argument("--plan", action="store_true", help="count window-months to read, compute nothing")
    ap.add_argument("--limit", type=int, default=None, help="first N events only (smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "experiment_seasonal_baseline.log")), logging.StreamHandler(sys.stdout)])
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    split = pd.read_csv(cfg.results_dir / "event_split.csv").set_index("code").split
    sel = sel[sel.code.map(split) == a.split]
    if a.limit:
        sel = sel.head(a.limit)
    windows = [(r.code, ew) for _, r in sel.iterrows() for ew in gold.event_windows(r)]
    n_wm = 0
    for _, ew in windows:
        ds = xr.open_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        n_wm += len(set(pd.to_datetime(ds.day.values).month))
        ds.close()
    log.info("%s: %d events, %d windows, %d window-months of recurrence to read (cached after the first run)",
             a.split, sel.code.nunique(), len(windows), n_wm)
    if a.plan:
        return

    scores, gold_rows, pats = [], [], []
    for k, (code, ew) in enumerate(windows, 1):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        days = pd.to_datetime(ds.day.values)
        wet_by_month = {m: baseline.normally_wet(cfg, ew.window, m) for m in sorted(set(days.month))}
        wet = np.stack([wet_by_month[d.month] for d in days])
        t, st = ds.truth.values, ds.stratum.values
        lab = ((t == 0) | (t == 1)) & ds.domain.values.astype(bool)[None]
        gd = ((days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"])))[:, None, None]
        obs = {s: ds[s].values for s in SOURCES}
        masked = {s: np.where((v == 1) & wet, NODATA, v).astype(v.dtype) for s, v in obs.items()}
        for code_st, st_name in STRATUM_NAMES.items():
            m = lab & (st == code_st)
            if not m.any():
                continue
            g = m & gd
            gold_rows.append({"code": code, "window": ew.key, "stratum": st_name,
                              "gold_flood": int((g & (t == 1)).sum()), "gold_flood_normally_wet": int((g & (t == 1) & wet).sum()),
                              "gold_dry": int((g & (t == 0)).sum()), "gold_dry_normally_wet": int((g & (t == 0) & wet).sum())})
            for variant, data in (("original", obs), ("masked", masked)):
                for s in SOURCES:
                    v = data[s]
                    for mode, mm in (("all", g), ("conditional", g & (v != NODATA))):
                        p, y = v[mm] == 1, t[mm] == 1
                        scores.append({"code": code, "stratum": st_name, "source": s, "variant": variant, "mode": mode,
                                       "tp": int((p & y).sum()), "fp": int((p & ~y).sum()), "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())})
                pc = accuracy.pattern_counts({s: data[s] for s in MODEL_SOURCES}, MODEL_SOURCES, m, t)
                pc["stratum"], pc["code"], pc["variant"] = st_name, code, variant
                pats.append(pc)
        if k % 20 == 0 or k == len(windows):
            log.info("window %d/%d done", k, len(windows))

    res = cfg.results_dir
    sc = pd.DataFrame(scores).groupby(["code", "stratum", "source", "variant", "mode"], as_index=False).sum()
    sc.to_csv(res / f"experiment_seasonal_baseline_{a.split}.csv", index=False)
    gr = pd.DataFrame(gold_rows).groupby(["code", "stratum"], as_index=False).sum(numeric_only=True)
    gr.to_csv(res / f"experiment_seasonal_baseline_gold_{a.split}.csv", index=False)
    pat = pd.concat(pats, ignore_index=True)
    dep = pd.concat([agreement.error_dependence(pat[pat.variant == v].drop(columns="variant")).assign(variant=v)
                     for v in ("original", "masked")], ignore_index=True)
    dep.to_csv(res / f"experiment_seasonal_baseline_dependence_{a.split}.csv", index=False)

    pd.set_option("display.width", 220)
    tot = gr[["gold_flood", "gold_flood_normally_wet", "gold_dry", "gold_dry_normally_wet"]].sum()
    log.info("gold on gold days: %d flood cells, %d (%.1f%%) normally wet; %d dry cells, %d (%.1f%%) normally wet",
             tot.gold_flood, tot.gold_flood_normally_wet, 100 * tot.gold_flood_normally_wet / max(tot.gold_flood, 1),
             tot.gold_dry, tot.gold_dry_normally_wet, 100 * tot.gold_dry_normally_wet / max(tot.gold_dry, 1))
    s = sc.groupby(["source", "mode", "variant"])[["tp", "fp", "fn"]].sum()
    s["recall"], s["precision"] = s.tp / (s.tp + s.fn), s.tp / (s.tp + s.fp)
    log.info("recall and precision on gold days, all strata:\n%s", s.round(3).to_string())
    d = dep[dep.truth == "dry"].pivot_table(index=["stratum", "pair"], columns="variant", values=["n_both_wrong", "lift", "lift_within_event"])
    log.info("coinciding false alarms on gold-dry cells:\n%s", d.round(2).to_string())


if __name__ == "__main__":
    main()
