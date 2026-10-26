"""Ordinary days (Next steps 6): how often each product, the cascade and fusion call flood when no flood was mapped.

Every precision so far is from map days inside mapped floods, where about 9% of labelled cells are flooded. On
an ordinary day flood is about 60x rarer, so the same false-alarm rate gives far lower precision. Two
measurements, rules fixed before running (2026-10-26):

A. The days before and after each event (30 before the first map, 60 after the last), cross-validated:
   fusion with each fold's model, the cascade, and every product, on labelled dry cells under the adopted
   dry-window rule (cells flooded on a map day get no label). Per product: false-alarm rate = flood calls /
   labelled dry cells it observed ("observed") and / all labelled dry cells ("all"); by land type, by
   continent, by normally-wet status, and by days from the maps (before: -30..-1; after: +1..+30, +31..+60).
   Still near floods, in the same places, and unmapped water counts as dry: an upper bound on false alarms.
B. The global random sample: the 30 blocks in the four continents with no mapped event, every day of
   2023-2025. No gold, so GFM is the referee where it observed (GFM precision against gold is 0.86). Fusion
   uses rates fitted on all 170 events (the sample is never fitted on). Per product: flood-call rate on
   floodable cell-days (all, and where the product observed); against GFM where both observed: precision
   and recall relative to GFM; and the share of block-days on which at least 10% of floodable cells are
   flagged (a block-level "event", the deferred false-event check away from floods, ADR 0007).
C. Implied ordinary-day precision, stated as a formula, not a measurement: map-day recall x base rate /
   (map-day recall x base rate + false-alarm rate x (1 - base rate)), with A's false-alarm rate and the
   sample base rate (prior.csv), per land type.

D. (added 2026-10-26 after A and B, before running it; Next steps 12) --microwave floodscan: the same A and B
   with FloodScan alone in fusion's microwave slot (and in the cascade), rates fitted the same way, to test
   whether GFDS is what makes fusion noisy on ordinary days. Outputs get the suffix _floodscan_only.

Outputs, results/cv/: ordinary_days_dry_windows.csv, ordinary_days_sample.csv, ordinary_days_sample_blocks.csv,
ordinary_days_implied_precision.csv.
Run: nice -n 19 uv run python -u scripts/fusion/cv_ordinary_days.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.validate import SINGLE

CASCADE = "cascade: VIIRS, else GFM, else microwave"
PRODUCTS = ["fusion", CASCADE, *SINGLE]


MW = "microwave"


def cascade(v):
    out = np.full(v["viirs"].shape, NODATA, "uint8")
    for s in [MW, "gfm", "viirs"]:
        out = np.where(v[s] != NODATA, v[s], out)
    return out


def calls_and_seen(name, v, p, thr):
    if name == "fusion":
        return np.isfinite(p) & (p >= thr), np.isfinite(p)
    if name == CASCADE:
        c = cascade(v)
        return c == 1, c != NODATA
    return v[name] == 1, v[name] != NODATA


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--microwave", default="microwave", choices=["microwave", "floodscan"], help="what fills fusion's (and the cascade's) microwave slot")
    a = ap.parse_args()
    mw = a.microwave
    sfx = "" if mw == "microwave" else f"_{mw}_only"
    srcs = ["gfm", mw, "viirs"]
    global MW
    MW = mw
    cfg = load_config()
    f, thr = cfg["fusion"], cfg["fusion"]["prob_threshold"]
    rule = cfg["gold"]["dry_window_rule"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code").reset_index(drop=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule]
    prior = pd.read_csv(res / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    if mw != "microwave":
        comp = pd.read_parquet(cfg.local("metrics", "cv_fit_counts_components.parquet"))
        counts = pd.concat([counts.drop(columns="rule"), comp[comp.source == mw]], ignore_index=True)
    models = {k: engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k], sources=srcs), prior, srcs, f["decay"], clip)
              for k in sorted(ev.fold.unique())}
    model_all = engine.Model.from_tables(crossval.fit_table(counts, ev.code, sources=srcs), prior, srcs, f["decay"], clip)
    t0 = time.time()

    # A. dry windows
    acc = defaultdict(lambda: np.zeros(3))  # (product, landtype, continent, wet, period) -> calls_obs, dry_obs, dry_all ... see below
    rows = []
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            truth = crossval.dry_truth(ds, rule)
            gd = crossval.gold_day_mask(ds)
            dom = ds.domain.values.astype(bool)
            lt = crossval.landtype_codes(ds.stratum.values)
            v = {s: ds[s].values for s in SINGLE}
            p = engine.run_sequence({s: v[s] for s in srcs}, ds.stratum.values, dom, models[r.fold])["p_flood"]
            start, end = pd.Timestamp(ds.attrs["event_start"]), pd.Timestamp(ds.attrs["event_end"])
            nw_cache = {}
            for i in np.where(~gd)[0]:
                dry = (truth[i] == gold.DRY) & dom & (lt[i] != NODATA)
                if not dry.any():
                    continue
                d = days[i]
                period = "before (-30..-1)" if d < start else ("after (+1..+30)" if (d - end).days <= 30 else "after (+31..+60)")
                m = d.month
                if m not in nw_cache:
                    nw_cache[m] = baseline.normally_wet(cfg, ew.window, m)
                nw = nw_cache[m]
                for name in PRODUCTS:
                    call, seen = calls_and_seen(name, {k: x[i] for k, x in v.items()}, p[i], thr)
                    for k, ltn in crossval.LANDTYPES.items():
                        for wet, wm in (("normally wet", nw), ("other", ~nw)):
                            mm = dry & (lt[i] == k) & wm
                            if not mm.any():
                                continue
                            rows.append({"code": r.code, "continent": r.continent, "product": name, "landtype": ltn, "wet": wet, "period": period,
                                         "calls": int((call & mm).sum()), "dry_observed": int((seen & mm).sum()), "dry_all": int(mm.sum())})
        print(f"A: {r.code} done, {time.time() - t0:.0f}s", flush=True)
    A = pd.DataFrame(rows)
    A.to_parquet(cfg.local("metrics", f"ordinary_days_dry_rows{sfx}.parquet"))

    def summarise(df, by):
        g = df.groupby(["product", *by], as_index=False)[["calls", "dry_observed", "dry_all"]].sum()
        g["false_alarm_rate_observed"] = g.calls / g.dry_observed
        g["false_alarm_rate_all"] = g.calls / g.dry_all
        g["view"] = ",".join(by) if by else "overall"
        return g

    out = pd.concat([summarise(A, []), summarise(A, ["landtype"]), summarise(A, ["continent"]), summarise(A, ["wet"]), summarise(A, ["period"]),
                     summarise(A, ["landtype", "wet"])], ignore_index=True)
    out.to_csv(res / f"ordinary_days_dry_windows{sfx}.csv", index=False)

    # B. the global random sample
    blocks = pd.read_parquet(cfg.local("sample", "blocks.parquet"))
    blocks = blocks[blocks.continent.isin(cfg["scope"]["continents"])]
    srows, brows = [], []
    for _, b in blocks.iterrows():
        ds = xr.load_dataset(cfg.local("cube", "sample", f"{b.block_id}.nc"), engine="netcdf4")
        dom = ds.domain.values.astype(bool)
        if not dom.any():
            continue
        lt = crossval.landtype_codes(ds.stratum.values)
        v = {s: ds[s].values for s in SINGLE}
        p = engine.run_sequence({s: v[s] for s in srcs}, ds.stratum.values, dom, model_all)["p_flood"]
        gfm_call, gfm_seen = v["gfm"] == 1, v["gfm"] != NODATA
        nday = p.shape[0]
        for name in PRODUCTS:
            call, seen = calls_and_seen(name, v, p, thr)
            for k, ltn in crossval.LANDTYPES.items():
                m = dom[None] & (lt == k)
                if not m.any():
                    continue
                both = m & seen & gfm_seen
                srows.append({"block_id": b.block_id, "continent": b.continent, "product": name, "landtype": ltn,
                              "cell_days": int(m.sum()), "observed": int((m & seen).sum()), "calls": int((call & m).sum()),
                              "both_seen": int(both.sum()), "gfm_flood_both": int((both & gfm_call).sum()),
                              "calls_both": int((both & call).sum()), "agree_flood": int((both & call & gfm_call).sum())})
            # block-level "event": >= 10% of floodable cells flagged that day (and >= 5 cells)
            n_dom = int(dom.sum())
            flagged = call[:, dom].sum(axis=1)
            fired = (flagged >= max(5, int(np.ceil(0.1 * n_dom)))).sum() if n_dom >= 5 else np.nan
            brows.append({"block_id": b.block_id, "continent": b.continent, "kind": b.kind, "product": name, "floodable_cells": n_dom,
                          "days": nday, "days_fired": int(fired) if n_dom >= 5 else np.nan})
        print(f"B: {b.block_id} done, {time.time() - t0:.0f}s", flush=True)
    S = pd.DataFrame(srows)
    S.to_parquet(cfg.local("metrics", f"ordinary_days_sample_rows{sfx}.parquet"))

    def ssum(df, by):
        g = df.groupby(["product", *by], as_index=False)[["cell_days", "observed", "calls", "both_seen", "gfm_flood_both", "calls_both", "agree_flood"]].sum()
        g["call_rate_all"] = g.calls / g.cell_days
        g["call_rate_observed"] = g.calls / g.observed
        g["precision_vs_gfm"] = g.agree_flood / g.calls_both
        g["recall_vs_gfm"] = g.agree_flood / g.gfm_flood_both
        g["gfm_flood_rate_where_both_seen"] = g.gfm_flood_both / g.both_seen
        g["view"] = ",".join(by) if by else "overall"
        return g

    sout = pd.concat([ssum(S, []), ssum(S, ["landtype"]), ssum(S, ["continent"])], ignore_index=True)
    sout.to_csv(res / f"ordinary_days_sample{sfx}.csv", index=False)
    B = pd.DataFrame(brows)
    bsum = B.groupby("product", as_index=False).agg(blocks=("block_id", "nunique"), block_days=("days", "sum"), days_fired=("days_fired", "sum"))
    bsum["share_of_block_days_fired"] = bsum.days_fired / bsum.block_days
    pd.concat([B, bsum.assign(block_id="ALL")], ignore_index=True).to_csv(res / f"ordinary_days_sample_blocks{sfx}.csv", index=False)

    # C. implied precision
    sk = pd.read_csv(res / "skill_by_landtype.csv")
    sk = sk[sk["mode"] == "all"]
    fa = out[(out.view == "landtype")].set_index(["product", "landtype"]).false_alarm_rate_all
    pr = prior.drop_duplicates("landtype").set_index("landtype").base_rate
    crow = []
    for _, s in sk.iterrows():
        if (s["product"], s.landtype) not in fa.index:
            continue
        rec, far, base = s.recall, fa[(s["product"], s.landtype)], pr[s.landtype]
        crow.append({"product": s["product"], "landtype": s.landtype, "map_day_recall": rec, "map_day_precision": s.precision,
                     "dry_window_false_alarm_rate_all": far, "base_rate": base,
                     "implied_ordinary_day_precision": rec * base / (rec * base + far * (1 - base)) if far > 0 else np.nan})
    C = pd.DataFrame(crow)
    C.to_csv(res / f"ordinary_days_implied_precision{sfx}.csv", index=False)

    pd.set_option("display.width", 220)
    print(out[out.view == "overall"][["product", "calls", "dry_observed", "dry_all", "false_alarm_rate_observed", "false_alarm_rate_all"]].round(4).to_string(index=False), "\n")
    print(out[out.view == "period"].pivot(index="product", columns="period", values="false_alarm_rate_all").round(4).to_string(), "\n")
    print(sout[sout.view == "overall"][["product", "cell_days", "observed", "call_rate_all", "call_rate_observed", "precision_vs_gfm", "recall_vs_gfm", "gfm_flood_rate_where_both_seen"]].round(4).to_string(index=False), "\n")
    print(bsum.round(4).to_string(index=False), "\n")
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
