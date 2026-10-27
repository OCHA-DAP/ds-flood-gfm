"""Diagnostic: do the cross-validated problems sit on normally-wet cells? (water vs flood, 2026-10-15)

Question. The products detect water and remove "normal" water in their own ways (GFM its own reference
water; GFDS a long-term mean; VIIRS and FloodScan unconfirmed), while gold counts seasonal water as
flood (D-38). If that mismatch drives the over-confidence and the false events, they should concentrate
on cells that are normally wet in that calendar month.

Rules, fixed before any result was computed:
  - normally wet: the D-38 rule (baseline.normally_wet: a 30 m pixel is wet in month m if JRC GSW monthly
    recurrence >= 50%; a cell is normally wet if >= 30% of its pixels are), for the month of each day;
  - models: the cross-validated ones (make crossval): same folds, rates, prior, adopted dry-window rule;
  - cell-level scores, error coincidence and reliability: gold-map days only, labelled floodable cells;
  - error coincidence: on gold-dry cells both sources observed, lift = P(both false alarm) /
    (P(A false alarm) * P(B false alarm)); 1 = independent;
  - event level: ADR 0007 rules with the event's footprint split by whether a cell is normally wet in the
    month of the first map (like the per-land-type split).
No rule or model changes; nothing is fitted here.

Outputs, results/cv/: normally_wet_skill.csv, normally_wet_coincidence.csv, normally_wet_reliability.csv,
normally_wet_event_level.csv. Needs the cached masks (data/fusion/static/baseline/); missing ones are read
from the JRC bucket (network).

Run: nice -n 19 uv run python scripts/fusion/cv_normally_wet.py [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, add_scores
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.validate import SINGLE

GROUPS = ("normally wet", "other")
PRODUCTS = ["fusion", *SINGLE]


def models_by_fold(cfg, ev: pd.DataFrame) -> dict:
    f = cfg["fusion"]
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == cfg["gold"]["dry_window_rule"]]
    prior = pd.read_csv(cfg.results_dir / "cv" / "prior.csv")
    return {k: engine.Model.from_tables(crossval.fit_table(counts, ev.code[ev.fold != k]), prior, MODEL_SOURCES,
                                        f["decay"], tuple(f["sens_spec_clip"])) for k in sorted(ev.fold.unique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N events only (smoke test)")
    a = ap.parse_args()
    cfg = load_config()
    thr, bins = cfg["fusion"]["prob_threshold"], cfg["validation"]["reliability_bins"]
    e = cfg["event_level"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code").reset_index(drop=True)
    models = models_by_fold(cfg, ev)  # always fitted on every fold event, as in make crossval
    if a.limit:
        ev = ev.head(a.limit)
    skill, coinc, rel, ev_rows = [], [], defaultdict(lambda: np.zeros((bins, 3))), []
    for _, r in ev.iterrows():
        flags_by_group = {g: defaultdict(lambda: None) for g in GROUPS}
        sizes = dict.fromkeys(GROUPS, 0)
        first_month = pd.Timestamp(r.start).month
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            gd = crossval.gold_day_mask(ds)
            t = ds.truth.values
            dom = ds.domain.values.astype(bool)
            p = engine.run_sequence({s: ds[s].values for s in MODEL_SOURCES}, ds.stratum.values, dom, models[r.fold])["p_flood"]
            nw = {m: baseline.normally_wet(cfg, ew.window, m) for m in sorted(set(days[gd].month) | {first_month})}
            obs = {s: ds[s].values for s in SINGLE}
            # 1-4: map days
            for i in np.where(gd)[0]:
                lab = ((t[i] == gold.DRY) | (t[i] == gold.FLOOD)) & np.isfinite(p[i])
                if not lab.any():
                    continue
                w = nw[days[i].month]
                for g, gm in zip(GROUPS, (w, ~w)):
                    m = lab & gm
                    if not m.any():
                        continue
                    tt = t[i][m]
                    base = {"code": r.code, "continent": r.continent, "group": g}
                    skill.append(base | {"product": "fusion", **validate._conf((p[i] >= thr)[m], tt)})
                    for s in SINGLE:
                        skill.append(base | {"product": s, **validate._conf((obs[s][i] == 1)[m], tt)})
                    dry = m & (t[i] == gold.DRY)
                    for sa, sb in itertools.combinations(MODEL_SOURCES, 2):
                        both = dry & (obs[sa][i] != NODATA) & (obs[sb][i] != NODATA)
                        fa, fb = (obs[sa][i] == 1) & both, (obs[sb][i] == 1) & both
                        coinc.append(base | {"pair": f"{sa}+{sb}", "n": int(both.sum()), "fa_a": int(fa.sum()),
                                             "fa_b": int(fb.sum()), "fa_both": int((fa & fb).sum())})
                    pp = p[i][m].astype("float64")
                    b = np.clip((pp * bins).astype(int), 0, bins - 1)
                    acc = rel[g]
                    acc[:, 0] += np.bincount(b, minlength=bins)
                    acc[:, 1] += np.bincount(b, weights=pp, minlength=bins)
                    acc[:, 2] += np.bincount(b, weights=(tt == gold.FLOOD), minlength=bins)
            # 5: event level, footprint split by the first map's month
            fp = (t[gd] == gold.FLOOD).any(axis=0) & dom
            w0 = nw[first_month]
            fl = {"fusion": np.isfinite(p) & (p >= thr)} | {s: obs[s] == 1 for s in SINGLE}
            for g, gm in zip(GROUPS, (w0, ~w0)):
                cells = fp & gm
                sizes[g] += int(cells.sum())
                if not cells.any():
                    continue
                for prod, f in fl.items():
                    s_ = pd.Series(f[:, cells].sum(axis=1), index=days)
                    prev = flags_by_group[g][prod]
                    flags_by_group[g][prod] = s_ if prev is None else prev.add(s_, fill_value=0)
        for g in GROUPS:
            for prod in PRODUCTS:
                s_ = flags_by_group[g][prod]
                if s_ is None:
                    continue
                res_ = crossval.event_level(s_, sizes[g], pd.Timestamp(r.start), pd.Timestamp(r.end), e["min_share"], e["min_cells"],
                                            e["days_around_maps"], tuple(e["false_event_days"]))
                ev_rows.append({"code": r.code, "continent": r.continent, "group": g, "product": prod, "footprint_cells": sizes[g], **res_})

    sk = pd.DataFrame(skill).groupby(["code", "continent", "group", "product"], as_index=False).sum(numeric_only=True)
    t = add_scores(sk.groupby(["group", "product"], as_index=False)[["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum())
    ci = crossval.bootstrap_scores(sk, ["group", "product"], cfg["accuracy"]["bootstrap_reps"], 20261015)
    t = t.merge(ci, on=["group", "product"]).rename(columns={"sens": "recall"})
    tot = t[t["product"] == "fusion"].set_index("group")
    t["share_of_gold_flood"] = t.group.map(tot.truth_flood / tot.truth_flood.sum())
    t["share_of_labelled"] = t.group.map((tot.truth_flood + tot.truth_dry) / (tot.truth_flood + tot.truth_dry).sum())
    t = t[["group", "product", "n_events", "truth_flood", "share_of_gold_flood", "share_of_labelled", "recall", "precision", "f1", "f1_lo", "f1_hi"]]
    t.to_csv(res / "normally_wet_skill.csv", index=False)

    c = pd.DataFrame(coinc).groupby(["group", "pair"], as_index=False)[["n", "fa_a", "fa_b", "fa_both"]].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        c["fa_rate_a"], c["fa_rate_b"] = c.fa_a / c.n, c.fa_b / c.n
        c["lift"] = (c.fa_both / c.n) / (c.fa_rate_a * c.fa_rate_b)
    c["share_of_joint_false_alarms"] = c.fa_both / c.groupby("pair").fa_both.transform("sum")
    c.to_csv(res / "normally_wet_coincidence.csv", index=False)

    rows = []
    for g, acc in rel.items():
        for k in range(bins):
            if acc[k, 0]:
                rows.append({"group": g, "bin": k, "n": int(acc[k, 0]), "mean_p": acc[k, 1] / acc[k, 0], "flood_share": acc[k, 2] / acc[k, 0]})
    rl = pd.DataFrame(rows)
    slopes = validate.summarise_reliability(rl.assign(stratum=rl.group, sum_p=rl.mean_p * rl.n, n_flood=rl.flood_share * rl.n)
                                            [["stratum", "bin", "n", "sum_p", "n_flood"]], bins)
    rl = rl.merge(slopes[["stratum", "reliability_slope_wls"]].rename(columns={"stratum": "group"}), on="group")
    rl.to_csv(res / "normally_wet_reliability.csv", index=False)

    el = pd.DataFrame(ev_rows)
    el.to_csv(res / "normally_wet_event_level.csv", index=False)
    elig = el[el.eligible.astype(bool)]
    es = elig.groupby(["group", "product"]).agg(events=("code", "nunique"), caught=("caught", "mean"),
                                                false_event=("false_event", "mean")).reset_index()

    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False), "\n")
    print(c[["group", "pair", "n", "fa_rate_a", "fa_rate_b", "lift", "share_of_joint_false_alarms"]].round(4).to_string(index=False), "\n")
    print(rl.round(3).to_string(index=False), "\n")
    print(es.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
