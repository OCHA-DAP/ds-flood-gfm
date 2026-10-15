"""Cross-validated re-run (ADRs 0003-0007): for each of five folds, fit on the other four and score it.

Inputs: results/event_folds.csv (scripts/fusion/make_folds.py), the event cubes of every event in it, and
the phase-2 sample pattern counts (prior). A fold event without its cubes raises: no event is dropped.

Two dry-window rules run. The configured one (gold.dry_window_rule, ADR 0006) gives every headline
number. The other is a sensitivity row: product 'fusion_<rule>_dry'.

Outputs, results/cv/:
  rates_by_fold.csv, prior.csv     fitted hit / false-alarm rates (rule x fold x source x land type); base rates
  skill_{overall,by_continent,by_landtype,by_continent_landtype}.csv
                                   recall, precision, F1, CSI per product and mode, 95% intervals (events resampled)
  skill_non_s1.csv                 the same on map days whose gold maps are not from Sentinel-1
  criteria.csv                     (a) coverage, (b) CSI per land type vs best single product, (c) reliability slope
  reliability_bins.csv, reliability_summary.csv, reliability_<land type>.png
  coverage_summary.csv
  event_level_events.csv, event_level_summary.csv
  f1_by_continent.png, f1_by_landtype.png
Cached per-window results: data/fusion/metrics/cv_*.parquet.

Run: nice -n 19 uv run python scripts/fusion/phase5_crossval.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import crossval, cube, engine, gold, plots, table, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.feasibility import s1_days_by_code
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, pooled
from ds_flood_gfm.fusion.validate import SINGLE

log = logging.getLogger("crossval")
RULES = ["corrected", "literal"]
BY = {"overall": [], "by_continent": ["continent"], "by_landtype": ["landtype"], "by_continent_landtype": ["continent", "landtype"]}


def load_events(cfg, folds_path=None) -> pd.DataFrame:
    path = Path(folds_path) if folds_path else cfg.results_dir / "event_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/fusion/make_folds.py once every in-scope event is built")
    folds = pd.read_csv(path)
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code")
    if len(ev) != len(folds):
        raise ValueError(f"{len(folds) - len(ev)} fold events are not in event_table_all.parquet")
    missing = [ew.key for _, r in ev.iterrows() for ew in gold.event_windows(r) if not cube.cube_path(cfg, ew.key).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} event-window cubes missing for fold events (build them first): {missing}")
    return ev


def windows(ev: pd.DataFrame):
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            yield r, ew


def fit_counts_all(cfg, ev: pd.DataFrame) -> pd.DataFrame:
    path = cfg.local("metrics", "cv_fit_counts.parquet")
    if path.exists():
        c = pd.read_parquet(path)
        if set(c.code) == set(ev.code) and set(c.rule) == set(RULES):
            return c
        log.info("cached fit counts cover other events; recomputing")
    out, t0 = [], time.time()
    for n, (r, ew) in enumerate(windows(ev), 1):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        for rule in RULES:
            out.append(crossval.fit_counts(ds, crossval.dry_truth(ds, rule), r.code).assign(rule=rule))
        if n % 25 == 0:
            log.info("fit counts: %d windows, %.0f s", n, time.time() - t0)
    c = pd.concat(out).groupby(["rule", "code", "day_type", "landtype", "source"], as_index=False)[crossval.COUNT4].sum()
    c.to_parquet(path)
    return c


def skill_tables(sk: pd.DataFrame, reps: int) -> dict[str, pd.DataFrame]:
    out = {}
    for name, by in BY.items():
        keys = ["product", "mode", *by]
        t = pooled(sk, keys)
        per_event = sk.groupby([*keys, "code"], as_index=False)[["tp", "fp", "fn"]].sum()
        ci = crossval.bootstrap_scores(per_event, keys, reps, seed=20261014)
        t = t.merge(ci, on=keys, how="left").rename(columns={"sens": "recall"})
        cols = [*keys, "n_events", "truth_flood", "tp", "fp", "fn", "tn", "recall", "recall_lo", "recall_hi",
                "precision", "precision_lo", "precision_hi", "f1", "f1_lo", "f1_hi", "csi"]
        out[name] = t[cols].sort_values([*by, "mode", "f1"], ascending=[*[True] * len(by), True, False])
    return out


def event_level_tables(cfg, ev: pd.DataFrame, flags: pd.DataFrame, sizes: dict, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg["event_level"]
    info = ev.set_index("code")
    rows = []
    for (code, lname, prod), g in flags.groupby(["code", "landtype", "product"]):
        r = crossval.event_level(g.set_index("day").flagged, sizes[(code, lname)], info.loc[code, "start"], info.loc[code, "end"],
                                 e["min_share"], e["min_cells"], e["days_around_maps"], tuple(e["false_event_days"]))
        rows.append({"code": code, "continent": info.loc[code, "continent"], "fold": int(info.loc[code, "fold"]),
                     "landtype": lname, "product": prod, "footprint_cells": sizes[(code, lname)], **r})
    per = pd.DataFrame(rows)
    # an event with no flooded floodable cell has no flag rows at all: list it as not eligible, never drop it
    none = sorted(set(ev.code) - set(per.code))
    if none:
        per = pd.concat([per, pd.DataFrame([{"code": c, "continent": info.loc[c, "continent"], "fold": int(info.loc[c, "fold"]),
                                             "landtype": "all", "product": "fusion", "footprint_cells": 0, "eligible": False}
                                            for c in none])], ignore_index=True)
    rng = np.random.default_rng(7)
    summ = []
    elig = per[per.eligible]
    for grp, keys in (("overall", ["landtype", "product"]), ("by_continent", ["continent", "landtype", "product"])):
        for key, g in elig.groupby(keys):
            c, f = g.caught.astype(float).to_numpy(), g.false_event.astype(float).to_numpy()
            draws = rng.integers(0, len(g), size=(reps, len(g)))
            row = dict(zip(keys, key)) | {"group": grp, "n_events": len(g), "caught": c.mean(), "false_event": f.mean(),
                                          "median_days_to_detection": float(g.days_to_detection.median())}
            for nm, v in (("caught", c), ("false_event", f)):
                b = v[draws].mean(axis=1)
                row[f"{nm}_lo"], row[f"{nm}_hi"] = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
            summ.append(row)
    s = pd.DataFrame(summ)
    s["continent"] = s.get("continent", pd.Series(index=s.index, dtype=object)).fillna("all")
    s = s[["group", "continent", "landtype", "product", "n_events", "caught", "caught_lo", "caught_hi", "false_event",
           "false_event_lo", "false_event_hi", "median_days_to_detection"]]
    n_small = per[(~per.eligible) & (per.landtype == "all")].code.nunique()
    log.info("event level: %d of %d events eligible; %d have a footprint under %d cells (%d of them none at all)",
             elig[elig.landtype == "all"].code.nunique(), len(ev), n_small, e["min_cells"], len(none))
    return per, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", default=None, help="fold file other than results/event_folds.csv (smoke tests)")
    ap.add_argument("--out", default=None, help="output folder other than results/cv (smoke tests)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "phase5_crossval.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    res = Path(a.out) if a.out else cfg.results_dir / "cv"
    res.mkdir(parents=True, exist_ok=True)
    primary = cfg["gold"]["dry_window_rule"]
    if primary not in RULES:
        raise ValueError(f"gold.dry_window_rule {primary!r} is not one of {RULES}")
    fusion_name = {r: "fusion" if r == primary else f"fusion_{r}_dry" for r in RULES}
    f, thr, bins = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["validation"]["reliability_bins"]
    reps = cfg["accuracy"]["bootstrap_reps"]

    ev = load_events(cfg, a.folds)
    folds = sorted(ev.fold.unique())
    log.info("%d events in %d folds; %d windows; headline dry-window rule: %s", len(ev), len(folds),
             sum(1 for _ in windows(ev)), primary)
    counts = fit_counts_all(cfg, ev)

    blocks = pd.read_parquet(cfg.local("sample", "blocks.parquet"))
    in_scope = blocks[blocks.continent.isin(cfg["scope"]["continents"])].block_id
    prior = crossval.prior_table(table.sample_pattern_counts(cfg), in_scope)
    prior.to_csv(res / "prior.csv", index=False)
    log.info("prior from %d in-scope sample blocks:\n%s", len(in_scope), prior.drop_duplicates("landtype").to_string(index=False))

    models, rates = {}, []
    for rule in RULES:
        for k in folds:
            t = crossval.fit_table(counts[counts.rule == rule], ev.code[ev.fold != k])
            rates.append(t.assign(rule=rule, fold=k))
            models[(rule, k)] = engine.Model.from_tables(t, prior, MODEL_SOURCES, f["decay"], tuple(f["sens_spec_clip"]))
    rates = pd.concat(rates, ignore_index=True)
    rates[rates.stratum_code % 2 == 0].drop(columns="stratum_code").to_csv(res / "rates_by_fold.csv", index=False)

    s1 = s1_days_by_code(cfg)
    skill, rel, cov, flags = [], [], [], []
    sizes = defaultdict(int)
    t0 = time.time()
    for n, (r, ew) in enumerate(windows(ev), 1):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        obs = {s: ds[s].values for s in MODEL_SOURCES}
        st, dom = ds.stratum.values, ds.domain.values.astype(bool)
        gd, lt = crossval.gold_day_mask(ds), crossval.landtype_codes(st)
        fl = {}
        for rule in RULES:
            out = engine.run_sequence(obs, st, dom, models[(rule, r.fold)])
            p = out["p_flood"]
            name = fusion_name[rule]
            skill.append(crossval.score_rows(ds, p, r.code, r.continent, s1.get(r.code, set()), thr, name, rule == primary))
            rel.append(crossval.reliability_rows(p, crossval.dry_truth(ds, rule), gd, lt, r.code, bins).assign(rule=rule))
            fl[name] = np.isfinite(p) & (p >= thr)
            if rule == primary:
                cov.append(validate.coverage_rows(ds, out, r.code, r.continent))
        for s in SINGLE:
            fl[s] = ds[s].values == 1
        ff, sz = crossval.footprint_flags(ds, fl)
        flags.append(ff.assign(code=r.code))
        for lname, v in sz.items():
            sizes[(r.code, lname)] += v
        if n % 25 == 0:
            log.info("scored %d windows, %.0f s", n, time.time() - t0)

    sk = pd.concat(skill, ignore_index=True)
    sk.to_parquet(cfg.local("metrics", "cv_skill_events.parquet"))
    tabs = skill_tables(sk, reps)
    for name, t in tabs.items():
        t.to_csv(res / f"skill_{name}.csv", index=False)
    non_s1 = skill_tables(sk[~sk.label_s1], reps)
    non_s1["overall"].to_csv(res / "skill_non_s1.csv", index=False)

    rl = pd.concat(rel, ignore_index=True)
    rl.to_csv(res / "reliability_bins.csv", index=False)
    rs = pd.concat([validate.summarise_reliability(g.rename(columns={"landtype": "stratum"}).drop(columns=["rule", "labels"]), bins)
                    .rename(columns={"stratum": "landtype"}).assign(rule=k[0], labels=k[1])
                    for k, g in rl.groupby(["rule", "labels"])], ignore_index=True)
    rs.to_csv(res / "reliability_summary.csv", index=False)
    for lname in crossval.LANDTYPES.values():
        b = rl[(rl.rule == primary) & (rl.labels == "map days")].rename(columns={"landtype": "stratum"})
        sm = rs[(rs.rule == primary) & (rs.labels == "map days")].rename(columns={"landtype": "stratum"})
        plots.reliability(b, sm, lname, res / f"reliability_{lname}.png")

    cv = pd.concat(cov, ignore_index=True)
    tot = cv[["n_domain", "fusion_today", "fusion_within_3d", *SINGLE]].sum()
    cov_sum = pd.DataFrame({"what": ["fusion_today", "fusion_within_3d", *SINGLE],
                            "coverage": [tot[c] / tot.n_domain for c in ["fusion_today", "fusion_within_3d", *SINGLE]]})
    cov_sum.to_csv(res / "coverage_summary.csv", index=False)

    by_lt = tabs["by_landtype"]
    a = by_lt[by_lt["mode"] == "all"].set_index(["landtype", "product"])
    lo, hi = cfg["validation"]["slope_ok"]
    crit = []
    for lname in crossval.LANDTYPES.values():
        singles = {p: a.loc[(lname, p), "csi"] for p in SINGLE if (lname, p) in a.index}
        best = max(singles, key=singles.get)
        sl = rs[(rs.rule == primary) & (rs.landtype == lname)].set_index("labels").reliability_slope_wls
        crit.append({"landtype": lname, "fusion_csi": a.loc[(lname, "fusion"), "csi"], "best_single": best,
                     "best_single_csi": singles[best], "b_pass": bool(a.loc[(lname, "fusion"), "csi"] >= singles[best]),
                     "slope_map_days": sl.get("map days", np.nan), "slope_map_and_dry_days": sl.get("map and dry days", np.nan),
                     "c_pass_map_days": bool(lo <= sl.get("map days", np.nan) <= hi)})
    crit = pd.DataFrame(crit)
    c = cov_sum.set_index("what").coverage
    crit.attrs["a"] = {"fusion_today": float(c["fusion_today"]), "best_single": c[SINGLE].idxmax(), "best_single_cov": float(c[SINGLE].max())}
    crit.to_csv(res / "criteria.csv", index=False)

    fl_all = pd.concat([x for x in flags if len(x)], ignore_index=True).groupby(["code", "landtype", "product", "day"], as_index=False).flagged.sum()
    per, summ = event_level_tables(cfg, ev, fl_all, sizes, reps)
    per.to_csv(res / "event_level_events.csv", index=False)
    summ.to_csv(res / "event_level_summary.csv", index=False)

    ov = tabs["overall"]
    cont = pd.concat([tabs["by_continent"], ov.assign(continent="All")])
    order = ["All", *[x for x in ["Asia", "Africa", "Oceania", "South America"] if x in set(cont.continent)]]
    plots.f1_by_group(cont[cont["mode"] == "all"], "continent", order, res / "f1_by_continent.png",
                      "F1 per continent, five-fold cross-validation")
    lt_tab = pd.concat([tabs["by_landtype"], ov.assign(landtype="all")])
    plots.f1_by_group(lt_tab[lt_tab["mode"] == "all"], "landtype", ["all", "open", "vegetated", "urban"], res / "f1_by_landtype.png",
                      "F1 per land type, five-fold cross-validation")

    pd.set_option("display.width", 220)
    log.info("overall (all cells):\n%s", ov[ov["mode"] == "all"][["product", "n_events", "recall", "precision", "f1", "f1_lo", "f1_hi", "csi"]].round(3).to_string(index=False))
    log.info("criteria: (a) %s\n%s", crit.attrs["a"], crit.round(3).to_string(index=False))
    log.info("event level (all land):\n%s", summ[(summ.group == "overall") & (summ.landtype == "all")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
