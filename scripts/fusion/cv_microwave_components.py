"""Ablation: one microwave product at a time in fusion's microwave slot (user request, 2026-10-16).

Fusion as built uses the microwave vote (FloodScan OR GFDS). This re-runs the cross-validation with the
slot filled by GFDS alone, then FloodScan alone, and with no microwave input at all, for reference.

Rules, fixed before running:
  - the same five folds, base rate (prior), dry-window rule (gold.dry_window_rule) and threshold;
  - each variant's sources get hit and false-alarm rates fitted on the other four folds, per land type,
    exactly as in make crossval (GFM and VIIRS rates are therefore identical across variants);
  - scored on gold-map days, all cells; F1 overall, per continent, per land type and by VIIRS period,
    with 95% intervals from resampling events and a paired interval for each variant's difference from
    fusion as built (the same resampled events for both);
  - event level: ADR 0007 rules on each event's whole footprint; reliability slope on map days.

Outputs, results/cv/: microwave_components_skill.csv, microwave_components_diff.csv,
microwave_components_event_level.csv, microwave_components_reliability.csv.
Cached: data/fusion/metrics/cv_fit_counts_components.parquet (FloodScan and GFDS fit counts).

Run: nice -n 19 uv run python scripts/fusion/cv_microwave_components.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import crossval, cube, engine, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import pooled

VARIANTS = {
    "fusion (microwave vote)": ["gfm", "microwave", "viirs"],
    "fusion with GFDS only": ["gfm", "gfds", "viirs"],
    "fusion with FloodScan only": ["gfm", "floodscan", "viirs"],
    "fusion without microwave": ["gfm", "viirs"],
}
BASE = "fusion (microwave vote)"
GAP = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-07-31"))


def windows(ev):
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            yield r, ew


def component_counts(cfg, ev, rule) -> pd.DataFrame:
    path = cfg.local("metrics", "cv_fit_counts_components.parquet")
    if path.exists():
        c = pd.read_parquet(path)
        if set(c.code) == set(ev.code):
            return c
    out = []
    for r, ew in windows(ev):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        out.append(crossval.fit_counts(ds, crossval.dry_truth(ds, rule), r.code, sources=["floodscan", "gfds"]))
    c = pd.concat(out).groupby(["code", "day_type", "landtype", "source"], as_index=False)[crossval.COUNT4].sum()
    c.to_parquet(path)
    return c


def paired_diff(sk: pd.DataFrame, by: list[str], reps: int) -> pd.DataFrame:
    """F1 of each variant minus fusion as built, with a paired event bootstrap within each group."""
    rng = np.random.default_rng(20261016)
    rows = []
    groups = sk.groupby(by) if by else [((), sk)]
    for key, g in groups:
        e = g.groupby(["code", "product"])[["tp", "fp", "fn"]].sum().unstack("product").fillna(0)
        f1 = lambda m: 2 * m[..., 0] / (2 * m[..., 0] + m[..., 1] + m[..., 2])  # noqa: E731
        base = e.xs(BASE, axis=1, level=1).to_numpy(float)
        idx = rng.integers(0, len(base), (reps, len(base)))
        for v in VARIANTS:
            if v == BASE:
                continue
            m = e.xs(v, axis=1, level=1).to_numpy(float)
            d = f1(m[idx].sum(1)) - f1(base[idx].sum(1))
            row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            rows.append(row | {"variant": v, "f1_minus_built": float(f1(m.sum(0)) - f1(base.sum(0))),
                               "diff_lo": float(np.quantile(d, 0.025)), "diff_hi": float(np.quantile(d, 0.975)),
                               "n_events": len(base)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    f, thr, bins = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["validation"]["reliability_bins"]
    e, reps, rule = cfg["event_level"], cfg["accuracy"]["bootstrap_reps"], cfg["gold"]["dry_window_rule"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold"]], on="code").reset_index(drop=True)
    inside = (ev.start >= GAP[0]) & (ev.end <= GAP[1])
    ev["viirs_period"] = np.where(inside, "VIIRS gap", "VIIRS available")

    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = pd.concat([counts[counts.rule == rule].drop(columns="rule"), component_counts(cfg, ev, rule)], ignore_index=True)
    prior = pd.read_csv(res / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    models, rates = {}, []
    for k in sorted(ev.fold.unique()):
        train = ev.code[ev.fold != k]
        t = crossval.fit_table(counts, train, sources=["gfm", "microwave", "viirs", "floodscan", "gfds"])
        rates.append(t.assign(fold=k))
        for v, srcs in VARIANTS.items():
            models[(k, v)] = engine.Model.from_tables(t, prior, srcs, f["decay"], clip)

    skill, flags, sizes = [], [], defaultdict(int)
    rel = {v: np.zeros((bins, 3)) for v in VARIANTS}
    for r, ew in windows(ev):
        ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
        st, dom = ds.stratum.values, ds.domain.values.astype(bool)
        gd, t = crossval.gold_day_mask(ds), ds.truth.values
        fl = {}
        for v in VARIANTS:
            m = models[(r.fold, v)]
            p = engine.run_sequence({s: ds[s].values for s in m.sources}, st, dom, m)["p_flood"]
            sr = crossval.score_rows(ds, p, r.code, r.continent, set(), thr, v, include_single=False)
            if len(sr):  # windows with no labelled floodable cell return an empty table
                skill.append(sr[sr["mode"] == "all"].assign(viirs_period=r.viirs_period))
            fl[v] = np.isfinite(p) & (p >= thr)
            lab = ((t == gold.DRY) | (t == gold.FLOOD)) & np.isfinite(p) & gd[:, None, None]
            pp = p[lab].astype("float64")
            b = np.clip((pp * bins).astype(int), 0, bins - 1)
            rel[v][:, 0] += np.bincount(b, minlength=bins)
            rel[v][:, 1] += np.bincount(b, weights=pp, minlength=bins)
            rel[v][:, 2] += np.bincount(b, weights=(t[lab] == gold.FLOOD), minlength=bins)
        ff, sz = crossval.footprint_flags(ds, fl)
        flags.append(ff[ff.landtype == "all"].assign(code=r.code))
        sizes[r.code] += sz.get("all", 0)

    sk = pd.concat(skill, ignore_index=True)
    tabs = []
    for by in ([], ["continent"], ["landtype"], ["viirs_period"]):
        keys = ["product", *by]
        tt = pooled(sk, keys).rename(columns={"sens": "recall"})
        ci = crossval.bootstrap_scores(sk.groupby([*keys, "code"], as_index=False)[["tp", "fp", "fn"]].sum(), keys, reps, 20261016)
        tt = tt.merge(ci, on=keys)
        tt["group"] = by[0] if by else "overall"
        tt["value"] = tt[by[0]] if by else "all"
        tabs.append(tt[["group", "value", "product", "n_events", "truth_flood", "recall", "precision", "f1", "f1_lo", "f1_hi"]])
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(res / "microwave_components_skill.csv", index=False)
    diffs = pd.concat([paired_diff(sk, by, reps).assign(group=by[0] if by else "overall")
                       for by in ([], ["continent"], ["landtype"], ["viirs_period"])], ignore_index=True)
    diffs.to_csv(res / "microwave_components_diff.csv", index=False)

    fa = pd.concat(flags, ignore_index=True).groupby(["code", "product", "day"], as_index=False).flagged.sum()
    info = ev.set_index("code")
    erows = []
    for (code, v), g in fa.groupby(["code", "product"]):
        out = crossval.event_level(g.set_index("day").flagged, sizes[code], info.loc[code, "start"], info.loc[code, "end"],
                                   e["min_share"], e["min_cells"], e["days_around_maps"], tuple(e["false_event_days"]))
        erows.append({"code": code, "variant": v, **out})
    el = pd.DataFrame(erows)
    el = el[el.eligible.astype(bool)]
    el["caught_clean"] = el.caught.astype(bool) & ~el.false_event.astype(bool)
    es = el.groupby("variant").agg(events=("code", "nunique"), caught=("caught", "mean"), false_event=("false_event", "mean"),
                                   caught_clean=("caught_clean", "mean"), median_days=("days_to_detection", "median")).reset_index()
    es.to_csv(res / "microwave_components_event_level.csv", index=False)

    rrows = []
    for v, a in rel.items():
        for k in range(bins):
            if a[k, 0]:
                rrows.append({"stratum": v, "bin": k, "n": int(a[k, 0]), "sum_p": a[k, 1], "n_flood": a[k, 2]})
    rs = validate.summarise_reliability(pd.DataFrame(rrows), bins).rename(columns={"stratum": "variant"})
    rs.to_csv(res / "microwave_components_reliability.csv", index=False)

    rt = pd.concat(rates, ignore_index=True)
    rt = rt[rt.stratum_code % 2 == 0].groupby(["source", "landtype"]).agg(sens_min=("sens", "min"), sens_max=("sens", "max"),
                                                                        spec_min=("spec", "min"), spec_max=("spec", "max")).reset_index()
    pd.set_option("display.width", 220)
    print(tab[tab.group.isin(["overall", "continent", "viirs_period"])].round(3).to_string(index=False), "\n")
    print(diffs.round(3).to_string(index=False), "\n")
    print(es.round(3).to_string(index=False), "\n")
    print(rs[["variant", "reliability_slope_wls", "mean_p", "obs_freq"]].round(3).to_string(index=False), "\n")
    print(rt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
