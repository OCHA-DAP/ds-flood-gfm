"""Two descriptive breakdowns of the cross-validated scores (no refit, no rule changes).

Reads the per-event map-day counts phase5_crossval.py caches (data/fusion/metrics/cv_skill_events.parquet)
and results/event_folds.csv. Writes, to results/cv/:
  skill_without_largest_basin.csv  every product, all cells and observed cells, without the basin that
                                   holds the most gold flood (one HydroBASINS level-4 basin in Bangladesh
                                   holds 40% of it), to show how much that one basin drives the totals
  skill_by_viirs_period.csv        events whose maps fall inside the VIIRS archive gap (no VIIRS files,
                                   2021-01-01 to 2023-07-31) against the rest
  skill_viirs_ablation.csv         fusion with and without VIIRS (same folds, rates and prior; the engine is
                                   re-run without VIIRS, which needs the event cubes), by VIIRS period, with a
                                   paired bootstrap interval for the difference (whole events resampled)

Run: uv run python scripts/fusion/cv_breakdowns.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import crossval, cube, engine, gold
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, pooled

GAP = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-07-31"))


def table(sk: pd.DataFrame, by: list[str], reps: int) -> pd.DataFrame:
    keys = ["product", "mode", *by]
    t = pooled(sk, keys).rename(columns={"sens": "recall"})
    ci = crossval.bootstrap_scores(sk.groupby([*keys, "code"], as_index=False)[["tp", "fp", "fn"]].sum(), keys, reps, seed=20261015)
    return t.merge(ci, on=keys)[[*keys, "n_events", "truth_flood", "recall", "precision", "f1", "f1_lo", "f1_hi"]]


def viirs_ablation(cfg, ev: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Fusion with all three sources vs without VIIRS: the same fold rates (naive Bayes rates are per source,
    so dropping one leaves the others' unchanged) and prior, the adopted dry-window rule, map days, all cells."""
    f, thr = cfg["fusion"], cfg["fusion"]["prob_threshold"]
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == cfg["gold"]["dry_window_rule"]]
    prior = pd.read_csv(cfg.results_dir / "cv" / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    variants = {"fusion": MODEL_SOURCES, "fusion without VIIRS": [s for s in MODEL_SOURCES if s != "viirs"]}
    models = {}
    for k in sorted(ev.fold.unique()):
        t = crossval.fit_table(counts, ev.code[ev.fold != k])
        for name, srcs in variants.items():
            models[(k, name)] = engine.Model.from_tables(t, prior, srcs, f["decay"], clip)
    rows = []
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            st, dom = ds.stratum.values, ds.domain.values.astype(bool)
            for name in variants:
                m = models[(r.fold, name)]
                p = engine.run_sequence({s: ds[s].values for s in m.sources}, st, dom, m)["p_flood"]
                sr = crossval.score_rows(ds, p, r.code, r.continent, set(), thr, name, include_single=False)
                rows.append(sr.assign(viirs_period=r.viirs_period))
    sk = pd.concat(rows, ignore_index=True)
    sk = sk[sk["mode"] == "all"]
    out = table(sk, ["viirs_period"], reps)
    out = pd.concat([out, table(sk, [], reps).assign(viirs_period="all")], ignore_index=True)
    # paired difference: the same resampled events for both variants
    rng = np.random.default_rng(20261015)
    diff = []
    for per, g in [*sk.groupby("viirs_period"), ("all", sk)]:
        e = g.groupby(["code", "product"])[["tp", "fp", "fn"]].sum().unstack("product")
        a = e.xs("fusion", axis=1, level=1).to_numpy(float)
        b = e.xs("fusion without VIIRS", axis=1, level=1).to_numpy(float)
        f1 = lambda m: 2 * m[..., 0] / (2 * m[..., 0] + m[..., 1] + m[..., 2])  # noqa: E731
        idx = rng.integers(0, len(a), (reps, len(a)))
        d = f1(a[idx].sum(1)) - f1(b[idx].sum(1))
        with np.errstate(invalid="ignore"):
            per_event = f1(a) - f1(b)  # NaN for events with no flood cell and no flood call
        per_event = per_event[np.isfinite(per_event)]
        diff.append({"viirs_period": per, "f1_gain_from_viirs": float(f1(a.sum(0)) - f1(b.sum(0))),
                     "gain_lo": float(np.quantile(d, 0.025)), "gain_hi": float(np.quantile(d, 0.975)),
                     "events_scored": len(per_event), "events_viirs_raises_f1": int((per_event > 0.005).sum()),
                     "events_viirs_lowers_f1": int((per_event < -0.005).sum())})
    return out.merge(pd.DataFrame(diff), on="viirs_period")


def main() -> None:
    cfg = load_config()
    reps = cfg["accuracy"]["bootstrap_reps"]
    res = cfg.results_dir / "cv"
    sk = pd.read_parquet(cfg.local("metrics", "cv_skill_events.parquet"))
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    if set(sk.code) != set(folds.code):
        raise ValueError("cached scores and event_folds.csv cover different events: re-run phase5_crossval.py")
    big = folds.groupby("basin_id").flood_cell_days.sum().idxmax()
    share = folds[folds.basin_id == big].flood_cell_days.sum() / folds.flood_cell_days.sum()
    out = table(sk[~sk.code.isin(folds[folds.basin_id == big].code)], [], reps)
    out.insert(0, "excluded_basin", big)
    out.insert(1, "excluded_share_of_gold_flood", round(share, 3))
    out.to_csv(res / "skill_without_largest_basin.csv", index=False)

    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet")).set_index("code")
    inside = (ev.start >= GAP[0]) & (ev.end <= GAP[1])
    label = lambda c: "VIIRS gap (2021-01 to 2023-07)" if inside[c] else "VIIRS archive available"  # noqa: E731
    by_p = table(sk.assign(viirs_period=sk.code.map(label)), ["viirs_period"], reps)
    by_p.to_csv(res / "skill_by_viirs_period.csv", index=False)
    ev_f = ev.loc[folds.code].reset_index().merge(folds[["code", "fold"]], on="code")
    ab = viirs_ablation(cfg, ev_f.assign(viirs_period=ev_f.code.map(label)), reps)
    ab.to_csv(res / "skill_viirs_ablation.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"largest basin {big}: {share:.0%} of gold flood")
    print(out[out["mode"] == "all"].round(3).to_string(index=False))
    print(by_p[by_p["mode"] == "all"].sort_values(["viirs_period", "f1"]).round(3).to_string(index=False))
    print(ab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
