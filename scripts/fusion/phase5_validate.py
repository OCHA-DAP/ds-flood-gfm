"""Phase 5: validation on held-out events, daily outputs for every held-out window, ablation.

Outputs (results/):
  phase5_skill_by_stratum.csv, phase5_skill_by_continent.csv, phase5_skill_overall.csv
  phase5_criteria.csv                (a) coverage, (b) CSI per stratum vs best single, (c) reliability slope
  phase5_reliability_bins.csv, phase5_reliability_summary.csv, reliability_<stratum>.png
  phase5_coverage_daily.csv, phase5_coverage_summary.csv, phase5_obs_age_on_gold_days.csv
  phase5_timing.csv, phase5_ablation.csv
Daily COGs: data/fusion/out/daily/<event window>/ (mirrored to blob by phase 6).

Run: nice -n 19 uv run python scripts/fusion/phase5_validate.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ds_flood_gfm.fusion import engine, job, validate  # noqa: E402
from ds_flood_gfm.fusion.config import load_config  # noqa: E402
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES  # noqa: E402

log = logging.getLogger("phase5")


def ablation_models(cfg) -> dict[str, engine.Model]:
    """Full model, each source dropped, and microwave replaced by one of its components."""
    acc = pd.read_csv(cfg.results_dir / "accuracy_table.csv")
    gold_all = pd.read_csv(cfg.results_dir / "accuracy_table_gold_all.csv")
    prior = pd.read_csv(cfg.results_dir / "prior_v1.csv")
    f = cfg["fusion"]
    clip = tuple(f["sens_spec_clip"])
    models = {"full": engine.Model.from_tables(acc, prior, MODEL_SOURCES, f["decay"], clip)}
    for s in MODEL_SOURCES:
        keep = [x for x in MODEL_SOURCES if x != s]
        models[f"drop_{s}"] = engine.Model.from_tables(acc, prior, keep, f["decay"], clip)
    from ds_flood_gfm.fusion.table import component_table

    for comp in ["floodscan", "gfds"]:
        ct, filled = component_table(gold_all, comp)
        log.info("ablation component %s: gold estimates, donor-filled strata %s", comp, filled)
        a2 = pd.concat([acc[acc.source != "microwave"], ct[["source", "stratum_code", "sens", "spec"]]])
        models[f"microwave_is_{comp}"] = engine.Model.from_tables(a2, prior, ["gfm", comp, "viirs"], f["decay"], clip)
    return models


def criteria(cfg, st_tab: pd.DataFrame, cov_sum: pd.DataFrame, rel_sum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = st_tab[st_tab.model == "full"]
    for st, g in full.groupby("stratum"):
        a = g[g["mode"] == "all"].set_index("product")
        singles = a.drop(index="fusion", errors="ignore")
        best = singles.csi.idxmax() if singles.csi.notna().any() else None
        rel = rel_sum.set_index("stratum")
        slope = rel.loc[st, "reliability_slope_wls"] if st in rel.index else np.nan
        lo, hi = cfg["validation"]["slope_ok"]
        rows.append({
            "stratum": st, "fusion_csi_all": a.loc["fusion", "csi"] if "fusion" in a.index else np.nan,
            "best_single": best, "best_single_csi_all": singles.csi.max() if best else np.nan,
            "b_csi_ge_best_single": bool(a.loc["fusion", "csi"] >= singles.csi.max()) if best else None,
            "n_gold_flood": int(a.loc["fusion", "truth_flood"]) if "fusion" in a.index else 0,
            "reliability_slope_wls": slope, "c_slope_in_range": bool(lo <= slope <= hi) if np.isfinite(slope) else None,
        })
    out = pd.DataFrame(rows)
    c = cov_sum.set_index("what").coverage
    singles = c.drop(index=["fusion_today", "fusion_within_3d"])
    out.attrs["a_coverage"] = {"fusion_today": float(c["fusion_today"]), "best_single": singles.idxmax(), "best_single_cov": float(singles.max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "phase5.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    res = cfg.results_dir
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    # the resumable job would reuse state/COGs written with an earlier model: clear the held-out
    # windows' own folders only (operational AOI outputs are left alone)
    import shutil

    for _, ew in validate.holdout_keys(cfg, sel):
        for sub in ("state", "daily"):
            d = cfg.local_root / "out" / sub / ew.key
            if d.exists():
                shutil.rmtree(d)
    models = ablation_models(cfg)
    log.info("validating %d held-out events with models %s", (sel.split == "holdout").sum(), list(models))
    allskill = []
    main_out = None
    runs = [(name, m, None) for name, m in models.items()]
    # in-design ablation (D-33): evidence only when a source's value changes day to day
    runs.append(("microwave_on_change", models["full"], {"microwave": validate.on_change}))
    runs.append(("all_on_change", models["full"], {s: validate.on_change for s in MODEL_SOURCES}))
    rel_by_model = {}
    for name, m, tf in runs:
        out = validate.run(cfg, sel, m, name, write_cogs=(name == "full"), transform=tf)
        rel_by_model[name] = validate.summarise_reliability(out["reliability"], cfg["validation"]["reliability_bins"]).assign(model=name)
        allskill.append(out["skill"])
        if name == "full":
            main_out = out
        log.info("model %s done", name)
    skill = pd.concat(allskill, ignore_index=True)
    skill.to_parquet(cfg.local("metrics", "phase5_skill_long.parquet"))
    gold_days = skill[skill.day_type == "gold"]
    st_tab = validate.skill_table(gold_days, ["stratum"])
    st_tab.to_csv(res / "phase5_skill_by_stratum.csv", index=False)
    validate.skill_table(gold_days, ["continent"]).to_csv(res / "phase5_skill_by_continent.csv", index=False)
    overall = validate.skill_table(gold_days, [])
    overall.to_csv(res / "phase5_skill_overall.csv", index=False)
    abl = overall[(overall["product"] == "fusion") & (overall["mode"] == "all")][["model", "tp", "fp", "fn", "sens", "far", "csi"]]
    rel_all = pd.concat(rel_by_model.values(), ignore_index=True)
    rel_all.to_csv(res / "phase5_reliability_by_model.csv", index=False)
    pooled_slope = rel_all.groupby("model").apply(
        lambda g: np.average(g.reliability_slope_wls.fillna(0), weights=g.n) if g.reliability_slope_wls.notna().any() else np.nan,
        include_groups=False).rename("reliability_slope_wls_weighted").reset_index()
    abl = abl.merge(pooled_slope, on="model", how="left")
    abl.to_csv(res / "phase5_ablation.csv", index=False)
    rel = main_out["reliability"]
    rel.to_csv(res / "phase5_reliability_bins.csv", index=False)
    rel_sum = validate.summarise_reliability(rel, cfg["validation"]["reliability_bins"])
    rel_sum.to_csv(res / "phase5_reliability_summary.csv", index=False)
    from ds_flood_gfm.fusion import plots

    for st in rel_sum.stratum:
        plots.reliability(rel, rel_sum, st, res / f"reliability_{st}.png")
    plots.csi_by_stratum(st_tab, res / "phase5_csi_by_stratum.png", mode="all")
    cov = main_out["coverage"]
    cov.to_csv(res / "phase5_coverage_daily.csv", index=False)
    tot = cov[["n_domain", "fusion_today", "fusion_within_3d", *validate.SINGLE]].sum()
    cov_sum = pd.DataFrame({"what": ["fusion_today", "fusion_within_3d", *validate.SINGLE],
                            "coverage": [tot[c] / tot.n_domain for c in ["fusion_today", "fusion_within_3d", *validate.SINGLE]]})
    cov_sum.to_csv(res / "phase5_coverage_summary.csv", index=False)
    cov[cov.gold_day][["code", "day", "age_median", "age_p90", "never_observed", "n_domain"]].to_csv(res / "phase5_obs_age_on_gold_days.csv", index=False)
    main_out["timing"].to_csv(res / "phase5_timing.csv", index=False)
    crit = criteria(cfg, st_tab, cov_sum, rel_sum)
    crit.to_csv(res / "phase5_criteria.csv", index=False)
    log.info("criteria (a) coverage: %s", crit.attrs["a_coverage"])
    log.info("criteria by stratum:\n%s", crit.to_string())
    log.info("ablation:\n%s", abl.to_string())
    _ = job


if __name__ == "__main__":
    main()
