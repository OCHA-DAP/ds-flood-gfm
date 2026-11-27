"""Phase 1 feasibility: per-source skill by stratum, oracle test, coverage test, gate.

Uses TRAINING events only, so the held-out events stay unseen until Phase 5.
Gate (prompt, Phase 1): shrink to a 5-basin prototype iff
    oracle gain < 3 CSI points AND 3-day coverage gain < 15 percentage points.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import cube, gold
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, add_scores, event_counts, event_coverage, pooled

log = logging.getLogger("phase1")
MIN_FLOOD_CELLS_FOR_ORACLE = 200


def s1_days_by_code(cfg: Config) -> dict[str, set]:
    kept = pd.read_parquet(cfg.local("gold", "label_sets_kept.parquet"))
    out: dict[str, set] = {}
    for (code, day), g in kept.groupby(["code", kept.label_day.dt.normalize()]):
        if bool(g.is_s1.all()):
            out.setdefault(code, set()).add(day)
    return out


def extract_all(cfg: Config, sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts and coverage for every selected event (cached as parquet)."""
    cpath, vpath = cfg.local("metrics", "counts.parquet"), cfg.local("metrics", "coverage.parquet")
    if cpath.exists() and vpath.exists():
        return pd.read_parquet(cpath), pd.read_parquet(vpath)
    missing = [ew.key for _, r in sel.iterrows() for ew in gold.event_windows(r) if not cube.cube_path(cfg, ew.key).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} event-window cubes missing (first {missing[:5]}); finish stage C or list "
                                "failures in results/phase1_stage_c_failures.csv before computing metrics")
    s1 = s1_days_by_code(cfg)
    counts, cover = [], []
    for _, r in sel.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            counts.append(event_counts(ds, r.code, r.continent, r.split, s1.get(r.code, set())))
            cover.append(event_coverage(ds, r.code, r.continent, r.split))
    c, v = pd.concat(counts, ignore_index=True), pd.concat(cover, ignore_index=True)
    c.to_parquet(cpath)
    v.to_parquet(vpath)
    return c, v


def oracle_test(gold_days: pd.DataFrame, score: str) -> dict:
    """Best single source per stratum (hindsight) vs best single source overall, pooled CSI."""
    by_ss = pooled(gold_days[gold_days.source.isin(MODEL_SOURCES)], ["source", "stratum"])
    overall = pooled(gold_days[gold_days.source.isin(MODEL_SOURCES)], ["source"]).set_index("source")
    best_overall = overall[score].idxmax()
    chosen = {}
    per_stratum_gain = []
    for st, g in by_ss.groupby("stratum"):
        n_col = "obs_truth_flood" if score == "csi" else "truth_flood"
        # candidates: sources with enough flood evidence in this stratum, plus the overall best
        # (so the hindsight choice can never be a worse-evidenced source than the baseline)
        cand = g[(g[n_col] >= MIN_FLOOD_CELLS_FOR_ORACLE) | (g.source == best_overall)]
        cand = cand[cand[score].notna()]
        if cand.empty:
            chosen[st] = best_overall
            continue
        chosen[st] = cand.loc[cand[score].idxmax(), "source"]
        base_st = g[g.source == best_overall][score]
        if len(base_st) and base_st.notna().all():
            per_stratum_gain.append({"stratum": st, "weight": int(g[g.source == best_overall].truth_flood.iloc[0]),
                                     "gain": float(cand[score].max() - base_st.iloc[0])})
    pick = pd.concat([by_ss[(by_ss.stratum == st) & (by_ss.source == s)] for st, s in chosen.items()])
    tot = add_scores(pick[["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum().to_frame().T)
    base = add_scores(by_ss[by_ss.source == best_overall][["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum().to_frame().T)
    return {
        "score": score,
        "best_overall_source": best_overall,
        "best_overall": float(base[score].iloc[0]),
        "oracle": float(tot[score].iloc[0]),
        "gain_points": float(100 * (tot[score].iloc[0] - base[score].iloc[0])),
        "chosen_by_stratum": chosen,
        # flood-weighted mean of per-stratum (best - baseline); >= 0 by construction
        "per_stratum_gain_points": float(100 * np.average([x["gain"] for x in per_stratum_gain], weights=[max(x["weight"], 1) for x in per_stratum_gain])) if per_stratum_gain else float("nan"),
    }


def coverage_test(cov: pd.DataFrame) -> pd.DataFrame:
    cols = ["gfm", "microwave", "viirs", "floodscan", "gfds", "any", "any_3d", "gfm_3d"]
    tot = cov[["n_domain", *cols]].sum()
    out = pd.DataFrame({"what": cols, "coverage": [tot[c] / tot.n_domain for c in cols]})
    # mean of daily coverage fractions (each window-day equally weighted)
    daily = cov.groupby(["code", "window", "day"])[["n_domain", *cols]].sum()
    for c in cols:
        daily[c] = daily[c] / daily.n_domain
    out["mean_daily_coverage"] = [daily[c].mean() for c in cols]
    return out


def run(cfg: Config, sel: pd.DataFrame) -> dict:
    counts, cov = extract_all(cfg, sel)
    res = cfg.results_dir
    tr = counts[counts.split == "train"]
    gold_days = tr[tr.day_type == "gold"]
    # 1. skill per source x stratum (gold days, conditional and all-cells)
    skill = pooled(gold_days, ["source", "stratum"])
    spec_all = pooled(tr, ["source", "stratum"])[["source", "stratum", "spec", "pofd", "truth_dry", "obs_truth_dry"]].rename(
        columns={"spec": "spec_incl_dry_windows", "pofd": "pofd_incl_dry_windows", "truth_dry": "dry_cells_incl_windows", "obs_truth_dry": "obs_dry_cells_incl_windows"})
    skill = skill.merge(spec_all, on=["source", "stratum"], how="left")
    skill.to_csv(res / "phase1_skill_by_source_stratum.csv", index=False)
    pooled(gold_days, ["source"]).to_csv(res / "phase1_skill_by_source.csv", index=False)
    pooled(gold_days, ["source", "continent"]).to_csv(res / "phase1_skill_by_source_continent.csv", index=False)
    # circularity: GFM against Sentinel-1-derived gold vs other gold
    pooled(gold_days, ["source", "label_s1"]).to_csv(res / "phase1_skill_by_label_s1.csv", index=False)
    # 2. oracle
    oracle_c = oracle_test(gold_days, "csi")
    oracle_a = oracle_test(gold_days, "csi_all")
    # 3. coverage (training event windows, every day, floodable domain)
    cv = coverage_test(cov[cov.split == "train"])
    cv.to_csv(res / "phase1_coverage.csv", index=False)
    cvi = cv.set_index("what").coverage
    cov_gain = float(100 * (cvi["any_3d"] - cvi["gfm"]))
    shrink = (oracle_c["gain_points"] < 3.0) and (cov_gain < 15.0)
    gate = {
        "oracle_conditional": oracle_c,
        "oracle_all_cells": oracle_a,
        "coverage": {k: float(v) for k, v in cvi.items()},
        "coverage_gain_any3d_minus_gfm_pp": cov_gain,
        "coverage_gain_any3d_minus_gfm3d_pp": float(100 * (cvi["any_3d"] - cvi["gfm_3d"])),
        "rule": "shrink iff oracle gain < 3 CSI points AND 3-day coverage gain < 15 pp",
        "decision": "shrink to 5-basin prototype" if shrink else "continue in full",
        "n_train_events": int(tr.code.nunique()),
        "n_gold_flood_cells_train": int(gold_days[gold_days.source == "gfm"].truth_flood.sum()),
        "n_gold_dry_cells_train_gold_days": int(gold_days[gold_days.source == "gfm"].truth_dry.sum()),
    }
    (res / "phase1_gate.json").write_text(json.dumps(gate, indent=2, default=str))
    log.info("gate: %s", json.dumps(gate, indent=2, default=str))
    _ = np
    return gate
