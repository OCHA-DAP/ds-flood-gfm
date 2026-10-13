"""Build the accuracy table (Phase 3) and prior v1 (Phase 2 deliverable, needs the table).

Accuracy table (fixed decision 5):
  * gold first: per source x stratum confusion matrices pooled over TRAINING events,
    gold days + matched dry windows; sens = TP/(TP+FN), spec = TN/(TN+FP);
    95% intervals by event-cluster bootstrap; sample sizes for every number.
  * strata with fewer than `em_min_gold_cells` observed gold cell-days (max over the
    three model sources) are re-estimated with semi-supervised Dawid-Skene EM:
    gold cells labelled, sample cells unlabelled, initialised from the nearest
    stratum's gold values (same land cover, other snow state; else same snow state,
    most gold), MAP with Beta pseudo-counts `em_prior_strength` (D-24).
  * any interval wider than `ci_flag_width` is flagged.
Prior v1: per-stratum prevalence MLE on the unlabelled sample with sens/spec from
the final table; 95% interval by block bootstrap.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import accuracy, cube, gold
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, bootstrap_ci, pooled
from ds_flood_gfm.fusion.static import STRATUM_NAMES

log = logging.getLogger("phase3")
CODE_OF = {v: k for k, v in STRATUM_NAMES.items()}


def nearest_strata(name: str) -> list[str]:
    lc, snow = name.rsplit("_", 1)
    other_snow = f"{lc}_{'nosnow' if snow == 'snow' else 'snow'}"
    same_snow = [f"{x}_{snow}" for x in ("open", "vegetated", "urban") if x != lc]
    return [other_snow, *same_snow, *[f"{x}_{'nosnow' if snow == 'snow' else 'snow'}" for x in ("open", "vegetated", "urban") if x != lc]]


def gold_pattern_counts(cfg: Config, sel: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Joint (gfm, microwave, viirs, truth) pattern counts per stratum and event (gold + dry days)."""
    path = cfg.local("metrics", f"gold_patterns_{split}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    out = []
    for _, r in sel[sel.split == split].iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            t = ds.truth.values
            st = ds.stratum.values
            obs = {s: ds[s].values for s in MODEL_SOURCES}
            for code, name in STRATUM_NAMES.items():
                m = ((t == 0) | (t == 1)) & (st == code)
                if not m.any():
                    continue
                pc = accuracy.pattern_counts(obs, MODEL_SOURCES, m, t)
                pc["stratum"], pc["code"] = name, r.code
                out.append(pc)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(path)
    return df


def sample_pattern_counts(cfg: Config) -> pd.DataFrame:
    """Unlabelled joint pattern counts per stratum and sample block (domain cells, every day)."""
    path = cfg.local("metrics", "sample_patterns.parquet")
    if path.exists():
        return pd.read_parquet(path)
    blocks = pd.read_parquet(cfg.local("sample", "blocks.parquet"))
    fpath = cfg.results_dir / "phase2_sample_failures.csv"
    failed = set(pd.read_csv(fpath).block_id) if fpath.exists() else set()
    if failed:
        log.warning("excluding %d sample blocks that failed (listed in %s): %s", len(failed), fpath, sorted(failed))
    out = []
    for bid in [b for b in blocks.block_id if b not in failed]:
        p = cfg.local("cube", "sample", f"{bid}.nc")
        if not p.exists():
            raise FileNotFoundError(f"sample cube missing for {bid}: run phase2 sources first")
        ds = xr.load_dataset(p, engine="netcdf4")
        st = ds.stratum.values
        dom = ds.domain.values.astype(bool)[None, :, :]
        obs = {s: ds[s].values for s in MODEL_SOURCES}
        for code, name in STRATUM_NAMES.items():
            m = dom & (st == code)
            if not m.any():
                continue
            pc = accuracy.pattern_counts(obs, MODEL_SOURCES, m)
            pc["stratum"], pc["block_id"] = name, bid
            out.append(pc)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(path)
    return df


def _effective_counts(lab: pd.DataFrame, unl: pd.DataFrame, res: accuracy.EMResult) -> list[tuple[float, float]]:
    """Per source: expected number of flood (for sens) and dry (for spec) cells it observed,
    under the EM posterior. Compared with the Beta prior's pseudo-counts it says whether the
    estimate is driven by data or by the donor stratum."""
    out = []
    pats, w, q = [], [], []
    if len(unl):
        U = unl[MODEL_SOURCES].values.astype("int64")
        l1, l0 = accuracy._loglik_terms(U, res.sens, res.spec)
        qq = 1.0 / (1.0 + np.exp((np.log1p(-res.prevalence) + l0) - (np.log(res.prevalence) + l1)))
        pats.append(U); w.append(unl.n.values.astype(float)); q.append(qq)
    if len(lab):
        pats.append(lab[MODEL_SOURCES].values.astype("int64")); w.append(lab.n.values.astype(float)); q.append(lab.truth.values.astype(float))
    if not pats:
        return [(0.0, 0.0)] * len(MODEL_SOURCES)
    P, W, Q = np.vstack(pats), np.concatenate(w), np.concatenate(q)
    for j in range(len(MODEL_SOURCES)):
        obs = P[:, j] != 255
        out.append((float((W * Q)[obs].sum()), float((W * (1 - Q))[obs].sum())))
    return out


def gold_table(cfg: Config, counts: pd.DataFrame) -> pd.DataFrame:
    a = cfg["accuracy"]
    tr = counts[(counts.split == "train") & counts.source.isin(MODEL_SOURCES + ["floodscan", "gfds"])]
    t = pooled(tr, ["source", "stratum"])
    # point estimates and intervals use the Jeffreys-smoothed proportions (D-32); raw kept for reference
    t = t.rename(columns={"sens": "sens_raw", "spec": "spec_raw"}).rename(columns={"sens_j": "sens", "spec_j": "spec"})
    ci = bootstrap_ci(tr, ["source", "stratum"], ["sens_j", "spec_j"], reps=a["bootstrap_reps"], seed=20261012)
    ci = ci.rename(columns={"sens_j_lo": "sens_lo", "sens_j_hi": "sens_hi", "spec_j_lo": "spec_lo", "spec_j_hi": "spec_hi"})
    t = t.merge(ci, on=["source", "stratum"], how="left")
    t["n_flood_obs"] = t.tp + t.fn
    t["n_dry_obs"] = t.tn + t.fp
    t["n_gold_obs"] = t.n_flood_obs + t.n_dry_obs
    t["stratum_code"] = t.stratum.map(CODE_OF)
    t["method"] = "gold"
    return t


def _em_stratum(lab: pd.DataFrame, unlab: pd.DataFrame, init_sens, init_spec, init_prev, strength) -> accuracy.EMResult:
    L = lab[MODEL_SOURCES].values.astype("int64") if len(lab) else np.zeros((0, 3), "int64")
    U = unlab[MODEL_SOURCES].values.astype("int64") if len(unlab) else np.zeros((0, 3), "int64")
    return accuracy.dawid_skene(U, unlab.n.values if len(unlab) else np.zeros(0), L,
                                lab.n.values if len(lab) else np.zeros(0), lab.truth.values if len(lab) else np.zeros(0),
                                np.asarray(init_sens, float), np.asarray(init_spec, float), float(init_prev), prior_strength=strength)


def build_accuracy_table(cfg: Config, counts: pd.DataFrame, gold_pat: pd.DataFrame, samp_pat: pd.DataFrame | None) -> pd.DataFrame:
    a = cfg["accuracy"]
    g = gold_table(cfg, counts)
    model = g[g.source.isin(MODEL_SOURCES)].copy()
    n_by_stratum = model.groupby("stratum").n_gold_obs.max()
    thin, why = [], {}
    for st in STRATUM_NAMES.values():
        sub = model[model.stratum == st].set_index("source")
        missing = [s for s in MODEL_SOURCES if s not in sub.index]
        zero_flood = [s for s in MODEL_SOURCES if s in sub.index and sub.loc[s, "n_flood_obs"] == 0]
        if n_by_stratum.get(st, 0) < a["em_min_gold_cells"]:
            thin.append(st)
            why[st] = f"{int(n_by_stratum.get(st, 0))} observed gold cell-days < {a['em_min_gold_cells']}"
        elif missing or zero_flood:
            # sensitivity would be 0/0 for these sources: not an estimate, so EM is required (D-24)
            thin.append(st)
            why[st] = f"no observed gold flood cells for {missing + zero_flood}"
    log.info("observed gold cell-days per stratum (max over sources): %s", n_by_stratum.to_dict())
    log.info("EM strata and why: %s", why)
    rows = []
    rng = np.random.default_rng(7)
    for st in STRATUM_NAMES.values():
        sub = model[model.stratum == st].set_index("source")
        if st not in thin:
            for s in MODEL_SOURCES:
                rows.append(sub.loc[s].to_dict() | {"source": s, "stratum": st})
            continue
        donor = next((d for d in nearest_strata(st) if d not in thin), None)
        if donor is None:
            raise RuntimeError(f"no stratum with complete gold estimates to initialise EM for {st}")
        dsub = model[model.stratum == donor].set_index("source")
        init_sens = [dsub.loc[s, "sens"] for s in MODEL_SOURCES]
        init_spec = [dsub.loc[s, "spec"] for s in MODEL_SOURCES]
        gp = gold_pat[gold_pat.stratum == st]
        sp = samp_pat[samp_pat.stratum == st] if samp_pat is not None else pd.DataFrame(columns=[*MODEL_SOURCES, "n", "block_id"])
        lab = gp.groupby([*MODEL_SOURCES, "truth"], as_index=False).n.sum() if len(gp) else gp
        unl = sp.groupby(MODEL_SOURCES, as_index=False).n.sum() if len(sp) else sp
        informative_lab = int(lab[(lab[MODEL_SOURCES] != 255).any(axis=1)].n.sum()) if len(lab) else 0
        informative_unl = int(unl[(unl[MODEL_SOURCES] != 255).any(axis=1)].n.sum()) if len(unl) else 0
        if informative_lab + informative_unl == 0:
            # nothing at all to learn from: the donor's values are copied, and the row says so
            for j, s in enumerate(MODEL_SOURCES):
                rows.append({"source": s, "stratum": st, "method": f"no_data_donor_copy({donor})", "sens": init_sens[j],
                             "spec": init_spec[j], "sens_lo": np.nan, "sens_hi": np.nan, "spec_lo": np.nan, "spec_hi": np.nan,
                             "n_unlabelled": 0, "em_converged": False, "force_flag": True, "em_reason": why[st]})
            continue
        init_prev = float(lab[lab.truth == 1].n.sum() / max(lab.n.sum(), 1)) if len(lab) else 0.01
        init_prev = min(max(init_prev, 1e-3), 0.5)
        res = _em_stratum(lab, unl, init_sens, init_spec, init_prev, a["em_prior_strength"])
        n_eff = _effective_counts(lab, unl, res)
        boots = []
        codes = gp.code.unique() if len(gp) else np.array([])
        blocks = sp.block_id.unique() if len(sp) else np.array([])
        for _ in range(min(a["bootstrap_reps"], 200)):
            cs = rng.choice(codes, len(codes)) if len(codes) else []
            bs = rng.choice(blocks, len(blocks)) if len(blocks) else []
            lab_b = pd.concat([gp[gp.code == c] for c in cs]) if len(cs) else gp.iloc[:0]
            unl_b = pd.concat([sp[sp.block_id == b] for b in bs]) if len(bs) else sp.iloc[:0]
            lab_b = lab_b.groupby([*MODEL_SOURCES, "truth"], as_index=False).n.sum() if len(lab_b) else lab_b
            unl_b = unl_b.groupby(MODEL_SOURCES, as_index=False).n.sum() if len(unl_b) else unl_b
            r = _em_stratum(lab_b, unl_b, init_sens, init_spec, init_prev, a["em_prior_strength"])
            boots.append(np.concatenate([r.sens, r.spec]))
        B = np.array(boots)
        for j, s in enumerate(MODEL_SOURCES):
            base = sub.loc[s].to_dict() if s in sub.index else {}
            nf, nd = n_eff[j]
            prior_dominated = nf < a["em_prior_strength"] or nd < a["em_prior_strength"]
            rows.append(base | {
                "source": s, "stratum": st, "sens": res.sens[j], "spec": res.spec[j],
                "method": f"em(init={donor})" + (",prior-dominated" if prior_dominated else ""),
                "em_n_eff_flood": nf, "em_n_eff_dry": nd, "force_flag": bool(prior_dominated),
                "sens_lo": np.quantile(B[:, j], 0.025), "sens_hi": np.quantile(B[:, j], 0.975),
                "spec_lo": np.quantile(B[:, 3 + j], 0.025), "spec_hi": np.quantile(B[:, 3 + j], 0.975),
                "n_unlabelled": int(unl.n.sum()) if len(unl) else 0, "n_informative_labelled": informative_lab,
                "n_informative_unlabelled": informative_unl, "em_converged": res.converged, "em_iterations": res.iterations,
                "em_prevalence": res.prevalence, "em_reason": why[st],
            })
    t = pd.DataFrame(rows)
    t["stratum_code"] = t.stratum.map(CODE_OF)
    t["sens_ci_width"] = t.sens_hi - t.sens_lo
    t["spec_ci_width"] = t.spec_hi - t.spec_lo
    if "force_flag" not in t:
        t["force_flag"] = False
    t["force_flag"] = t.force_flag.astype("boolean").fillna(False).astype(bool)
    t["flag_wide_ci"] = (t.sens_ci_width > a["ci_flag_width"]) | (t.spec_ci_width > a["ci_flag_width"]) | t.force_flag \
        | t.sens_ci_width.isna() | t.spec_ci_width.isna()
    # a bootstrap interval from a handful of cells (often one event) can collapse to zero width;
    # such rows are flagged separately because their width says nothing (D-32)
    nf = t.n_flood_obs.fillna(0) if "n_flood_obs" in t else 0
    nd = t.n_dry_obs.fillna(0) if "n_dry_obs" in t else 0
    t["flag_small_n"] = ((nf < a["min_obs_for_interval"]) | (nd < a["min_obs_for_interval"])) & t.method.eq("gold")
    cols = ["source", "stratum", "stratum_code", "method", "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi",
            "n_flood_obs", "n_dry_obs", "n_gold_obs", "n_events", "n_unlabelled", "n_informative_labelled",
            "n_informative_unlabelled", "em_converged", "em_prevalence", "em_n_eff_flood", "em_n_eff_dry", "em_reason",
            "sens_raw", "spec_raw", "sens_ci_width", "spec_ci_width", "flag_wide_ci", "flag_small_n"]
    for c in cols:
        if c not in t:
            t[c] = np.nan
    t = t[cols].sort_values(["stratum_code", "source"]).reset_index(drop=True)
    if not np.isfinite(t[["sens", "spec"]].astype(float)).all().all():
        raise RuntimeError("accuracy table still has non-finite sens/spec after EM")
    return t, g


def component_table(gold_all: pd.DataFrame, comp: str) -> tuple[pd.DataFrame, list[str]]:
    """Gold sens/spec for one microwave component (ablation only). Strata where the component has no
    observed gold flood cells take the nearest stratum's value; returns the table and the donor-filled strata."""
    g = gold_all[gold_all.source == comp].set_index("stratum")
    ok = [s for s in g.index if g.loc[s, "n_flood_obs"] > 0 and np.isfinite(g.loc[s, "sens"]) and np.isfinite(g.loc[s, "spec"])]
    rows, filled = [], []
    for code, st in STRATUM_NAMES.items():
        src = st if st in ok else next((d for d in nearest_strata(st) if d in ok), None)
        if src is None:
            raise RuntimeError(f"{comp}: no stratum with gold flood observations to fill {st}")
        if src != st:
            filled.append(f"{st}<-{src}")
        rows.append({"source": comp, "stratum_code": code, "stratum": st, "sens": g.loc[src, "sens"], "spec": g.loc[src, "spec"]})
    return pd.DataFrame(rows), filled


def post_window_pofd(counts: pd.DataFrame) -> pd.DataFrame:
    """D-12 diagnostic: false-alarm rate in the post window by days after the event end."""
    d = counts[(counts.split == "train") & (counts.day_type.isin(["dry_pre", "dry_post"]))].copy()
    d["bin"] = np.where(d.day_type == "dry_pre", "pre (-30..-1)", np.where(d.days_after_end <= 30, "post 1-30", "post 31-60"))
    return pooled(d, ["source", "bin"])[["source", "bin", "fp", "tn", "pofd", "obs_truth_dry"]]


def build_prior_v1(cfg: Config, table: pd.DataFrame, samp_pat: pd.DataFrame) -> pd.DataFrame:
    """Prior v1: per-stratum base flood rate from the stratified random sample (D-35).

    base_rate = share of GFM-observed sample cell-days that GFM flags as flood (GFM has the
    lowest false-alarm rate of the three sources). The prevalence MLE with gold sens/spec is
    kept as a diagnostic (base_rate_mle): it collapses to its lower bound because false-alarm
    rates measured in event dry windows exceed those at random times.
    Strata where GFM observed nothing in the sample take the nearest stratum's rate (flagged).
    95% intervals: bootstrap over sample blocks.
    """
    rng = np.random.default_rng(11)
    rows = []
    gfm_rate = {}
    for code, st in STRATUM_NAMES.items():
        sp = samp_pat[samp_pat.stratum == st]
        agg = sp.groupby(MODEL_SOURCES, as_index=False).n.sum()
        obs = agg[agg.gfm != 255]
        n_obs, n_fl = int(obs.n.sum()), int(obs[obs.gfm == 1].n.sum())
        blocks = sp.block_id.unique()
        boots, boots_mle = [], []
        t = table[table.stratum == st].set_index("source")
        sens = np.array([t.loc[s, "sens"] for s in MODEL_SOURCES])
        spec = np.array([t.loc[s, "spec"] for s in MODEL_SOURCES])
        mle = accuracy.prevalence_mle(agg[MODEL_SOURCES].values.astype("int64"), agg.n.values, sens, spec) if len(agg) else np.nan
        per_block = {b: sp[sp.block_id == b] for b in blocks}
        for _ in range(200):
            bs = rng.choice(blocks, len(blocks)) if len(blocks) else []
            if not len(bs):
                break
            b = pd.concat([per_block[x] for x in bs])
            o = b[b.gfm != 255]
            boots.append(o[o.gfm == 1].n.sum() / o.n.sum() if o.n.sum() else np.nan)
        rate = n_fl / n_obs if n_obs else np.nan
        gfm_rate[st] = rate
        rows.append({"stratum_code": code, "stratum": st, "base_rate": rate,
                     "base_rate_lo": float(np.nanquantile(boots, 0.025)) if boots else np.nan,
                     "base_rate_hi": float(np.nanquantile(boots, 0.975)) if boots else np.nan,
                     "n_gfm_observed_cell_days": n_obs, "n_gfm_flood_cell_days": n_fl,
                     "n_cell_days": int(agg.n.sum()), "n_blocks": int(len(blocks)), "base_rate_mle": mle, "method": "gfm_sample_rate"})
    out = pd.DataFrame(rows)
    for i, r in out.iterrows():
        if not (r.n_gfm_observed_cell_days > 0) or not np.isfinite(r.base_rate):
            donor = next((d for d in nearest_strata(r.stratum) if np.isfinite(gfm_rate.get(d, np.nan)) and gfm_rate[d] > 0), None)
            if donor is None:
                raise RuntimeError(f"no stratum with GFM sample observations to fill the prior of {r.stratum}")
            out.loc[i, ["base_rate", "method"]] = [gfm_rate[donor], f"donor({donor})"]
    if not np.isfinite(out.base_rate.astype(float)).all():
        raise RuntimeError("prior v1 has non-finite base rates")
    return out
