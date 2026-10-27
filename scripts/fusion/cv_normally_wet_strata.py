"""Normally-wet stratum (Next steps 8): rates fitted per land type x normally-wet, scored in the same five folds.

Why: each source's false-alarm rate is 3-11x higher on cells normally wet in that calendar month (Results,
Water versus flood), and one rate per land type is wrong for both kinds of cell.

Rules, fixed before running (2026-10-26):
  - stratum = land type x normally wet (6 codes: landcover * 2 + wet, the same 6-slot layout the engine
    indexes by, with the "wet" bit in the slot v1 used for snow); "normally wet" is the D-38 rule for the
    day's calendar month, read from the cached JRC masks;
  - everything else as in make crossval: folds, prior (the same base rate for both wet states of a land
    type: the random sample is too thin to split), adopted dry-window rule, floodable cells, threshold 0.5,
    memory 0.9, independent evidence;
  - a stratum with no observed gold flood for a source in a fold raises (no donor filling);
  - scored on map days beside fusion as built and the cascade: pooled, basin-weighted (>= 50 gold flooded
    cell-days), per continent, per land type, normally-wet / other; event level (ADR 0007); reliability on
    map days; paired intervals against as built.
Expectation, written first: false-alarm rates on normally-wet cells rise and elsewhere fall; precision on
other cells improves; the flat probabilities do not change much.

Outputs, results/cv/: normally_wet_strata_skill.csv, normally_wet_strata_rates.csv, normally_wet_strata_event_level.csv,
normally_wet_strata_reliability.csv. Cache: data/fusion/metrics/cv_fit_counts_nw.parquet.
Run: nice -n 19 uv run python -u scripts/fusion/cv_normally_wet_strata.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES, pooled
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.validate import SINGLE

STRATA = {0: "open, other", 1: "open, normally wet", 2: "vegetated, other", 3: "vegetated, normally wet", 4: "urban, other", 5: "urban, normally wet"}
BUILT, NEW, CASCADE = "fusion as built (land type)", "fusion, land type x normally wet", "cascade: VIIRS, else GFM, else microwave"
MIN_BASIN_FLOOD = 50


def wet_strata(cfg, ds, ew) -> np.ndarray:
    """(day, lat, lon) stratum codes: land type * 2 + normally wet in that day's month; 255 where unknown."""
    lt = crossval.landtype_codes(ds.stratum.values)
    days = pd.to_datetime(ds.day.values)
    out = np.full(lt.shape, NODATA, "uint8")
    masks = {m: baseline.normally_wet(cfg, ew.window, m) for m in sorted(set(days.month))}
    for i, d in enumerate(days):
        ok = lt[i] != NODATA
        out[i][ok] = lt[i][ok] * 2 + masks[d.month][ok].astype("uint8")
    return out


def fit_counts6(ds, truth, st6, code) -> pd.DataFrame:
    gd = crossval.gold_day_mask(ds)
    dom = ds.domain.values.astype(bool)[None]
    rows = []
    lab = ((truth == gold.DRY) | (truth == gold.FLOOD)) & dom & (st6 != NODATA)
    for s in MODEL_SOURCES:
        v = ds[s].values
        obs = lab & (v != NODATA)
        for k in STRATA:
            m = obs & (st6 == k)
            fl, tf = v[m] == 1, truth[m] == gold.FLOOD
            rows.append({"code": code, "stratum_code": k, "source": s, "tp": int((fl & tf).sum()), "fn": int((~fl & tf).sum()),
                         "fp": int((fl & ~tf).sum()), "tn": int((~fl & ~tf).sum())})
    return pd.DataFrame(rows)


def fit_table6(counts, train_codes) -> pd.DataFrame:
    c = counts[counts.code.isin(set(train_codes))].groupby(["source", "stratum_code"])[["tp", "fn", "fp", "tn"]].sum()
    rows, empty = [], []
    for s in MODEL_SOURCES:
        for k in STRATA:
            r = c.loc[(s, k)] if (s, k) in c.index else pd.Series(0, index=["tp", "fn", "fp", "tn"])
            if r.tp + r.fn == 0 or r.tn + r.fp == 0:
                empty.append(f"{s}/{STRATA[k]} (flood {int(r.tp + r.fn)}, dry {int(r.tn + r.fp)})")
                continue
            rows.append({"source": s, "stratum_code": k, "stratum": STRATA[k], "sens": (r.tp + 0.5) / (r.tp + r.fn + 1), "spec": (r.tn + 0.5) / (r.tn + r.fp + 1),
                         "n_flood_obs": int(r.tp + r.fn), "n_dry_obs": int(r.tn + r.fp)})
    if empty:
        raise ValueError(f"no observed gold cells to fit: {empty}")
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    f, thr, e = cfg["fusion"], cfg["fusion"]["prob_threshold"], cfg["event_level"]
    reps, rule, bins = cfg["accuracy"]["bootstrap_reps"], cfg["gold"]["dry_window_rule"], cfg["validation"]["reliability_bins"]
    res = cfg.results_dir / "cv"
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold", "basin_id"]], on="code").reset_index(drop=True)
    K = sorted(ev.fold.unique())
    t0 = time.time()

    cpath = cfg.local("metrics", "cv_fit_counts_nw.parquet")
    if cpath.exists() and set(pd.read_parquet(cpath).code) == set(ev.code):
        c6 = pd.read_parquet(cpath)
    else:
        rows = []
        for _, r in ev.iterrows():
            for ew in gold.event_windows(r):
                ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
                rows.append(fit_counts6(ds, crossval.dry_truth(ds, rule), wet_strata(cfg, ds, ew), r.code))
        c6 = pd.concat(rows).groupby(["code", "stratum_code", "source"], as_index=False)[["tp", "fn", "fp", "tn"]].sum()
        c6.to_parquet(cpath)
    print(f"counts ready {time.time() - t0:.0f}s", flush=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule]
    prior = pd.read_csv(res / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    models, rates = {}, []
    for k in K:
        train = ev.code[ev.fold != k]
        models[(k, BUILT)] = engine.Model.from_tables(crossval.fit_table(counts, train), prior, MODEL_SOURCES, f["decay"], clip)
        t6 = fit_table6(c6, train)
        rates.append(t6.assign(fold=k))
        models[(k, NEW)] = engine.Model.from_tables(t6, prior, MODEL_SOURCES, f["decay"], clip)
    pd.concat(rates).to_csv(res / "normally_wet_strata_rates.csv", index=False)

    skill, flags, sizes = [], [], defaultdict(int)
    rel = {NEW: np.zeros((3, bins)), BUILT: np.zeros((3, bins))}
    for _, r in ev.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            dom, t = ds.domain.values.astype(bool), ds.truth.values
            gd = crossval.gold_day_mask(ds)
            st6 = wet_strata(cfg, ds, ew)
            obs = {s: ds[s].values for s in MODEL_SOURCES}
            wet = (st6 % 2 == 1) & (st6 != NODATA)
            fl = {}
            for name, st in ((BUILT, ds.stratum.values), (NEW, st6)):
                p = engine.run_sequence(obs, st, dom, models[(r.fold, name)])["p_flood"]
                sr = crossval.score_rows(ds, p, r.code, r.continent, set(), thr, name, include_single=(name == BUILT))
                if len(sr):
                    skill.append(sr[sr["mode"] == "all"])
                # by normally-wet status on map days
                for g, gm in (("normally wet", wet), ("other", ~wet)):
                    lab = ((t == gold.DRY) | (t == gold.FLOOD)) & np.isfinite(p) & gd[:, None, None] & gm
                    if lab.any():
                        call, tf = (p[lab] >= thr), (t[lab] == gold.FLOOD)
                        skill.append(pd.DataFrame([{"code": r.code, "continent": r.continent, "landtype": "all", "label_s1": False, "product": name, "mode": g,
                                                    "tp": int((call & tf).sum()), "fp": int((call & ~tf).sum()), "fn": int((~call & tf).sum()), "tn": int((~call & ~tf).sum()),
                                                    "truth_flood": int(tf.sum()), "truth_dry": int((~tf).sum()), "obs_truth_flood": int(tf.sum()), "obs_truth_dry": int((~tf).sum())}]))
                lab = ((t == gold.DRY) | (t == gold.FLOOD)) & np.isfinite(p) & gd[:, None, None]
                pp = p[lab].astype("float64")
                b = np.clip((pp * bins).astype(int), 0, bins - 1)
                rel[name][0] += np.bincount(b, minlength=bins); rel[name][1] += np.bincount(b, weights=pp, minlength=bins)
                rel[name][2] += np.bincount(b, weights=(t[lab] == gold.FLOOD), minlength=bins)
                fl[name] = np.isfinite(p) & (p >= thr)
            v = {s: ds[s].values for s in SINGLE}
            cas = np.full(v["viirs"].shape, NODATA, "uint8")
            for s in ["microwave", "gfm", "viirs"]:
                cas = np.where(v[s] != NODATA, v[s], cas)
            fl[CASCADE] = cas == 1
            lab = ((t == gold.DRY) | (t == gold.FLOOD)) & gd[:, None, None] & dom[None] & (crossval.landtype_codes(ds.stratum.values) != NODATA)
            lt = crossval.landtype_codes(ds.stratum.values)
            for kk, ltn in crossval.LANDTYPES.items():
                m = lab & (lt == kk)
                if m.any():
                    call, tf = fl[CASCADE][m], t[m] == gold.FLOOD
                    skill.append(pd.DataFrame([{"code": r.code, "continent": r.continent, "landtype": ltn, "label_s1": False, "product": CASCADE, "mode": "all",
                                                "tp": int((call & tf).sum()), "fp": int((call & ~tf).sum()), "fn": int((~call & tf).sum()), "tn": int((~call & ~tf).sum()),
                                                "truth_flood": int(tf.sum()), "truth_dry": int((~tf).sum()), "obs_truth_flood": int(tf.sum()), "obs_truth_dry": int((~tf).sum())}]))
            ff, sz = crossval.footprint_flags(ds, {k: fl[k] for k in (BUILT, NEW, CASCADE)})
            if len(ff):
                flags.append(ff[ff.landtype == "all"].assign(code=r.code))
            sizes[r.code] += sz.get("all", 0)
        print(f"{r.code} done, {time.time() - t0:.0f}s", flush=True)

    sk = pd.concat(skill, ignore_index=True)
    sk = sk.groupby(["code", "continent", "landtype", "product", "mode"], as_index=False)[["tp", "fp", "fn", "tn", "truth_flood", "truth_dry", "obs_truth_flood", "obs_truth_dry"]].sum()
    sk["basin"] = sk.code.map(ev.set_index("code").basin_id)
    rng = np.random.default_rng(20261026)
    out = []

    def view(name, df, unit="code", weights=None):
        d = df.copy()
        if weights is not None:
            for c in ("tp", "fp", "fn"):
                d[c] = d[c] * d[unit].map(weights)
        m = d.groupby([unit, "product"])[["tp", "fp", "fn"]].sum().unstack("product").fillna(0)
        idx = rng.integers(0, len(m), (reps, len(m)))
        prods = [p for p in (BUILT, NEW, CASCADE) if ("tp", p) in m.columns]
        mats = {p: np.stack([m[("tp", p)], m[("fp", p)], m[("fn", p)]], 1).astype(float) for p in prods}
        f1 = lambda x: 2 * x[..., 0] / (2 * x[..., 0] + x[..., 1] + x[..., 2])  # noqa: E731
        F = {p: f1(mats[p][idx].sum(1)) for p in prods}
        for p in prods:
            s = mats[p].sum(0)
            out.append({"view": name, "product": p, "units": len(m), "recall": s[0] / (s[0] + s[2]), "precision": s[0] / (s[0] + s[1]), "f1": float(f1(s)),
                        "f1_lo": np.nanquantile(F[p], 0.025), "f1_hi": np.nanquantile(F[p], 0.975),
                        "minus_built": float(f1(s) - f1(mats[BUILT].sum(0))), "minus_built_lo": np.nanquantile(F[p] - F[BUILT], 0.025), "minus_built_hi": np.nanquantile(F[p] - F[BUILT], 0.975)})

    main_rows = sk[sk["mode"] == "all"]
    fb = main_rows[main_rows["product"] == BUILT].assign(fl=lambda d: d.tp + d.fn).groupby("basin").fl.sum()
    keep = fb[fb >= MIN_BASIN_FLOOD]
    view("pooled", main_rows)
    view(f"basin-weighted ({len(keep)} basins)", main_rows[main_rows.basin.isin(keep.index)], "basin", 1.0 / keep)
    for c in ["Africa", "Asia", "Oceania", "South America"]:
        view(c, main_rows[main_rows.continent == c])
    for ltn in crossval.LANDTYPES.values():
        view(f"{ltn} land", main_rows[main_rows.landtype == ltn])
    for g in ("normally wet", "other"):
        view(f"{g} cells", sk[sk["mode"] == g])
    tab = pd.DataFrame(out)
    tab.to_csv(res / "normally_wet_strata_skill.csv", index=False)

    fa = pd.concat(flags, ignore_index=True).groupby(["code", "product", "day"], as_index=False).flagged.sum()
    info = ev.set_index("code")
    erows = []
    for (code, prod), g in fa.groupby(["code", "product"]):
        o = crossval.event_level(g.set_index("day").flagged, sizes[code], info.loc[code, "start"], info.loc[code, "end"], e["min_share"], e["min_cells"], e["days_around_maps"], tuple(e["false_event_days"]))
        erows.append({"code": code, "product": prod, **o})
    el = pd.DataFrame(erows); el = el[el.eligible.astype(bool)]
    el["caught_clean"] = el.caught.astype(bool) & ~el.false_event.astype(bool)
    es = el.groupby("product").agg(events=("code", "nunique"), caught=("caught", "mean"), false_event=("false_event", "mean"), caught_clean=("caught_clean", "mean")).reset_index()
    es.to_csv(res / "normally_wet_strata_event_level.csv", index=False)
    rr = []
    for name, a in rel.items():
        rr.append(validate.summarise_reliability(pd.DataFrame({"stratum": name, "bin": range(bins), "n": a[0], "sum_p": a[1], "n_flood": a[2]}).query("n > 0"), bins))
    rs = pd.concat(rr, ignore_index=True).rename(columns={"stratum": "model"})
    rs.to_csv(res / "normally_wet_strata_reliability.csv", index=False)
    pd.set_option("display.width", 230)
    print(tab[["view", "product", "recall", "precision", "f1", "f1_lo", "f1_hi", "minus_built", "minus_built_lo", "minus_built_hi"]].round(3).to_string(index=False), "\n")
    print(es.round(3).to_string(index=False), "\n")
    print(rs[["model", "reliability_slope_wls", "mean_p", "obs_freq"]].round(3).to_string(index=False))
    r6 = pd.concat(rates)
    print(r6.groupby(["source", "stratum"]).agg(sens=("sens", "mean"), fa=("spec", lambda x: 1 - x.mean())).round(4).to_string())


if __name__ == "__main__":
    main()
