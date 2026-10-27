"""Beat the cascade? Learned evidence per combination, the decision threshold and day-to-day memory (Next steps 9).

Design, fixed before running (2026-10-16), 2 x 2 x 2 variants of fusion plus the cascade:
  - evidence: "independent" (each source's rate, as built) or "pattern" (learned per combination of the
    reports that exist, per land type; fusion/pattern.py; shrinkage alpha = 100; missingness carries no
    evidence);
  - memory: decay 0.9 (as built) or 0 (each day on its own, like the cascade);
  - threshold: 0.5 (as built) or chosen inside each fold: the value in {0.02, 0.05, 0.1, 0.2, ..., 0.9}
    with the highest pooled F1 on the training folds' map days, P computed with that fold's own model (a
    small in-sample optimism, stated). Pooled F1 is the reference score, so the choice is not switched to
    basin-weighted afterwards even if it helps there less.
  - Everything else as in make crossval: folds, prior, adopted dry-window rule, floodable cells.
  - Views as fixed for the cascade comparison: pooled (reference), basin-weighted (basins >= 50 gold
    flooded cell-days), normally-wet cells not scored, per continent, per land type; event level (ADR 0007)
    and the reliability slope on map days. Paired intervals against the cascade and against fusion as built.
  - Success, fixed now: a variant beats the cascade only if it is above it on pooled AND basin-weighted F1
    AND has a lower false-event rate than the cascade (64%). F1, false events and slope are always reported
    together.
  - Check: "pattern, no memory, threshold in fold" can represent the cascade's rule, so on the training
    folds its F1 should be at least the cascade's; if not, the fit or the threshold grid is wrong.
Expectations, written before running: pattern evidence at 0.9 and 0.5 changes F1 a little and leaves the
probabilities flat; a lower threshold raises F1 and false events; no memory moves towards the cascade,
with a better slope and fewer gap-filled days. If so, the finding is that memory, not the combination
rule, separates fusion from the cascade.

Outputs, results/cv/: pattern_model_skill.csv, pattern_model_diff.csv, pattern_model_event_level.csv,
pattern_model_thresholds.csv. Cache: data/fusion/metrics/cv_pattern_counts.parquet.
Run: nice -n 19 uv run python -u scripts/fusion/cv_pattern_model.py
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import baseline, crossval, cube, engine, gold, pattern, validate
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA

ALPHA = 100
GRID_BASE = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # the pre-registered grid
# follow-up (2026-10-16, after the run): every fold chose 0.02, the grid's floor, and the check below failed,
# the pre-registered sign that the grid was wrong. --extended-grid adds lower thresholds; labelled as such.
GRID_EXT = [0.001, 0.002, 0.005, 0.01, *GRID_BASE]
EDGES = np.array([0, 0.0005, *GRID_EXT, 1.0 + 1e-9])  # histogram bins: every threshold is a bin edge
NB = len(EDGES) - 1
IDX = {th: int(np.where(np.isclose(EDGES, th))[0][0]) for th in GRID_EXT}
DECILE = np.minimum((EDGES[:-1] * 10 + 1e-9).astype(int), 9)  # bin -> reliability decile
COMBOS = [("independent", 0.9), ("independent", 0.0), ("pattern", 0.9), ("pattern", 0.0)]
MIN_BASIN_FLOOD = 50
CASCADE = "cascade: VIIRS, else GFM, else microwave"
BUILT = "independent, memory 0.9, threshold 0.5"  # fusion as built


def vname(evid, decay, thr):
    return f"{evid}, memory {decay:g}, threshold {'0.5' if thr == 'fixed' else 'in fold'}"


def hist(p, w=None):
    b = np.clip(np.searchsorted(EDGES, p, side="right") - 1, 0, NB - 1)
    return np.bincount(b, weights=w, minlength=NB)


def cascade(v):
    out = np.full(v["viirs"].shape, NODATA, "uint8")
    for s in ["microwave", "gfm", "viirs"]:  # last assignment wins: VIIRS first
        out = np.where(v[s] != NODATA, v[s], out)
    return out == 1


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score the first N events only (smoke test; fitting uses all)")
    ap.add_argument("--extended-grid", action="store_true", help="follow-up: thresholds down to 0.001 (outputs *_grid_extended.csv)")
    ap.add_argument("--sweep", action="store_true", help="also score every threshold for every version (results/cv/pattern_model_sweep.csv)")
    args = ap.parse_args()
    grid = GRID_EXT if args.extended_grid else GRID_BASE
    sfx = "_grid_extended" if args.extended_grid else ""
    cfg = load_config()
    f, e = cfg["fusion"], cfg["event_level"]
    reps, rule = cfg["accuracy"]["bootstrap_reps"], cfg["gold"]["dry_window_rule"]
    res = cfg.results_dir / "cv" if not args.limit else cfg.local("metrics", "smoke_pattern")
    folds = pd.read_csv(cfg.results_dir / "event_folds.csv")
    ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet"))
    ev = ev[ev.code.isin(folds.code)].merge(folds[["code", "fold", "basin_id"]], on="code").reset_index(drop=True)
    K = sorted(ev.fold.unique())
    t0 = time.time()

    # pattern counts (cached), and the fitted tables per fold
    cpath = cfg.local("metrics", "cv_pattern_counts.parquet")
    if cpath.exists() and set(pd.read_parquet(cpath).code) == set(ev.code):
        pc = pd.read_parquet(cpath)
    else:
        rows = []
        for _, r in ev.iterrows():
            for ew in gold.event_windows(r):
                ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
                rows.append(pattern.pattern_counts(ds, crossval.dry_truth(ds, rule), r.code))
        pc = pd.concat(rows).groupby(["code", "landtype", "pattern"], as_index=False)[["n_flood", "n_dry"]].sum()
        pc.to_parquet(cpath)
    print(f"pattern counts ready {time.time() - t0:.0f}s", flush=True)
    counts = pd.read_parquet(cfg.local("metrics", "cv_fit_counts.parquet"))
    counts = counts[counts.rule == rule]
    prior = pd.read_csv(res / "prior.csv")
    clip = tuple(f["sens_spec_clip"])
    rates = {k: crossval.fit_table(counts, ev.code[ev.fold != k]) for k in K}
    lrs = {k: pattern.fit_log_lr(pc, ev.code[ev.fold != k], ALPHA) for k in K}
    models = {(k, d): engine.Model.from_tables(rates[k], prior, MODEL_SOURCES, d, clip) for k in K for d in (0.9, 0.0)}

    train_h = defaultdict(lambda: np.zeros((2, NB)))  # (combo, fold) -> flood / dry histograms on training map days
    cas_train = defaultdict(lambda: np.zeros(3))  # fold -> cascade tp, fp, fn on its training events
    score = []  # per event, group: histograms (scoring fold only)
    fp_hist = defaultdict(lambda: np.zeros(NB))  # (code, combo, day) -> footprint P histogram
    cas_rows, cas_fp, sizes = [], defaultdict(int), defaultdict(int)
    rel = defaultdict(lambda: np.zeros((3, NB)))  # combo -> n, sum_p, n_flood on map days
    n_done = 0
    ev_run = ev.head(args.limit) if args.limit else ev
    for _, r in ev_run.iterrows():
        for ew in gold.event_windows(r):
            ds = xr.load_dataset(cube.cube_path(cfg, ew.key), engine="netcdf4")
            days = pd.to_datetime(ds.day.values)
            gd = crossval.gold_day_mask(ds)
            t, dom, st = ds.truth.values, ds.domain.values.astype(bool), ds.stratum.values
            lt = crossval.landtype_codes(st)
            obs = {s: ds[s].values for s in MODEL_SOURCES}
            lab = ((t == gold.DRY) | (t == gold.FLOOD)) & gd[:, None, None] & dom[None] & (lt != NODATA)
            fl_mask, dr_mask = lab & (t == gold.FLOOD), lab & (t == gold.DRY)
            nw = np.zeros(t.shape, bool)
            for i in np.where(gd)[0]:
                nw[i] = baseline.normally_wet(cfg, ew.window, days[i].month)
            fp = (t[gd] == gold.FLOOD).any(axis=0) & dom & (lt[0] != NODATA)
            sizes[r.code] += int(fp.sum())
            cas = cascade({s: ds[s].values for s in MODEL_SOURCES})
            for k in K:
                if k != r.fold:
                    cas_train[k] += [(cas & fl_mask).sum(), (cas & dr_mask).sum(), (~cas & fl_mask).sum()]
            for g, gm in (("normally wet", nw), ("other", ~nw)):
                for kk, ltn in crossval.LANDTYPES.items():
                    m = gm & (lt == kk)
                    cas_rows.append({"code": r.code, "group": g, "landtype": ltn, "tp": int((cas & fl_mask & m).sum()),
                                     "fp": int((cas & dr_mask & m).sum()), "fn": int((~cas & fl_mask & m).sum())})
            for di, d in enumerate(days):
                cas_fp[(r.code, d)] += int(cas[di][fp].sum())
            for evid, decay in COMBOS:
                for k in K:
                    if evid == "independent":
                        p = engine.run_sequence(obs, st, dom, models[(k, decay)])["p_flood"]
                    else:
                        p = pattern.run_sequence(obs, st, dom, lrs[k], models[(k, decay)])
                    pf = np.nan_to_num(p, nan=0.0)
                    if k != r.fold:
                        train_h[(evid, decay, k)] += [hist(pf[fl_mask]), hist(pf[dr_mask])]
                        continue
                    for g, gm in (("normally wet", nw), ("other", ~nw)):
                        for kk, ltn in crossval.LANDTYPES.items():
                            m = gm & (lt == kk)
                            score.append({"code": r.code, "evid": evid, "decay": decay, "group": g, "landtype": ltn,
                                          "hf": hist(pf[fl_mask & m]), "hd": hist(pf[dr_mask & m])})
                    a = rel[(evid, decay)]
                    a[0] += hist(pf[lab])
                    a[1] += hist(pf[lab], pf[lab])
                    a[2] += hist(pf[fl_mask])
                    for di, d in enumerate(days):
                        fp_hist[(r.code, evid, decay, d)] += hist(pf[di][fp])
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done} windows, {time.time() - t0:.0f}s", flush=True)

    # thresholds chosen inside each fold, on its training map days
    thr_rows, chosen = [], {}
    for evid, decay in COMBOS:
        for k in K:
            hf, hd = train_h[(evid, decay, k)]
            f1s = {th: f1(hf[IDX[th]:].sum(), hd[IDX[th]:].sum(), hf[:IDX[th]].sum()) for th in grid}
            th = max(f1s, key=f1s.get)
            chosen[(evid, decay, k)] = IDX[th]
            thr_rows.append({"evidence": evid, "memory": decay, "fold": int(k), "threshold": th, "training_f1": f1s[th],
                             "training_f1_at_0.5": f1s[0.5], "cascade_training_f1": f1(*cas_train[k])})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(res / f"pattern_model_thresholds{sfx}.csv", index=False)

    # per-event counts for every variant
    fold_of = ev.set_index("code").fold
    rows = []
    for s in score:
        for tk in ("fixed", "fold"):
            j = IDX[0.5] if tk == "fixed" else chosen[(s["evid"], s["decay"], fold_of[s["code"]])]
            rows.append({"code": s["code"], "group": s["group"], "landtype": s["landtype"], "product": vname(s["evid"], s["decay"], tk),
                         "tp": s["hf"][j:].sum(), "fp": s["hd"][j:].sum(), "fn": s["hf"][:j].sum()})
    sk = pd.concat([pd.DataFrame(rows), pd.DataFrame(cas_rows).assign(product=CASCADE)], ignore_index=True)
    sk = sk.groupby(["code", "group", "landtype", "product"], as_index=False)[["tp", "fp", "fn"]].sum()
    info = ev.set_index("code")
    sk["continent"], sk["basin"] = sk.code.map(info.continent), sk.code.map(info.basin_id)
    products = sorted(sk["product"].unique(), key=lambda x: (x != BUILT, x != CASCADE, x))

    rng = np.random.default_rng(20261016)

    def view(name, df, unit="code", weights=None):
        d = df.copy()
        if weights is not None:
            for c in ("tp", "fp", "fn"):
                d[c] = d[c] * d[unit].map(weights)
        m = d.groupby([unit, "product"])[["tp", "fp", "fn"]].sum().unstack("product").fillna(0)
        idx = rng.integers(0, len(m), (reps, len(m)))
        mats = {p: np.stack([m[("tp", p)], m[("fp", p)], m[("fn", p)]], 1).astype(float) for p in products}
        F = {p: (lambda x: 2 * x[..., 0] / (2 * x[..., 0] + x[..., 1] + x[..., 2]))(mats[p][idx].sum(1)) for p in products}
        out = []
        for p in products:
            s = mats[p].sum(0)
            out.append({"view": name, "product": p, "units": len(m), "recall": s[0] / (s[0] + s[2]), "precision": s[0] / (s[0] + s[1]),
                        "f1": f1(*s), "f1_lo": np.nanquantile(F[p], 0.025), "f1_hi": np.nanquantile(F[p], 0.975),
                        "minus_cascade": f1(*s) - f1(*mats[CASCADE].sum(0)),
                        "minus_cascade_lo": np.nanquantile(F[p] - F[CASCADE], 0.025), "minus_cascade_hi": np.nanquantile(F[p] - F[CASCADE], 0.975),
                        "minus_built": f1(*s) - f1(*mats[BUILT].sum(0)),
                        "minus_built_lo": np.nanquantile(F[p] - F[BUILT], 0.025), "minus_built_hi": np.nanquantile(F[p] - F[BUILT], 0.975)})
        return out

    flood_by_basin = sk[sk["product"] == CASCADE].assign(fl=lambda x: x.tp + x.fn).groupby("basin").fl.sum()
    keep = flood_by_basin[flood_by_basin >= MIN_BASIN_FLOOD]
    out = view("pooled", sk)
    out += view(f"basin-weighted ({len(keep)} basins)", sk[sk.basin.isin(keep.index)], unit="basin", weights=1.0 / keep)
    out += view("normally-wet cells not scored", sk[sk.group == "other"])
    for c in ["Africa", "Asia", "Oceania", "South America"]:
        out += view(c, sk[sk.continent == c])
    for ltn in crossval.LANDTYPES.values():
        out += view(f"{ltn} land", sk[sk.landtype == ltn])
    tab = pd.DataFrame(out)
    tab.to_csv(res / f"pattern_model_skill{sfx}.csv", index=False)

    # event level and reliability
    erows = []
    days_by_code = defaultdict(list)
    for (code, evid, decay, d) in fp_hist:
        days_by_code[(code, evid, decay)].append(d)
    for (code, evid, decay), ds_ in days_by_code.items():
        ds_ = sorted(set(ds_))
        for tk in ("fixed", "fold"):
            j = IDX[0.5] if tk == "fixed" else chosen[(evid, decay, fold_of[code])]
            s = pd.Series([fp_hist[(code, evid, decay, d)][j:].sum() for d in ds_], index=ds_)
            o = crossval.event_level(s, sizes[code], info.loc[code, "start"], info.loc[code, "end"], e["min_share"], e["min_cells"],
                                     e["days_around_maps"], tuple(e["false_event_days"]))
            erows.append({"code": code, "product": vname(evid, decay, tk), **o})
    for code in ev_run.code:
        ds_ = sorted(d for (c, d) in cas_fp if c == code)
        s = pd.Series([cas_fp[(code, d)] for d in ds_], index=ds_)
        o = crossval.event_level(s, sizes[code], info.loc[code, "start"], info.loc[code, "end"], e["min_share"], e["min_cells"],
                                 e["days_around_maps"], tuple(e["false_event_days"]))
        erows.append({"code": code, "product": CASCADE, **o})
    el = pd.DataFrame(erows)
    el = el[el.eligible.astype(bool)]
    el["caught_clean"] = el.caught.astype(bool) & ~el.false_event.astype(bool)
    es = el.groupby("product").agg(events=("code", "nunique"), caught=("caught", "mean"), false_event=("false_event", "mean"),
                                   caught_clean=("caught_clean", "mean"), median_days=("days_to_detection", "median")).reset_index()
    rr = []
    for (evid, decay), a in rel.items():
        n10 = np.stack([np.bincount(DECILE, weights=a[i], minlength=10) for i in range(3)])
        df = pd.DataFrame({"stratum": vname(evid, decay, "fixed").split(", threshold")[0], "bin": range(10), "n": n10[0], "sum_p": n10[1], "n_flood": n10[2]})
        rr.append(validate.summarise_reliability(df[df.n > 0], 10))
    rs = pd.concat(rr, ignore_index=True).rename(columns={"stratum": "model"})
    es.to_csv(res / f"pattern_model_event_level{sfx}.csv", index=False)
    rs.to_csv(res / f"pattern_model_reliability{sfx}.csv", index=False)

    if args.sweep:  # the same threshold in every fold, for every version: the trade-off between F1 and false events
        sweep = []
        bw_keep = set(keep.index)
        for evid, decay in COMBOS:
            for th in GRID_EXT:
                j = IDX[th]
                c = pd.DataFrame([{"code": x["code"], "tp": x["hf"][j:].sum(), "fp": x["hd"][j:].sum(), "fn": x["hf"][:j].sum()}
                                  for x in score if x["evid"] == evid and x["decay"] == decay]).groupby("code").sum()
                c["basin"] = c.index.map(info.basin_id)
                b = c[c.basin.isin(bw_keep)]
                w = b.basin.map(1.0 / keep)
                ers = []
                for (code, ev_, dc), ds_ in days_by_code.items():
                    if ev_ != evid or dc != decay:
                        continue
                    ds_ = sorted(set(ds_))
                    s_ = pd.Series([fp_hist[(code, evid, decay, d)][j:].sum() for d in ds_], index=ds_)
                    ers.append(crossval.event_level(s_, sizes[code], info.loc[code, "start"], info.loc[code, "end"], e["min_share"],
                                                    e["min_cells"], e["days_around_maps"], tuple(e["false_event_days"])))
                er = pd.DataFrame(ers)
                er = er[er.eligible.astype(bool)]
                sweep.append({"evidence": evid, "memory": decay, "threshold": th, "pooled_f1": f1(c.tp.sum(), c.fp.sum(), c.fn.sum()),
                              "pooled_recall": c.tp.sum() / (c.tp.sum() + c.fn.sum()), "pooled_precision": c.tp.sum() / (c.tp.sum() + c.fp.sum()),
                              "basin_weighted_f1": f1((b.tp * w).sum(), (b.fp * w).sum(), (b.fn * w).sum()),
                              "caught": er.caught.astype(bool).mean(), "false_event": er.false_event.astype(bool).mean(),
                              "caught_clean": (er.caught.astype(bool) & ~er.false_event.astype(bool)).mean(), "events": len(er)})
        cs = sk[sk["product"] == CASCADE].groupby("code")[["tp", "fp", "fn"]].sum()
        cs["basin"] = cs.index.map(info.basin_id)
        cb = cs[cs.basin.isin(bw_keep)]
        wc = cb.basin.map(1.0 / keep)
        ce = el[el["product"] == CASCADE]
        sweep.append({"evidence": "cascade", "memory": np.nan, "threshold": np.nan, "pooled_f1": f1(cs.tp.sum(), cs.fp.sum(), cs.fn.sum()),
                      "pooled_recall": cs.tp.sum() / (cs.tp.sum() + cs.fn.sum()), "pooled_precision": cs.tp.sum() / (cs.tp.sum() + cs.fp.sum()),
                      "basin_weighted_f1": f1((cb.tp * wc).sum(), (cb.fp * wc).sum(), (cb.fn * wc).sum()),
                      "caught": ce.caught.astype(bool).mean(), "false_event": ce.false_event.astype(bool).mean(),
                      "caught_clean": ce.caught_clean.mean(), "events": len(ce)})
        pd.DataFrame(sweep).to_csv(cfg.results_dir / "cv" / "pattern_model_sweep.csv", index=False)
        print(pd.DataFrame(sweep).round(3).to_string(index=False))

    pd.set_option("display.width", 250)
    print(thr.round(3).to_string(index=False), "\n")
    cols = ["view", "product", "recall", "precision", "f1", "f1_lo", "f1_hi", "minus_cascade", "minus_cascade_lo", "minus_cascade_hi", "minus_built"]
    print(tab[tab.view.isin(["pooled", f"basin-weighted ({len(keep)} basins)", "normally-wet cells not scored"])][cols].round(3).to_string(index=False), "\n")
    print(es.round(3).to_string(index=False), "\n")
    print(rs[["model", "reliability_slope_wls", "mean_p", "obs_freq"]].round(3).to_string(index=False))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
