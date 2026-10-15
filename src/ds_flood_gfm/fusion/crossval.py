"""Cross-validated re-run (ADRs 0003-0007): every in-scope event is scored once, by a model fitted on
the other four folds.

The pieces, in the order scripts/fusion/phase5_crossval.py uses them:
  dry_truth          an event cube's truth under a dry-window rule (ADR 0006)
  fit_counts         per event, source and land type: confusion counts on labelled floodable cells the
                     source observed (map days give hit rates; map days plus dry days, false-alarm rates)
  fit_table          Jeffreys rates from the training events' counts, written to the six stratum codes
                     the engine indexes by (ADR 0005)
  prior_table        base flood rate per land type from the in-scope sample blocks (D-35, ADR 0005)
  score_rows         map-day counts for fusion and each product, per land type and Sentinel-1 flag
  reliability_rows   binned P against gold: map days only, and map days plus dry days
  footprint_flags    per day, how many gold-footprint cells each product flags
  event_level        caught / days to detection / false event for one event (ADR 0007)

Land type: the cubes keep v1's 6-class stratum code (landcover * 2 + snow); land type is code // 2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from ds_flood_gfm.fusion import gold
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES
from ds_flood_gfm.fusion.sources import NODATA
from ds_flood_gfm.fusion.static import LANDCOVER_NAMES as LANDTYPES
from ds_flood_gfm.fusion.validate import SINGLE, _conf

COUNT4 = ["tp", "fn", "fp", "tn"]


def landtype_codes(stratum: np.ndarray) -> np.ndarray:
    """0 open, 1 vegetated, 2 urban, 255 unknown, from the 6-class stratum code."""
    s = np.asarray(stratum)
    return np.where(s == NODATA, NODATA, s // 2).astype("uint8")


def gold_day_mask(ds: xr.Dataset) -> np.ndarray:
    days = pd.to_datetime(ds.day.values)
    return np.asarray((days >= pd.Timestamp(ds.attrs["event_start"])) & (days <= pd.Timestamp(ds.attrs["event_end"])))


def dry_truth(ds: xr.Dataset, rule: str) -> np.ndarray:
    """Truth (day, lat, lon) under a dry-window rule. Map days are the same under both rules."""
    if rule == "corrected":
        return gold.corrected_dry_windows(ds)[0]
    if rule == "literal":
        return ds.truth.values
    raise ValueError(f"unknown dry-window rule {rule!r}: expected 'corrected' or 'literal'")


def fit_counts(ds: xr.Dataset, truth: np.ndarray, code: str) -> pd.DataFrame:
    """Counts per (day type, land type, source) over labelled floodable cells the source observed.

    Only floodable-domain cells count: the model is fitted where it predicts.
    """
    gd = gold_day_mask(ds)
    dom = ds.domain.values.astype(bool)[None]
    lt = landtype_codes(ds.stratum.values)
    rows = []
    for kind, sel in (("map", gd), ("dry", ~gd)):
        if not sel.any():
            continue
        t, ll = truth[sel], lt[sel]
        lab = ((t == gold.DRY) | (t == gold.FLOOD)) & dom & (ll != NODATA)
        for s in MODEL_SOURCES:
            v = ds[s].values[sel]
            obs = lab & (v != NODATA)
            for k, name in LANDTYPES.items():
                m = obs & (ll == k)
                fl, tf = v[m] == 1, t[m] == gold.FLOOD
                rows.append({"code": code, "day_type": kind, "landtype": name, "source": s,
                             "tp": int((fl & tf).sum()), "fn": int((~fl & tf).sum()),
                             "fp": int((fl & ~tf).sum()), "tn": int((~fl & ~tf).sum())})
    return pd.DataFrame(rows)


def fit_table(counts: pd.DataFrame, train_codes, sources=MODEL_SOURCES) -> pd.DataFrame:
    """Hit and false-alarm rates per source and land type from the training events only.

    sens = (TP + 0.5) / (TP + FN + 1) and spec = (TN + 0.5) / (TN + FP + 1) (Jeffreys, D-32), from map
    days and dry days pooled (flood labels exist only on map days). Each row is written to both snow
    codes of its land type (2k, 2k + 1), the codes engine.Model indexes by. A source with no observed
    gold flood (or dry) cell in a land type has no estimate: that raises and names it.
    """
    train = set(train_codes)
    c = counts[counts.code.isin(train) & counts.source.isin(sources)]
    g = c.groupby(["source", "landtype"])[COUNT4].sum()
    rows, empty = [], []
    for s in sources:
        for k, name in LANDTYPES.items():
            r = g.loc[(s, name)] if (s, name) in g.index else pd.Series(0, index=COUNT4)
            if r.tp + r.fn == 0 or r.tn + r.fp == 0:
                empty.append(f"{s}/{name} (flood {int(r.tp + r.fn)}, dry {int(r.tn + r.fp)})")
                continue
            sens = (r.tp + 0.5) / (r.tp + r.fn + 1.0)
            spec = (r.tn + 0.5) / (r.tn + r.fp + 1.0)
            for code in (2 * k, 2 * k + 1):
                rows.append({"source": s, "landtype": name, "stratum_code": code, "sens": sens, "spec": spec,
                             "n_flood_obs": int(r.tp + r.fn), "n_dry_obs": int(r.tn + r.fp)})
    if empty:
        raise ValueError(f"no observed gold cells to fit (training events {len(train)}): {empty}")
    return pd.DataFrame(rows)


def prior_table(samp_pat: pd.DataFrame, block_ids) -> pd.DataFrame:
    """Base flood rate per land type: GFM flood share of GFM-observed sample cell-days (D-35), from the
    given sample blocks, pooled over snow. Written to both snow codes of each land type."""
    sp = samp_pat[samp_pat.block_id.isin(set(block_ids))]
    sp = sp.assign(landtype=sp.stratum.str.rsplit("_", n=1).str[0])
    rows = []
    for k, name in LANDTYPES.items():
        g = sp[(sp.landtype == name) & (sp.gfm != NODATA)]
        n_obs, n_fl = int(g.n.sum()), int(g[g.gfm == 1].n.sum())
        if n_obs == 0:
            raise ValueError(f"no GFM-observed sample cell-days in {name} land among {len(set(block_ids))} blocks")
        for code in (2 * k, 2 * k + 1):
            rows.append({"landtype": name, "stratum_code": code, "base_rate": n_fl / n_obs,
                         "n_gfm_observed_cell_days": n_obs, "n_gfm_flood_cell_days": n_fl})
    return pd.DataFrame(rows)


def score_rows(ds: xr.Dataset, p: np.ndarray, code: str, continent: str, s1_days: set, threshold: float,
               fusion_name: str = "fusion", include_single: bool = True) -> pd.DataFrame:
    """Map-day confusion counts per land type and Sentinel-1 flag, summed over the window's days.

    Cells: labelled floodable cells (P finite). 'all' mode: a product's no-data counts as not flagged.
    'conditional' mode: only cells the product observed, with fusion scored on the same cells.
    """
    t = ds.truth.values
    lt = landtype_codes(ds.stratum.values)
    days = pd.to_datetime(ds.day.values)
    rows = []
    for i in np.where(gold_day_mask(ds))[0]:
        lab = ((t[i] == gold.DRY) | (t[i] == gold.FLOOD)) & np.isfinite(p[i])
        if not lab.any():
            continue
        fl = np.where(np.isfinite(p[i]), p[i] >= threshold, False)
        base0 = {"code": code, "continent": continent, "label_s1": days[i].normalize() in s1_days}
        for k, name in LANDTYPES.items():
            m = lab & (lt[i] == k)
            if not m.any():
                continue
            base = base0 | {"landtype": name}
            rows.append(base | {"product": fusion_name, "mode": "all", **_conf(fl[m], t[i][m])})
            if not include_single:
                continue
            for s in SINGLE:
                v = ds[s].values[i]
                rows.append(base | {"product": s, "mode": "all", **_conf((v == 1)[m], t[i][m])})
                obs = m & (v != NODATA)
                if obs.any():
                    rows.append(base | {"product": s, "mode": "conditional", **_conf((v == 1)[obs], t[i][obs])})
                    rows.append(base | {"product": f"{fusion_name}|{s}", "mode": "conditional", **_conf(fl[obs], t[i][obs])})
    if not rows:
        return pd.DataFrame()
    keys = ["code", "continent", "landtype", "label_s1", "product", "mode"]
    return pd.DataFrame(rows).groupby(keys, as_index=False).sum(numeric_only=True)


def reliability_rows(p: np.ndarray, truth: np.ndarray, gd: np.ndarray, lt: np.ndarray, code: str, bins: int) -> pd.DataFrame:
    """Binned P against truth per land type, on two label sets: map days only; map days plus dry days."""
    out = []
    lab = ((truth == gold.DRY) | (truth == gold.FLOOD)) & np.isfinite(p)
    for labels, days in (("map days", gd), ("map and dry days", np.ones_like(gd))):
        sel = lab & days[:, None, None]
        for k, name in LANDTYPES.items():
            m = sel & (lt == k)
            if not m.any():
                continue
            pp, tt = p[m].astype("float64"), truth[m] == gold.FLOOD
            b = np.clip((pp * bins).astype(int), 0, bins - 1)
            n = np.bincount(b, minlength=bins)
            sp = np.bincount(b, weights=pp, minlength=bins)
            nf = np.bincount(b, weights=tt, minlength=bins)
            for j in np.where(n > 0)[0]:
                out.append({"code": code, "labels": labels, "landtype": name, "bin": int(j), "n": int(n[j]),
                            "sum_p": float(sp[j]), "n_flood": int(nf[j])})
    return pd.DataFrame(out)


def footprint_flags(ds: xr.Dataset, flags: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per day and land type ('all' included): how many footprint cells each product flags.

    Footprint: floodable cells flooded on any map day of the window. flags[product] is a boolean
    (day, lat, lon) array. Returns the long table and the footprint size per land type.
    """
    t = ds.truth.values
    fp = (t[gold_day_mask(ds)] == gold.FLOOD).any(axis=0) & ds.domain.values.astype(bool)
    lt0 = landtype_codes(ds.stratum.values[0])
    masks = {"all": fp & (lt0 != NODATA)} | {name: fp & (lt0 == k) for k, name in LANDTYPES.items()}
    days = pd.to_datetime(ds.day.values)
    rows = []
    for lname, m in masks.items():
        if not m.any():
            continue
        for prod, f in flags.items():
            rows.append(pd.DataFrame({"day": days, "landtype": lname, "product": prod, "flagged": f[:, m].sum(axis=1)}))
    sizes = {lname: int(m.sum()) for lname, m in masks.items()}
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["day", "landtype", "product", "flagged"])), sizes


def event_level(flagged: pd.Series, footprint_cells: int, first_map: pd.Timestamp, last_map: pd.Timestamp,
                min_share: float, min_cells: int, days_around: int, false_days: tuple[int, int]) -> dict:
    """ADR 0007 for one event, one product, one land type. `flagged` is indexed by day.

    need = max(min_cells, ceil(min_share * footprint)). Caught: some day in [first_map - days_around,
    last_map + days_around] reaches need; days_to_detection is the first such day minus first_map.
    False event: some day with offset in [false_days[0], false_days[1]] from first_map reaches need.
    Footprints under min_cells are not eligible (caught and false_event are None).
    """
    if footprint_cells < min_cells:
        return {"eligible": False, "need": None, "caught": None, "days_to_detection": np.nan, "false_event": None}
    need = max(min_cells, math.ceil(min_share * footprint_cells))
    idx = pd.to_datetime(flagged.index)
    off = np.asarray((idx - first_map).days)
    hit = flagged.values >= need
    in_event = (idx >= first_map - pd.Timedelta(days=days_around)) & (idx <= last_map + pd.Timedelta(days=days_around))
    caught = hit & np.asarray(in_event)
    lo, hi = false_days
    false_ev = hit & (off >= lo) & (off <= hi)
    first = off[caught].min() if caught.any() else np.nan
    return {"eligible": True, "need": need, "caught": bool(caught.any()), "days_to_detection": float(first),
            "false_event": bool(false_ev.any())}


def bootstrap_scores(per_event: pd.DataFrame, by: list[str], reps: int, seed: int) -> pd.DataFrame:
    """95% intervals for recall, precision and F1 per group, resampling whole events within the group.

    per_event: one row per (event, group) with summed tp, fp, fn. Resampling the group's own events
    (not all events) keeps the number of events per draw fixed for a continent or land type.
    """
    rng = np.random.default_rng(seed)
    keys = by or ["_all"]
    d = per_event.assign(_all="all") if not by else per_event
    out = []
    for key, g in d.groupby(keys):
        e = g.groupby("code")[["tp", "fp", "fn"]].sum().to_numpy(dtype=float)
        draws = rng.integers(0, len(e), size=(reps, len(e)))
        tp, fp, fn = e[draws].sum(axis=1).T
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = {"recall": tp / (tp + fn), "precision": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn)}
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for name, v in scores.items():
            v = v[np.isfinite(v)]
            row[f"{name}_lo"] = float(np.quantile(v, 0.025)) if len(v) else np.nan
            row[f"{name}_hi"] = float(np.quantile(v, 0.975)) if len(v) else np.nan
        row["n_events"] = len(e)
        out.append(row)
    res = pd.DataFrame(out)
    return res.drop(columns="_all") if not by else res
