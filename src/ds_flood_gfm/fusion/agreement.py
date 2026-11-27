"""Agreement accuracy table, its comparison with gold, and the error-dependence check (plan v2 Phase 3, ADR 0002).

Agreement table: unsupervised two-class Dawid-Skene on the random global sample (no gold),
per stratum, from several starts; the best log-likelihood wins and flood is the minority class.
With three sources and two classes the model is exactly identified, so it rests entirely on
the sources erring independently given the truth. Agreement data cannot test that; gold can,
and `error_dependence` does.

Comparison: per source x stratum, on the engine's own scale, the evidence weights
w1 = log(sens / (1 - spec)) (a flood report) and w0 = log((1 - sens) / spec) (a dry report).
Status, in order:
  no_gold_to_check    the gold row is not a gold estimate (the thin snow strata, D-24)
  agreement_unstable  EM did not converge, another start reached a different solution with a
                      near-equal likelihood, or a source sits at a bound / is no better than chance
  agree               both weights within `weight_tolerance` (log 1.5 by default)
  diverge             otherwise
The agreement values are used for a stratum only when every source in it is `agree` (they come
from one joint fit); otherwise the stratum keeps its gold rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ds_flood_gfm.fusion import accuracy
from ds_flood_gfm.fusion.metrics import MODEL_SOURCES

NODATA = accuracy.NODATA


def orient(r: accuracy.EMResult) -> accuracy.EMResult:
    """Label the minority class as flood. Swapping the classes maps sens <-> 1-spec and prev <-> 1-prev."""
    if r.prevalence <= 0.5:
        return r
    return accuracy.EMResult(1 - r.spec, 1 - r.sens, 1 - r.prevalence, r.iterations, r.converged)


def fit(pat: np.ndarray, n: np.ndarray, starts: list[tuple], a: dict) -> dict:
    """Unsupervised EM from each start; returns the best fit and how the other starts compare."""
    fits = []
    for s0, p0, q0 in starts:
        r = accuracy.dawid_skene(pat, n, np.zeros((0, pat.shape[1]), "int64"), np.zeros(0), np.zeros(0),
                                 np.asarray(s0, float), np.asarray(p0, float), float(q0), max_iter=a["max_iter"],
                                 tol=a["tol"], prev_clip=(a["prev_clip_low"], 1 - a["prev_clip_low"]))
        r = orient(r)
        fits.append((accuracy.observed_loglik(pat, n, r.sens, r.spec, r.prevalence), r))
    fits.sort(key=lambda x: -x[0])
    ll, best = fits[0]
    # another start within `start_ll_tolerance` log-units of the best, at different parameters,
    # means the likelihood is flat: the data do not pick one solution
    rivals = [r for l2, r in fits[1:] if ll - l2 <= a["start_ll_tolerance"]
              and max(np.abs(r.sens - best.sens).max(), np.abs(r.spec - best.spec).max()) > a["param_tolerance"]]
    at_bound = bool(((best.sens < a["boundary"]) | (best.sens > 1 - a["boundary"])).any())
    no_better_than_chance = bool((best.sens <= 1 - best.spec).any())
    return {"result": best, "loglik": ll, "ll_gaps": [round(ll - l2, 3) for l2, _ in fits[1:]], "rival_solutions": len(rivals),
            "at_bound": at_bound, "no_better_than_chance": no_better_than_chance,
            "unstable": (not best.converged) or bool(rivals) or at_bound or no_better_than_chance}


def agreement_table(samp_pat: pd.DataFrame, gold_tab: pd.DataFrame, a: dict) -> pd.DataFrame:
    """One row per source x stratum: agreement-only sens/spec/prevalence and the fit diagnostics."""
    rows = []
    for st, sp in samp_pat.groupby("stratum"):
        agg = sp.groupby(MODEL_SOURCES, as_index=False).n.sum()
        pat, n = agg[MODEL_SOURCES].values.astype("int64"), agg.n.values.astype(float)
        if not (pat != NODATA).any():
            raise ValueError(f"{st}: the sample has no observations at all; nothing to fit")
        g = gold_tab[gold_tab.stratum == st].set_index("source").reindex(MODEL_SOURCES)
        starts = [([0.5] * 3, [0.95] * 3, 0.01), ([0.8] * 3, [0.99] * 3, 0.001)]
        if g[["sens", "spec"]].notna().all().all():
            starts.append((list(g.sens), list(g.spec), 0.002))
        f = fit(pat, n, starts, a)
        r = f["result"]
        for j, s in enumerate(MODEL_SOURCES):
            o = pat[:, j] != NODATA
            rows.append({"source": s, "stratum": st, "sens": r.sens[j], "spec": r.spec[j], "prevalence": r.prevalence,
                         "flag_rate": n[o & (pat[:, j] == 1)].sum() / n[o].sum() if n[o].sum() else np.nan,
                         "n_cell_days": int(n.sum()), "n_observed": int(n[o].sum()), "n_blocks": int(sp.block_id.nunique()),
                         "converged": r.converged, "iterations": r.iterations, "loglik": f["loglik"], "other_starts_ll_gap": str(f["ll_gaps"]),
                         "rival_solutions": f["rival_solutions"], "at_bound": f["at_bound"],
                         "no_better_than_chance": f["no_better_than_chance"], "unstable": f["unstable"]})
    return pd.DataFrame(rows)


def evidence_weights(sens, spec) -> tuple[np.ndarray, np.ndarray]:
    sens, spec = np.asarray(sens, float), np.asarray(spec, float)
    return np.log(sens / (1 - spec)), np.log((1 - sens) / spec)


def compare(agr: pd.DataFrame, gold_tab: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Agreement vs gold per source x stratum, with the status and the direction of each difference."""
    g = gold_tab[["source", "stratum", "method", "sens", "spec", "sens_lo", "sens_hi", "spec_lo", "spec_hi"]].rename(
        columns={c: f"gold_{c}" for c in ["method", "sens", "spec", "sens_lo", "sens_hi", "spec_lo", "spec_hi"]})
    m = agr[["source", "stratum", "sens", "spec", "prevalence", "flag_rate", "unstable"]].rename(
        columns={"sens": "agr_sens", "spec": "agr_spec", "prevalence": "agr_prevalence", "unstable": "agr_unstable"}).merge(
        g, on=["source", "stratum"], how="left", validate="one_to_one")
    if m.gold_sens.isna().any():
        raise ValueError(f"no gold row for {m[m.gold_sens.isna()][['source', 'stratum']].values.tolist()}")
    w1a, w0a = evidence_weights(m.agr_sens, m.agr_spec)
    w1g, w0g = evidence_weights(m.gold_sens, m.gold_spec)
    m["gold_w_flood"], m["agr_w_flood"], m["gold_w_dry"], m["agr_w_dry"] = w1g, w1a, w0g, w0a
    m["d_w_flood"], m["d_w_dry"] = w1a - w1g, w0a - w0g
    status = np.where(m.gold_method != "gold", "no_gold_to_check",
                      np.where(m.agr_unstable, "agreement_unstable",
                               np.where((np.abs(m.d_w_flood) <= tolerance) & (np.abs(m.d_w_dry) <= tolerance), "agree", "diverge")))
    m["status"] = status
    # all three sources' values come from ONE latent class per stratum: if that class does not match
    # gold flood for some sources, a match for another is coincidence. Use the agreement values only
    # when every source in the stratum agrees.
    m["stratum_all_agree"] = m.groupby("stratum").status.transform(lambda s: bool((s == "agree").all()))
    m["table_used"] = np.where(m.stratum_all_agree, "agreement", "gold")
    m["direction"] = [
        f"agreement sens {'higher' if a > g else 'lower'} ({a:.3f} vs {g:.3f}); "
        f"agreement false-alarm rate {'lower' if sa > sg else 'higher'} ({1 - sa:.4f} vs {1 - sg:.4f})"
        for a, g, sa, sg in zip(m.agr_sens, m.gold_sens, m.agr_spec, m.gold_spec)]
    return m


def error_dependence(gold_pat: pd.DataFrame, strata: list[str] | None = None) -> pd.DataFrame:
    """Do two sources err together more than independence (the naive-Bayes assumption) predicts?

    Per stratum, truth class and source pair, over cells where both observed: an error is a flood
    report on gold-dry or a dry report on gold-flood. lift = P(both wrong) / (P(a wrong) P(b wrong)).
    lift_within_event uses each event's own error rates for the expectation, so it is ~1 when the
    pooled lift comes only from some events being wetter (or worse) than others for every source.
    """
    rows = []
    pairs = [(a, b) for i, a in enumerate(MODEL_SOURCES) for b in MODEL_SOURCES[i + 1:]]
    for (st, z), d in gold_pat.groupby(["stratum", "truth"]):
        if strata is not None and st not in strata:
            continue
        wrong = 1 if z == 0 else 0
        for a, b in pairs:
            both = d[(d[a] != NODATA) & (d[b] != NODATA)]
            N = float(both.n.sum())
            if N == 0:
                continue
            ka = float(both[both[a] == wrong].n.sum())
            kb = float(both[both[b] == wrong].n.sum())
            kab = float(both[(both[a] == wrong) & (both[b] == wrong)].n.sum())
            exp_within = 0.0
            for _, e in both.groupby("code"):
                Ne = float(e.n.sum())
                exp_within += float(e[e[a] == wrong].n.sum()) * float(e[e[b] == wrong].n.sum()) / Ne
            rows.append({"stratum": st, "truth": "dry" if z == 0 else "flood", "pair": f"{a}+{b}",
                         "n_both_observed": int(N), "n_events": int(both.code.nunique()),
                         "p_a_wrong": ka / N, "p_b_wrong": kb / N, "p_both_wrong": kab / N, "n_both_wrong": int(kab),
                         "lift": kab * N / (ka * kb) if ka * kb else np.nan,
                         "lift_within_event": kab / exp_within if exp_within else np.nan})
    return pd.DataFrame(rows)
