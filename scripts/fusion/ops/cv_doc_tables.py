"""Markdown tables for the review doc and final_report.md, straight from results/cv/*.csv (no hand-typed numbers).

Run: uv run python cv_doc_tables.py <results/cv dir>
"""

import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1])
NAME = {"fusion": "fusion", "viirs": "VIIRS", "microwave": "microwave vote", "floodscan": "FloodScan", "gfds": "GFDS",
        "gfm": "GFM", "fusion_literal_dry": "fusion, old dry-day rule"}
SINGLE = ["viirs", "microwave", "floodscan", "gfds", "gfm"]


def f(x, d=2):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def ci(r, s):
    return f"{f(r[s])} ({f(r[s + '_lo'])}–{f(r[s + '_hi'])})"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


ov = pd.read_csv(R / "skill_overall.csv")
a = ov[ov["mode"] == "all"].set_index("product")
print("## overall, all cells\n")
rows = [{"product": NAME[p], "recall": ci(a.loc[p], "recall"), "precision": ci(a.loc[p], "precision"), "F1": ci(a.loc[p], "f1")}
        for p in a.sort_values("f1", ascending=False).index if p in NAME and p != "fusion_literal_dry"]
print(md(pd.DataFrame(rows)))
print(f"\nevents {int(a.loc['fusion','n_events'])}, gold flooded cell-days scored {int(a.loc['fusion','truth_flood']):,}")
lit = a.loc["fusion_literal_dry"]
print(f"\ndry-rule sensitivity: fusion adopted rule {f(a.loc['fusion','recall'])} · {f(a.loc['fusion','precision'])} (F1 {f(a.loc['fusion','f1'])});"
      f" old rule {f(lit.recall)} · {f(lit.precision)} (F1 {f(lit.f1)})")

c = ov[ov["mode"] == "conditional"].set_index("product")
print("\n## observed cells\n")
rows = []
for p in SINGLE:
    if p in c.index and f"fusion|{p}" in c.index:
        x, y = c.loc[p], c.loc[f"fusion|{p}"]
        rows.append({"cells": f"where {NAME[p]} observed", "product: recall · precision (F1)": f"{f(x.recall)} · {f(x.precision)} ({f(x.f1)})",
                     "fusion: recall · precision (F1)": f"{f(y.recall)} · {f(y.precision)} ({f(y.f1)})",
                     "share of gold flood seen": f"{(x.truth_flood / a.loc['fusion','truth_flood']):.0%}"})
print(md(pd.DataFrame(rows)))

for grp, fname in (("continent", "skill_by_continent.csv"), ("landtype", "skill_by_landtype.csv")):
    t = pd.read_csv(R / fname)
    t = t[t["mode"] == "all"]
    print(f"\n## F1 by {grp}, all cells (95% interval)\n")
    rows = []
    for g, gg in t.groupby(grp):
        gg = gg.set_index("product")
        r = {grp: f"{g} ({int(gg.loc['fusion','n_events'])} events; {int(gg.loc['fusion','truth_flood']):,} flooded cell-days)"}
        for p in ["fusion", *SINGLE]:
            r[NAME[p]] = ci(gg.loc[p], "f1") if p in gg.index else "–"
        best = max(SINGLE, key=lambda p: gg.loc[p, "f1"] if p in gg.index else -1)
        r["best single"] = NAME[best]
        rows.append(r)
    print(md(pd.DataFrame(rows)))
    print(f"\n### recall · precision by {grp}\n")
    rows = []
    for g, gg in t.groupby(grp):
        gg = gg.set_index("product")
        r = {grp: g}
        for p in ["fusion", *SINGLE]:
            r[NAME[p]] = f"{f(gg.loc[p,'recall'])} · {f(gg.loc[p,'precision'])}" if p in gg.index else "–"
        rows.append(r)
    print(md(pd.DataFrame(rows)))

print("\n## criteria\n")
cr = pd.read_csv(R / "criteria.csv")
cov = pd.read_csv(R / "coverage_summary.csv").set_index("what").coverage
print(md(pd.DataFrame([{"land type": r.landtype, "fusion CSI": f(r.fusion_csi), "best single (CSI)": f"{NAME[r.best_single]} ({f(r.best_single_csi)})",
                        "(b) passes": "yes" if r.b_pass else "no", "slope, map days": f(r.slope_map_days),
                        "slope, map and dry days": f(r.slope_map_and_dry_days), "(c) passes": "yes" if r.c_pass_map_days else "no"}
                       for r in cr.itertuples()])))
print("\ncoverage (share of floodable cell-days with a value): " + ", ".join(f"{k} {v:.1%}" for k, v in cov.items()))

rs = pd.read_csv(R / "reliability_summary.csv")
print("\n## reliability summary\n")
print(md(rs[["rule", "labels", "landtype", "n", "n_flood", "reliability_slope_wls", "mean_p", "obs_freq"]].round(3)))

ev = pd.read_csv(R / "event_level_summary.csv")
print("\n## event level, all land\n")
e = ev[(ev.group == "overall") & (ev.landtype == "all")].sort_values("caught", ascending=False)
print(md(pd.DataFrame([{"product": NAME.get(r["product"], r["product"]), "events": int(r.n_events),
                        "caught": f"{r.caught:.0%} ({r.caught_lo:.0%}–{r.caught_hi:.0%})",
                        "false event": f"{r.false_event:.0%} ({r.false_event_lo:.0%}–{r.false_event_hi:.0%})",
                        "caught, no false event": f"{r.caught_clean:.0%} ({r.caught_clean_lo:.0%}–{r.caught_clean_hi:.0%})",
                        "median days to detection": f(r.median_days_to_detection, 0)} for _, r in e.iterrows()])))
print("\n## event level by continent (caught / false event)\n")
e = ev[(ev.group == "by_continent") & (ev.landtype == "all")]
rows = []
for g, gg in e.groupby("continent"):
    gg = gg.set_index("product")
    r = {"continent": f"{g} ({int(gg.loc['fusion','n_events'])})"}
    for p in ["fusion", *SINGLE]:
        r[NAME[p]] = f"{gg.loc[p,'caught']:.0%} / {gg.loc[p,'false_event']:.0%}" if p in gg.index else "–"
    rows.append(r)
print(md(pd.DataFrame(rows)))
print("\n## event level by land type (caught / false event)\n")
e = ev[(ev.group == "overall") & (ev.landtype != "all")]
rows = []
for g, gg in e.groupby("landtype"):
    gg = gg.set_index("product")
    r = {"land type": f"{g} ({int(gg.loc['fusion','n_events'])})"}
    for p in ["fusion", *SINGLE]:
        r[NAME[p]] = f"{gg.loc[p,'caught']:.0%} / {gg.loc[p,'false_event']:.0%}" if p in gg.index else "–"
    rows.append(r)
print(md(pd.DataFrame(rows)))
per = pd.read_csv(R / "event_level_events.csv")
allp = per[(per.landtype == "all") & (per["product"] == "fusion")]
print(f"\neligible events: {int(allp.eligible.astype(bool).sum())} of {allp.code.nunique()}")

ns = pd.read_csv(R / "skill_non_s1.csv")
n = ns[ns["mode"] == "all"].set_index("product")
print("\n## non-Sentinel-1 map days, all cells\n")
print(md(pd.DataFrame([{"product": NAME[p], "recall · precision (F1)": f"{f(n.loc[p,'recall'])} · {f(n.loc[p,'precision'])} ({f(n.loc[p,'f1'])})",
                        "F1 interval": f"{f(n.loc[p,'f1_lo'])}–{f(n.loc[p,'f1_hi'])}"}
                       for p in n.sort_values("f1", ascending=False).index if p in NAME])))
print(f"\nnon-S1: {int(n.loc['fusion','n_events'])} events, {int(n.loc['fusion','truth_flood']):,} flooded cell-days")

rates = pd.read_csv(R / "rates_by_fold.csv")
r = rates[rates.rule == "corrected"]
print("\n## fitted rates, range over folds (adopted rule)\n")
g = r.groupby(["source", "landtype"]).agg(sens_min=("sens", "min"), sens_max=("sens", "max"), spec_min=("spec", "min"), spec_max=("spec", "max"),
                                          n_flood=("n_flood_obs", "min")).reset_index()
print(md(g.round(4)))
print("\nprior:\n" + md(pd.read_csv(R / "prior.csv").drop_duplicates("landtype")[["landtype", "base_rate", "n_gfm_observed_cell_days"]]))

NAME2 = NAME
wb = pd.read_csv(R / "skill_without_largest_basin.csv")
w = wb[wb["mode"] == "all"].sort_values("f1", ascending=False)
print(f"\n## without largest basin ({int(w.n_events.max())} events, {int(w.truth_flood.max()):,} flooded cell-days; basin share {w.excluded_share_of_gold_flood.iloc[0]:.0%})\n")
print(md(pd.DataFrame([{"product": NAME[r["product"]], "recall": f(r.recall), "precision": f(r.precision), "F1": f"{f(r.f1)} ({f(r.f1_lo)}–{f(r.f1_hi)})"}
                       for _, r in w.iterrows() if r["product"] != "fusion_literal_dry"])))
vp = pd.read_csv(R / "skill_by_viirs_period.csv")
v = vp[vp["mode"] == "all"]
print("\n## by VIIRS period, F1 (interval)\n")
rows = []
for p in ["fusion", *SINGLE]:
    r = {"product": NAME[p]}
    for per, g in v.groupby("viirs_period"):
        x = g.set_index("product").loc[p]
        r[f"{per} ({int(x.n_events)} events)"] = f"{f(x.f1)} ({f(x.f1_lo)}–{f(x.f1_hi)})"
    rows.append(r)
print(md(pd.DataFrame(rows)))
rb = pd.read_csv(R / "reliability_bins.csv")
o = rb[(rb.rule == "corrected") & (rb.labels == "map days")].groupby(["landtype", "bin"])[["n", "sum_p", "n_flood"]].sum()
o["mean_p"], o["flood_share"] = o.sum_p / o.n, o.n_flood / o.n
print("\n## reliability bins, map days\n")
print(o[["n", "mean_p", "flood_share"]].round(3).to_string())
r = rates[rates.rule == "corrected"]
print("\n## rates over folds, as hit rate and false-alarm rate (1 - spec)\n")
rows = []
for (s, lt), g in r.groupby(["source", "landtype"]):
    rows.append({"source": NAME[s], "land type": lt, "hit rate": f"{g.sens.min():.2f}–{g.sens.max():.2f}",
                 "false-alarm rate": f"{(1 - g.spec.max()):.1%}–{(1 - g.spec.min()):.1%}"})
print(md(pd.DataFrame(rows)))
