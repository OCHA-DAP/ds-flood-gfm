"""Events-tab table, continent table and fold-balance table from results/event_folds*.csv (after make_folds).

Run from the repo: uv run python <this> > events_final.md
"""

import sys

import pandas as pd

S = "/Users/zackarno/Documents/CHD/repos/ds-flood-gfm/data/fusion/resume"
ev = pd.read_csv(f"{S}/events_in_scope2.csv")
fo = pd.read_csv("results/event_folds.csv")
if set(ev.code) != set(fo.code):
    sys.exit(f"event sets differ: {sorted(set(ev.code) ^ set(fo.code))}")
ev = ev.drop(columns="flood").merge(fo[["code", "fold", "flood_cell_days", "basin_id"]], on="code")
built = ev[ev.status == "built"]
old = pd.read_csv(f"{S}/events_in_scope2.csv").set_index("code").flood
chk = built.set_index("code").flood_cell_days.sub(old.reindex(built.code)).abs()
print(f"<!-- check: events built before whose flood count changed: {int((chk > 0).sum())} -->")

ev = ev.sort_values(["continent", "start"], ascending=[True, False])
lines = ["| continent | event | catalog | country | first map | last map | map days | windows | cells | gold flooded cell-days | fold |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in ev.itertuples():
    lines.append(f"| {r.continent} | {r.code} | {r.catalog} | {r.country} | {r.start} | {r.end} | {r.n_days} | {r.n_windows} | "
                 f"{r.cells:,} | {int(r.flood_cell_days):,} | {int(r.fold) + 1} |")
print("\n".join(lines))

print("\n\n## continent table\n")
g = ev.groupby("continent").agg(events=("code", "size"), flood=("flood_cell_days", "sum"))
tot = g.flood.sum()
print("| continent | events | gold flooded cell-days |\n|---|---|---|")
for c, r in g.sort_values("flood", ascending=False).iterrows():
    print(f"| {c} | {r.events} | {int(r.flood):,} ({r.flood / tot:.0%}) |")
print(f"| all | {g.events.sum()} | {int(tot):,} |")
top = ev.sort_values("flood_cell_days", ascending=False).head(5)
print(f"\ntop 5 events hold {top.flood_cell_days.sum() / tot:.0%}: " + ", ".join(f"{r.code} ({r.country}, {int(r.flood_cell_days):,})" for r in top.itertuples()))

print("\n\n## fold balance\n")
b = pd.read_csv("results/event_folds_balance.csv")
cols = [c for c in b.columns if c.startswith("events_")]
fl = [c for c in b.columns if c.startswith("flood_share_")]
print("| fold | " + " | ".join(c.removeprefix("events_") for c in cols) + " | " + " | ".join(c.removeprefix("flood_share_") + " flood" for c in fl) + " |")
print("|---|" + "---|" * (len(cols) + len(fl)))
for _, d in b.iterrows():
    pass
    print(f"| {int(d['fold']) + 1} | " + " | ".join(str(int(d[c])) for c in cols) + " | " + " | ".join(f"{d[c]:.0%}" for c in fl) + " |")
print(f"\nbasins: {ev.basin_id.nunique()}")
