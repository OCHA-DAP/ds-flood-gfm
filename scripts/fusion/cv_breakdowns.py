"""Two descriptive breakdowns of the cross-validated scores (no refit, no rule changes).

Reads the per-event map-day counts phase5_crossval.py caches (data/fusion/metrics/cv_skill_events.parquet)
and results/event_folds.csv. Writes, to results/cv/:
  skill_without_largest_basin.csv  every product, all cells and observed cells, without the basin that
                                   holds the most gold flood (one HydroBASINS level-4 basin in Bangladesh
                                   holds 40% of it), to show how much that one basin drives the totals
  skill_by_viirs_period.csv        events whose maps fall inside the VIIRS archive gap (no VIIRS files,
                                   2021-01-01 to 2023-07-31) against the rest

Run: uv run python scripts/fusion/cv_breakdowns.py
"""

from __future__ import annotations

import pandas as pd

from ds_flood_gfm.fusion import crossval
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.metrics import pooled

GAP = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-07-31"))


def table(sk: pd.DataFrame, by: list[str], reps: int) -> pd.DataFrame:
    keys = ["product", "mode", *by]
    t = pooled(sk, keys).rename(columns={"sens": "recall"})
    ci = crossval.bootstrap_scores(sk.groupby([*keys, "code"], as_index=False)[["tp", "fp", "fn"]].sum(), keys, reps, seed=20261015)
    return t.merge(ci, on=keys)[[*keys, "n_events", "truth_flood", "recall", "precision", "f1", "f1_lo", "f1_hi"]]


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
    period = sk.code.map(lambda c: "VIIRS gap (2021-01 to 2023-07)" if inside[c] else "VIIRS archive available")
    by_p = table(sk.assign(viirs_period=period), ["viirs_period"], reps)
    by_p.to_csv(res / "skill_by_viirs_period.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"largest basin {big}: {share:.0%} of gold flood")
    print(out[out["mode"] == "all"].round(3).to_string(index=False))
    print(by_p[by_p["mode"] == "all"].sort_values(["viirs_period", "f1"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
