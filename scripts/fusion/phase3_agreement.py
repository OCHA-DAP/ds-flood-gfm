"""Phase 3 (plan v2, ADR 0002): the agreement accuracy table, its comparison with gold, and the gold check
on the independence assumption the agreement table rests on.

Inputs (cached by phase3_accuracy.py): data/fusion/metrics/sample_patterns.parquet (random sample) and
gold_patterns_train.parquet (training gold cells, per event); results/accuracy_table.csv (v1 model table).
Outputs (results/):
  accuracy_table_agreement.csv     agreement-only sens/spec/prevalence per source x stratum, fit diagnostics
  accuracy_table_comparison.csv    agreement vs gold on evidence weights, status, direction, table_used
  phase3_gold_error_dependence.csv do sources err together on gold cells? pooled and within-event lift
The v1 model table is not modified: every status other than `agree` keeps the gold row.
No bootstrap intervals: one fit takes 10-30 s with 3 starts, so 200 replicates x 6 strata would take hours.

Run: nice -n 19 uv run python scripts/fusion/phase3_agreement.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ds_flood_gfm.fusion import agreement
from ds_flood_gfm.fusion.config import load_config

log = logging.getLogger("phase3_agreement")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "phase3_agreement.log")), logging.StreamHandler(sys.stdout)])
    a = cfg["accuracy"]["agreement"]
    res = cfg.results_dir
    for p in [cfg.local("metrics", "sample_patterns.parquet"), cfg.local("metrics", "gold_patterns_train.parquet"), res / "accuracy_table.csv"]:
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing: run make phase3 first")
    samp = pd.read_parquet(cfg.local("metrics", "sample_patterns.parquet"))
    gold_pat = pd.read_parquet(cfg.local("metrics", "gold_patterns_train.parquet"))
    gold_tab = pd.read_csv(res / "accuracy_table.csv")

    agr = agreement.agreement_table(samp, gold_tab, a)
    agr.to_csv(res / "accuracy_table_agreement.csv", index=False)
    cmp = agreement.compare(agr, gold_tab, a["weight_tolerance"])
    cmp.to_csv(res / "accuracy_table_comparison.csv", index=False)
    dep = agreement.error_dependence(gold_pat)
    dep.to_csv(res / "phase3_gold_error_dependence.csv", index=False)

    pd.set_option("display.width", 250)
    log.info("agreement table:\n%s", agr[["stratum", "source", "sens", "spec", "prevalence", "flag_rate", "converged", "unstable"]].round(4).to_string(index=False))
    log.info("comparison:\n%s", cmp[["stratum", "source", "gold_sens", "agr_sens", "gold_spec", "agr_spec", "d_w_flood", "d_w_dry", "status"]].round(4).to_string(index=False))
    log.info("status counts: %s", cmp.status.value_counts().to_dict())
    log.info("gold error dependence (dry cells):\n%s", dep[dep.truth == "dry"][["stratum", "pair", "n_both_wrong", "lift", "lift_within_event"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
