"""Phase 3: accuracy table (gold confusion matrices + EM for thin strata) and prior v1.

Inputs: phase1 event cubes (training events) and phase2 sample cubes.
Outputs:
  results/accuracy_table.csv          the model: sens/spec per source x stratum, 95% CI, sample sizes, flags
  results/accuracy_table_gold_all.csv gold-only estimates for every source incl. FloodScan/GFDS (diagnostic)
  results/phase3_post_window_pofd.csv false-alarm rate by dry-window position (recession check, D-12)
  results/prior_v1.csv                per-stratum base flood rate with 95% CI

Run: nice -n 19 uv run python scripts/fusion/phase3_accuracy.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ds_flood_gfm.fusion import table
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.feasibility import extract_all

log = logging.getLogger("phase3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg.local("logs", "phase3.log")), logging.StreamHandler(sys.stdout)])
    logging.getLogger("azure").setLevel(logging.WARNING)
    sel = pd.read_parquet(cfg.local("gold", "events_selected.parquet"))
    counts, _ = extract_all(cfg, sel)
    gold_pat = table.gold_pattern_counts(cfg, sel, "train")
    samp_pat = table.sample_pattern_counts(cfg)
    acc, gold_all = table.build_accuracy_table(cfg, counts, gold_pat, samp_pat)
    res = cfg.results_dir
    acc.to_csv(res / "accuracy_table.csv", index=False)
    gold_all.to_csv(res / "accuracy_table_gold_all.csv", index=False)
    table.post_window_pofd(counts).to_csv(res / "phase3_post_window_pofd.csv", index=False)
    prior = table.build_prior_v1(cfg, acc, samp_pat)
    prior.to_csv(res / "prior_v1.csv", index=False)
    log.info("accuracy table:\n%s", acc[["source", "stratum", "method", "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi", "n_flood_obs", "n_dry_obs", "flag_wide_ci"]].to_string())
    log.info("prior v1:\n%s", prior.to_string())
    flagged = acc[acc.flag_wide_ci]
    log.info("%d of %d rows have a 95%% interval wider than %.2f", len(flagged), len(acc), cfg["accuracy"]["ci_flag_width"])


if __name__ == "__main__":
    main()
