#!/bin/zsh
# Does giving the in-season 61-day expectation real weight (w_exp >> 1/var61 ~ 1e6) remove the 2023 Sahel false
# alarms without hurting 2024? Tables: 2023 tested with 2024+2025 tables, 2024 with 2023+2025 tables.
set -e
cd "$(dirname "$0")/../../.."  # repo root
S=experiments/floodscan_repro/diagnostics
for y in 2023 2024; do for w in 1e5 1e6 1e7; do printf '{"year": %s, "w_exp": %s, "variants": ["full"]}\n' $y $w > $S/wexp_${y}_${w}.json; done; done
for w in 1e5 1e6 1e7; do
  echo "== 2023 w_exp $w"
  uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/05_ff_chain.py data/floodscan_repro/static_tables_2425.npz data/floodscan_repro/ff_2023_wexp$w.npz $S/wexp_2023_$w.json 2>&1 | grep "^full\|test year"
  uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2023_wexp$w.npz results/floodscan_repro/wexp/2023_$w 2>&1 | grep -A2 "^variant\|^full"
  echo "== 2024 w_exp $w"
  uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/05_ff_chain.py data/floodscan_repro/static_tables.npz data/floodscan_repro/ff_2024_wexp$w.npz $S/wexp_2024_$w.json 2>&1 | grep "^full\|test year"
  uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2024_wexp$w.npz results/floodscan_repro/wexp/2024_$w 2>&1 | grep -A2 "^variant\|^full"
done
echo "== done"
