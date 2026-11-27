#!/bin/zsh
# Hold-out run of the FloodScan chain: tables from 2024+2025, test year 2023.
set -e
cd "$(dirname "$0")/../../.."  # repo root
echo "== 04: static tables from 2024,2025"
uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/04_static_tables.py 2024,2025 data/floodscan_repro/static_tables_2425.npz 2>&1 | grep -v "RuntimeWarning\|warn\|^ *$"
echo "== 05: FF chain, test year 2023"
uv run --no-project --with numpy --with scipy --with rasterio python experiments/floodscan_repro/05_ff_chain.py data/floodscan_repro/static_tables_2425.npz data/floodscan_repro/ff_2023.npz experiments/floodscan_repro/diagnostics/holdout_2023.json 2>&1 | grep -v "RuntimeWarning\|warn\|^ *$"
echo "== 06: score against FloodScan 2023"
uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2023.npz results/floodscan_repro/holdout_2023 2>&1 | grep -v "RuntimeWarning\|warn\|^ *$"
echo "== done"
