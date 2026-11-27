#!/bin/zsh
# Regenerate every result table cited in docs/floodscan-algorithm-notes.md from the npz files on disk, so
# `git diff results/` shows whether the committed tables are what the code produces today.
set -e
cd "$(dirname "$0")/../../.."
echo "== 06 on ff_2024.npz (step 2 results, tables 2023+2025)"
uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2024.npz results/floodscan_repro 2>&1 | grep -v "Warning\|warn\|^ *$" | tail -3
echo "== 06 on ff_2024_atmos.npz"
uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2024_atmos.npz results/floodscan_repro/atmos 2>&1 | grep -v "Warning\|warn\|^ *$" | tail -3
echo "== 06 on ff_2023.npz (hold-out, tables 2024+2025)"
uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_2023.npz results/floodscan_repro/holdout_2023 2>&1 | grep -v "Warning\|warn\|^ *$" | tail -3
for y in 2023 2024; do for w in 1e5 1e6 1e7; do
  echo "== 06 on ff_${y}_wexp${w}.npz"
  uv run python experiments/floodscan_repro/06_score_2024.py data/floodscan_repro/ff_${y}_wexp${w}.npz results/floodscan_repro/wexp/${y}_${w} 2>&1 | grep -v "Warning\|warn\|^ *$" | tail -3
done; done
echo "== git diff of the result tables"
git --no-pager diff --stat -- results/
echo "== done"
