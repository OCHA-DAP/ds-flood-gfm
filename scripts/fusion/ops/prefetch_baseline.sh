#!/bin/zsh
cd /Users/zackarno/Documents/CHD/repos/ds-flood-gfm
for i in $(seq 1 10); do
  nice -n 19 uv run python -u scripts/fusion/ops/prefetch_baseline.py && exit 0
  echo "attempt $i FAILED $(date)"; sleep 30
done
exit 1
