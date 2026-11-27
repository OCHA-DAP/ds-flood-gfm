#!/bin/zsh
# Sentinel-2 referee, resumable; restart on transient failures (network), at most 8 attempts.
cd /Users/zackarno/Documents/CHD/repos/ds-flood-gfm
LOG=data/fusion/resume/logs/s2_referee.out
for i in {1..8}; do
  nice -n 19 uv run python -u scripts/fusion/experiment_s2_referee.py >> $LOG 2>&1 && { echo "all done $(date)" >> $LOG; exit 0; }
  echo "attempt $i failed $(date)" >> $LOG
  sleep 60
done
echo "gave up $(date)" >> $LOG
