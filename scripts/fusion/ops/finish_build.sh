#!/bin/zsh
# Finish the flood fusion re-run after a shutdown (written 2026-10-14). Every step resumes: built gold
# windows, stored GFDS days and built cubes are skipped. Safe to stop (kill this script's process tree)
# and start again at any point. Low priority throughout. Log: data/fusion/resume/logs/finish.out
cd /Users/zackarno/Documents/CHD/repos/ds-flood-gfm
L=data/fusion/resume/logs
ALL=data/fusion/gold/events_extra_large.parquet
A=data/fusion/gold/events_extra_large_part1.parquet
B=data/fusion/gold/events_extra_large_part2.parquet   # FL20200713BGD (2.1 GB of label polygons), alone
P="nice -n 19 uv run python -u scripts/fusion/phase1_feasibility.py"
step() { echo "$(date '+%F %T') $*"; }
retry() {  # retry <attempts> <log> <args...>: a fresh process each time (clears GDAL's cache; resumes)
  local n=$1 log=$2; shift 2
  for i in $(seq 1 $n); do
    ${=P} "$@" >> $L/$log 2>&1 && { step "$log done"; return 0; }
    step "$log attempt $i FAILED"; sleep 60
  done
  step "$log GAVE UP after $n attempts"; return 1
}
step start
# GFDS (network) runs alongside the gold labels (memory); sources need both
( retry 40 gfds.out gfds --events-file $ALL ) &
GFDS=$!
retry 15 gold_a.out gold --workers 1 --events-file $A || exit 1
retry 15 gold_b.out gold --workers 1 --events-file $B || exit 1
wait $GFDS || { step "GFDS did not finish"; exit 1; }
retry 3 sources_a.out sources --workers 2 --continue-on-error --events-file $A || exit 1
retry 3 sources_a2.out sources --workers 2 --continue-on-error --events-file $A || exit 1
retry 3 sources_b.out sources --workers 1 --continue-on-error --events-file $B || exit 1
retry 3 sources_b2.out sources --workers 1 --continue-on-error --events-file $B || exit 1
for f in results/phase1_stage_c_failures_events_extra_large_part{1,2}.csv; do
  [ -f $f ] && [ $(wc -l < $f) -gt 1 ] && { step "sources failures remain in $f:"; cat $f; exit 1; }
done
nice -n 19 uv run python scripts/fusion/make_folds.py > $L/make_folds.out 2>&1 || { step "make_folds FAILED"; exit 1; }
step "folds written"
nice -n 19 uv run python -u scripts/fusion/phase5_crossval.py > $L/crossval.out 2>&1 || { step "crossval FAILED"; exit 1; }
step "ALL DONE: results in results/cv/"
