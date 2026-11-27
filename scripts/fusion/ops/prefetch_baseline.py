"""Cache the JRC normally-wet masks (baseline.normally_wet) for the diagnostic's months:
map-day months and the false-event period (days 30 to 10 before the first map). Smallest first; resumes."""
import time, pandas as pd
from ds_flood_gfm.fusion import gold, baseline
from ds_flood_gfm.fusion.config import load_config
cfg = load_config()
folds = pd.read_csv("results/event_folds.csv")
ev = pd.read_parquet(cfg.local("gold", "event_table_all.parquet")); ev = ev[ev.code.isin(folds.code)]
todo = []
for _, r in ev.iterrows():
    months = set(pd.date_range(r.window_start, r.window_end).month)
    for ew in gold.event_windows(r):
        for m in sorted(months):
            if not cfg.local("static", "baseline", f"{ew.window.key()}_m{m:02d}.npy").exists():
                todo.append((ew.window.shape[0] * ew.window.shape[1], r.code, ew, m))
todo.sort(key=lambda x: x[0])
print(f"{len(todo)} window-months to fetch, {sum(t[0] for t in todo):,} cells", flush=True)
t0 = time.time(); done = 0
for n, (cells, code, ew, m) in enumerate(todo, 1):
    baseline.normally_wet(cfg, ew.window, m); done += cells
    if n % 20 == 0 or n == len(todo):
        print(f"{n}/{len(todo)} {code} m{m:02d}; {done:,} cells; {time.time() - t0:.0f} s", flush=True)
print("prefetch done", flush=True)
