"""One day of GFDS per-sensor signal: how much a single sensor covers, and how well sensors agree.

Feeds docs/gfds-rebuild-evaluation.md section 3. Downloads JRC's own AMSR2, GPM and ALL
signal rasters for DAY plus oceanmask.tif (whole files, ~70 MB, >= 1.2 s apart), then reports:
- share of land pixels with a valid signal, by latitude band, per sensor and for the merge;
- AMSR2 vs GPM signal where both are valid on land (65S-65N).

Valid = not a sentinel (-32000, int32-min) and signal > 0, as in fusion/sources/gfds.py.
oceanmask.tif: 1 = ocean, 0 = land (checked: rows south of 70S are ~79% land, 0-10N ~23%).
Any HTTP error raises; nothing is skipped.

    uv run python experiments/gfds_rebuild/01_sensor_coverage.py [YYYYMMDD]
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

DAY = sys.argv[1] if len(sys.argv) > 1 else "20250815"
CACHE = Path("data/gdacs_gfds/rebuild")
BASE = "https://www.gdacs.org/flooddetection/DATA"
UA = "ds-flood-gfm research (OCHA CHD)"
SENTINELS = (-32000, np.iinfo("int32").min)
y, m = DAY[:4], DAY[4:6]
FILES = {
    "amsr2": f"AMSR2/SignalTiffs/{y}/{m}/signal_{DAY}_HD5.tif",
    "gpm": f"GPM/SignalTiffs/{y}/{m}/signal_{DAY}_GPM_HDF5.tif",
    "all": f"ALL/SignalTiffs/{y}/{m}/signal_{DAY}_ALL.tif",
    "oceanmask": "ALL/oceanmask.tif",
}


def fetch(rel: str) -> Path:
    out = CACHE / rel
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    time.sleep(1.2)
    return out


def signal(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    if a.shape != (2000, 4000):
        raise ValueError(f"unexpected GFDS grid {a.shape} in {path}")
    v = a.astype("float64") / 1e6
    for s in SENTINELS:
        v[a == s] = np.nan
    v[v <= 0] = np.nan
    return v


paths = {k: fetch(rel) for k, rel in FILES.items()}
amsr2, gpm, merged = (signal(paths[k]) for k in ("amsr2", "gpm", "all"))
with rasterio.open(paths["oceanmask"]) as ds:
    land = ds.read(1) == 0
    lat = ds.transform.f + (np.arange(ds.height) + 0.5) * ds.transform.e

bands = {"> 65N": lat > 65, "65S-65N": (lat >= -65) & (lat <= 65), "< 65S": lat < -65}
print(f"GFDS {DAY}: share of land pixels with a valid signal")
print(f"{'band':10s} {'pixels':>9s} {'AMSR2':>7s} {'GPM':>7s} {'ALL':>7s}")
for name, rows in bands.items():
    mask = land & rows[:, None]
    shares = [np.isfinite(x)[mask].mean() for x in (amsr2, gpm, merged)]
    print(f"{name:10s} {int(mask.sum()):9d} {shares[0]:7.3f} {shares[1]:7.3f} {shares[2]:7.3f}")

both = np.isfinite(amsr2) & np.isfinite(gpm) & land & bands["65S-65N"][:, None]
diff = gpm[both] - amsr2[both]
print(f"\nland 65S-65N, both valid: n = {int(both.sum())}, r = {np.corrcoef(amsr2[both], gpm[both])[0, 1]:.3f}")
print(f"GPM - AMSR2: median {np.median(diff):+.4f}, IQR {np.percentile(diff, 25):+.4f} .. {np.percentile(diff, 75):+.4f}")
print(f"median signal: AMSR2 {np.median(amsr2[both]):.4f}, GPM {np.median(gpm[both]):.4f}")
