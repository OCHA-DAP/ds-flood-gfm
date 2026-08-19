# GFDS Prototype & Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal GFDS (GDACS Global Flood Detection System) access layer, compute our own flood anomalies over the Pakistan 2022 floods, validate them against FloodScan SFED, and produce a go/no-go memo on adopting GFDS as a free FloodScan analogue.

**Architecture:** A single new datasource module (`gfds.py`) provides URL building, nodata masking/scaling, and windowed remote reads over plain HTTP (no auth). Case-study scripts in `experiments/` use it to build a small space-time stack for one flood event, compute anomalies two ways (JRC's static 2009 baseline vs our own day-of-year climatology), and compare both against FloodScan at admin level. **The archive backfill pipeline is explicitly OUT of scope** — it gets its own plan only if the validation gate passes.

**Tech Stack:** Python 3.12, `uv`, rasterio/rioxarray/xarray, rasterstats, ocha-stratus (blob access for FloodScan + CODAB), pytest.

**Spec:** `docs/gdacs-gfds-assessment.md` — read it fully before starting; every magic number below is verified there.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python ...`). Install dev deps once: `uv sync --extra dev`.
- Work on branch `feat/gfds-prototype` off `main`.
- **Fail loudly.** No broad `try/except ... continue`. Distinguish three states everywhere: *data absent upstream* (HTTP 404 → record explicitly), *fetch failed* (network/server error → raise), *value nodata* (sentinel → NaN). They must never look the same downstream.
- GFDS rasters are int32 with **two** nodata sentinels: `-32000` and `-2147483648`. Mask both, always.
- Scale factors: signal ×1,000,000 (so does the static baseline avg/sd), magnitude ×1,000. After scaling, plausible signal values are ~0.4–3.0; anything ≤ 0 is invalid.
- Anomaly sign convention: floods LOWER the signal, so anomaly = `(avg − signal) / sd` → flood = **positive** anomaly.
- Be polite to the GDACS server: max 1 request/second, `User-Agent: OCHA-CHD-DS ds-flood-gfm GFDS prototype`. All bulk fetches must be resumable (skip what's already cached).
- Never commit anything under `data/` or `outputs/` (both gitignored). Commit code, tests, and the memo only.
- Study area for the whole plan: Pakistan Indus bbox `(66.0, 24.0, 71.0, 29.0)` (lon_min, lat_min, lon_max, lat_max); event window 2022-06-01 → 2022-10-31; flood peak ≈ 2022-08-29.

## File Structure

```
src/ds_flood_gfm/datasources/gfds.py    # NEW: all GFDS access logic (Tasks 1–5)
tests/test_gfds.py                      # NEW: unit tests (first tests in repo)
tests/conftest.py                       # NEW: synthetic GeoTIFF fixture
experiments/gfds_pakistan_2022.py       # NEW: case study — stack, anomalies, plots (Task 6)
experiments/gfds_vs_floodscan.py        # NEW: FloodScan comparison + metrics (Task 7)
experiments/gfds_climatology.py         # NEW: DOY climatology + re-validation (Task 8)
docs/gfds-validation-memo.md            # NEW: results + go/no-go (Task 9)
data/gdacs_gfds/                        # gitignored cache (per-day .npy + baselines)
outputs/gfds_validation/                # gitignored plots + metrics CSVs
```

---

### Task 1: Repo test scaffolding + GFDS URL builder

**Files:**
- Create: `tests/conftest.py`, `tests/test_gfds.py`
- Create: `src/ds_flood_gfm/datasources/gfds.py`

**Interfaces:**
- Produces: module constants `SENTINELS = (-32000, -2147483648)`, `SCALE = {"signal": 1_000_000, "mag": 1_000}`, `GFDS_BASE = "https://www.gdacs.org/flooddetection/DATA"`; function `gfds_raster_url(date: datetime.date, kind: str = "signal", product: str = "ALL") -> str`; function `gfds_baseline_url(which: str) -> str` (`which` in `{"avg", "sd"}`).

- [ ] **Step 1: Create `tests/conftest.py` with a synthetic-raster fixture** (used from Task 2 on; harmless to add now)

```python
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_gfds_tif(tmp_path):
    """A 10x10 int32 raster shaped like a GFDS signal file.

    Layout: value 900_000 everywhere (signal 0.9), except
    row 0 = -32000 (declared nodata), row 1 = -2147483648
    (undeclared sentinel), row 2 = -5 (impossible negative).
    """
    arr = np.full((10, 10), 900_000, dtype="int32")
    arr[0, :] = -32000
    arr[1, :] = -2147483648
    arr[2, :] = -5
    path = tmp_path / "signal_20220829_ALL.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=1,
        dtype="int32", crs="EPSG:4326", nodata=-32000.0,
        transform=from_origin(66.0, 29.0, 0.09, 0.09),
    ) as dst:
        dst.write(arr, 1)
    return path
```

- [ ] **Step 2: Write failing URL tests in `tests/test_gfds.py`**

```python
import datetime

from ds_flood_gfm.datasources.gfds import gfds_baseline_url, gfds_raster_url


def test_signal_url():
    assert gfds_raster_url(datetime.date(2022, 8, 29)) == (
        "https://www.gdacs.org/flooddetection/DATA/ALL/"
        "SignalTiffs/2022/08/signal_20220829_ALL.tif"
    )


def test_mag_url_has_mag_signal_prefix_and_folder():
    url = gfds_raster_url(datetime.date(1998, 3, 5), kind="mag")
    assert url == (
        "https://www.gdacs.org/flooddetection/DATA/ALL/"
        "MagTiffs/1998/03/mag_signal_19980305_ALL.tif"
    )


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError, match="kind"):
        gfds_raster_url(datetime.date(2022, 8, 29), kind="bogus")


def test_baseline_urls():
    assert gfds_baseline_url("avg").endswith("AveragesAndSd/bt_signal_avg.tif")
    assert gfds_baseline_url("sd").endswith("AveragesAndSd/bt_signal_sd.tif")
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_gfds.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (module doesn't exist yet)

- [ ] **Step 4: Implement in `src/ds_flood_gfm/datasources/gfds.py`**

```python
"""Access to GDACS GFDS flood rasters.

See docs/gdacs-gfds-assessment.md for what GFDS is and why the
constants below look the way they do. Everything here was verified
against the live server on 2026-08-17.
"""
import datetime

GFDS_BASE = "https://www.gdacs.org/flooddetection/DATA"
USER_AGENT = "OCHA-CHD-DS ds-flood-gfm GFDS prototype"

# int32 sentinels: -32000 is the declared nodata; int32-min appears
# undeclared in real files. Both mean "no observation".
SENTINELS = (-32000, -2147483648)

# Stored value = physical value * scale. Signal is the M/C brightness
# temperature ratio; magnitude is sd-from-mean, clipped to +/-20.
SCALE = {"signal": 1_000_000, "mag": 1_000}

_KIND_LAYOUT = {  # kind -> (folder, filename prefix)
    "signal": ("SignalTiffs", "signal"),
    "mag": ("MagTiffs", "mag_signal"),
}


def gfds_raster_url(
    date: datetime.date, kind: str = "signal", product: str = "ALL"
) -> str:
    if kind not in _KIND_LAYOUT:
        raise ValueError(f"kind must be one of {sorted(_KIND_LAYOUT)}, got {kind!r}")
    folder, prefix = _KIND_LAYOUT[kind]
    return (
        f"{GFDS_BASE}/{product}/{folder}/{date:%Y}/{date:%m}/"
        f"{prefix}_{date:%Y%m%d}_{product}.tif"
    )


def gfds_baseline_url(which: str) -> str:
    if which not in ("avg", "sd"):
        raise ValueError(f"which must be 'avg' or 'sd', got {which!r}")
    return f"{GFDS_BASE}/ALL/AveragesAndSd/bt_signal_{which}.tif"
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_gfds.py -v` — Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add tests/ src/ds_flood_gfm/datasources/gfds.py
git commit -m "feat(gfds): URL builders and constants for GDACS flood rasters"
```

---

### Task 2: Mask-and-scale

**Files:**
- Modify: `src/ds_flood_gfm/datasources/gfds.py`
- Modify: `tests/test_gfds.py`

**Interfaces:**
- Consumes: `SENTINELS`, `SCALE` from Task 1.
- Produces: `mask_and_scale(arr: np.ndarray, kind: str) -> np.ndarray` — float64, sentinels → NaN, divided by scale; for `kind="signal"`, values ≤ 0 also → NaN.

- [ ] **Step 1: Write failing tests (append to `tests/test_gfds.py`)**

```python
import numpy as np

from ds_flood_gfm.datasources.gfds import mask_and_scale


def test_mask_and_scale_signal():
    arr = np.array([[900_000, -32000], [-2147483648, -5]], dtype="int32")
    out = mask_and_scale(arr, "signal")
    assert out[0, 0] == 0.9
    assert np.isnan(out[0, 1])   # declared nodata
    assert np.isnan(out[1, 0])   # undeclared int32-min sentinel
    assert np.isnan(out[1, 1])   # negative signal is physically impossible


def test_mask_and_scale_mag_keeps_negatives():
    arr = np.array([[-3000, 20000]], dtype="int32")
    out = mask_and_scale(arr, "mag")
    assert out[0, 0] == -3.0  # negative magnitude is valid (drier than normal)
    assert out[0, 1] == 20.0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_gfds.py -v -k mask` → FAIL (ImportError)

- [ ] **Step 3: Implement (append to `gfds.py`)**

```python
import numpy as np


def mask_and_scale(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind not in SCALE:
        raise ValueError(f"kind must be one of {sorted(SCALE)}, got {kind!r}")
    out = arr.astype("float64")
    invalid = np.isin(arr, SENTINELS)
    if kind == "signal":
        invalid |= arr <= 0
    out[invalid] = np.nan
    return out / SCALE[kind]
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_gfds.py -v` → all PASS

- [ ] **Step 5: Commit** — `git commit -am "feat(gfds): sentinel masking and scaling"`

---

### Task 3: Windowed reader (local file or remote URL)

**Files:**
- Modify: `src/ds_flood_gfm/datasources/gfds.py`
- Modify: `tests/test_gfds.py`

**Interfaces:**
- Consumes: `mask_and_scale`, `gfds_raster_url`.
- Produces: `read_gfds(path_or_url: str, kind: str, bbox: tuple | None = None) -> xarray.DataArray` — dims `(y, x)`, float64, georeferenced, masked+scaled. Remote URLs are read via GDAL `/vsicurl/` range requests (only the needed bytes travel — verified to work against this server).

- [ ] **Step 1: Write failing test (uses the Task 1 fixture)**

```python
from ds_flood_gfm.datasources.gfds import read_gfds


def test_read_gfds_local(synthetic_gfds_tif):
    da = read_gfds(str(synthetic_gfds_tif), kind="signal")
    assert da.shape == (10, 10)
    assert float(da.isel(y=5, x=5)) == 0.9
    assert np.isnan(float(da.isel(y=0, x=0)))  # -32000 row
    assert np.isnan(float(da.isel(y=1, x=0)))  # int32-min row
    assert np.isnan(float(da.isel(y=2, x=0)))  # negative row


def test_read_gfds_bbox_subset(synthetic_gfds_tif):
    # fixture spans lon 66->66.9, lat 28.1->29 at 0.09 deg
    da = read_gfds(str(synthetic_gfds_tif), kind="signal",
                   bbox=(66.0, 28.55, 66.45, 29.0))
    assert da.shape == (5, 5)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_gfds.py -v -k read` → FAIL

- [ ] **Step 3: Implement (append to `gfds.py`)**

```python
import rasterio
import xarray as xr
from rasterio.windows import from_bounds


def _gdal_path(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return f"/vsicurl/{path_or_url}"
    return path_or_url


def read_gfds(path_or_url: str, kind: str, bbox: tuple | None = None) -> xr.DataArray:
    with rasterio.open(_gdal_path(path_or_url)) as src:
        window = from_bounds(*bbox, src.transform) if bbox else None
        raw = src.read(1, window=window)
        transform = src.window_transform(window) if window else src.transform
    data = mask_and_scale(raw, kind)
    ny, nx = data.shape
    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e
    return xr.DataArray(
        data, dims=("y", "x"), coords={"y": ys, "x": xs},
        attrs={"crs": "EPSG:4326", "kind": kind, "source": path_or_url},
    )
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_gfds.py -v` → all PASS

- [ ] **Step 5: One-off remote smoke check (not a committed test — server-dependent)**

```bash
uv run python -c "
import datetime
from ds_flood_gfm.datasources.gfds import gfds_raster_url, read_gfds
da = read_gfds(gfds_raster_url(datetime.date(2022, 8, 29), 'mag'), 'mag',
               bbox=(66, 24, 71, 29))
print(da.shape, float(da.max()))"
```
Expected: shape ≈ `(56, 56)`, max = 20.0 (the Pakistan flood, clipped at +20σ).

- [ ] **Step 6: Commit** — `git commit -am "feat(gfds): windowed local/remote raster reader"`

---

### Task 4: Resumable stack fetcher

**Files:**
- Modify: `src/ds_flood_gfm/datasources/gfds.py`
- Modify: `tests/test_gfds.py`

**Interfaces:**
- Consumes: `read_gfds`, `gfds_raster_url`.
- Produces: `fetch_stack(start: date, end: date, bbox: tuple, cache_dir: Path, kind: str = "signal", throttle_s: float = 1.0) -> xarray.DataArray` — dims `(time, y, x)`. Caches one `.npy` per day in `cache_dir`; a day the server reports 404 for is cached as an all-NaN array via an empty marker file `{date}.missing` (absence is recorded, not invented); any other error raises immediately.

- [ ] **Step 1: Write failing test — cache round-trip and missing-day handling, with the network layer monkeypatched**

```python
import datetime
import pathlib

import ds_flood_gfm.datasources.gfds as gfds_mod
from ds_flood_gfm.datasources.gfds import fetch_stack


class _FakeHTTPError(Exception):
    pass


def test_fetch_stack_caches_and_records_missing(tmp_path, monkeypatch):
    d1 = datetime.date(2022, 8, 29)
    d2 = datetime.date(2022, 8, 30)
    calls = []

    def fake_read(path_or_url, kind, bbox=None):
        calls.append(path_or_url)
        if "20220830" in path_or_url:
            raise gfds_mod.GFDSNotFound(path_or_url)
        import numpy as np, xarray as xr
        return xr.DataArray(
            np.full((2, 2), 0.9), dims=("y", "x"),
            coords={"y": [28.0, 27.0], "x": [66.0, 67.0]},
        )

    monkeypatch.setattr(gfds_mod, "read_gfds", fake_read)
    stack = fetch_stack(d1, d2, bbox=(66, 24, 71, 29),
                        cache_dir=tmp_path, throttle_s=0)
    assert stack.sizes["time"] == 2
    assert float(stack.sel(time="2022-08-29").mean()) == 0.9
    assert bool(stack.sel(time="2022-08-30").isnull().all())
    assert (tmp_path / "signal_20220830.missing").exists()

    # second call must hit the cache, not the network
    n_calls = len(calls)
    fetch_stack(d1, d2, bbox=(66, 24, 71, 29), cache_dir=tmp_path, throttle_s=0)
    assert len(calls) == n_calls
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_gfds.py -v -k fetch` → FAIL

- [ ] **Step 3: Implement (append to `gfds.py`)**

```python
import time as _time
from pathlib import Path

import pandas as pd
from rasterio.errors import RasterioIOError


class GFDSNotFound(Exception):
    """The server has no file for this date (HTTP 404)."""


def _read_remote_day(date, kind, bbox, product):
    url = gfds_raster_url(date, kind, product)
    try:
        return read_gfds(url, kind, bbox=bbox)
    except RasterioIOError as e:
        # GDAL surfaces vsicurl 404s as "not recognized"/"does not exist".
        if "404" in str(e) or "does not exist" in str(e):
            raise GFDSNotFound(url) from e
        raise  # real network/server errors must not be swallowed


def fetch_stack(start, end, bbox, cache_dir, kind="signal",
                product="ALL", throttle_s=1.0):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, end, freq="D")
    template = None
    days, missing = [], []
    for ts in dates:
        d = ts.date()
        npy = cache_dir / f"{kind}_{d:%Y%m%d}.npy"
        marker = cache_dir / f"{kind}_{d:%Y%m%d}.missing"
        if npy.exists():
            arr = np.load(npy)
        elif marker.exists():
            arr = None
        else:
            try:
                da = read_gfds(gfds_raster_url(d, kind, product), kind, bbox=bbox)
            except GFDSNotFound:
                marker.touch()
                missing.append(d)
                arr = None
            else:
                template = da
                arr = da.values
                np.save(npy, arr)
            _time.sleep(throttle_s)
        if arr is not None and template is None:
            # rebuild coords from any cached day on later runs
            template = read_gfds(gfds_raster_url(d, kind, product), kind, bbox=bbox)
        days.append(arr)
    if template is None:
        raise RuntimeError(
            f"No GFDS data found for {start}..{end} — every day was missing. "
            "That is not a normal state; check the URL pattern and server."
        )
    shape = template.shape
    cube = np.stack([a if a is not None else np.full(shape, np.nan) for a in days])
    if missing:
        print(f"NOTE: {len(missing)} day(s) absent upstream (recorded): {missing}")
    return xr.DataArray(
        cube, dims=("time", "y", "x"),
        coords={"time": dates, "y": template.y.values, "x": template.x.values},
    )
```

The `except GFDSNotFound: raise` distinction is the point: a 404 becomes a
*recorded absence*; anything else (timeouts, 500s, DNS) raises and stops the run.

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_gfds.py -v` → all PASS

- [ ] **Step 5: Commit** — `git commit -am "feat(gfds): resumable throttled stack fetcher with explicit missing-day records"`

---

### Task 5: Static-baseline anomaly

**Files:**
- Modify: `src/ds_flood_gfm/datasources/gfds.py`
- Modify: `tests/test_gfds.py`

**Interfaces:**
- Consumes: nothing new (pure array math).
- Produces: `static_anomaly(signal: xr.DataArray, avg: xr.DataArray, sd: xr.DataArray, sd_floor: float = 0.005) -> xr.DataArray` — `(avg − signal) / max(sd, sd_floor)`, in σ units, flood-positive.

- [ ] **Step 1: Write failing tests**

```python
import xarray as xr

from ds_flood_gfm.datasources.gfds import static_anomaly


def _da(v):
    return xr.DataArray(np.array([[v]]), dims=("y", "x"))


def test_flood_is_positive_anomaly():
    # signal drops from a 0.95 average to 0.80 with sd 0.05 -> +3 sigma
    out = static_anomaly(_da(0.80), avg=_da(0.95), sd=_da(0.05))
    assert float(out) == 3.0


def test_sd_floor_prevents_explosions():
    out = static_anomaly(_da(0.90), avg=_da(0.95), sd=_da(0.000001))
    assert float(out) == (0.95 - 0.90) / 0.005  # floored, not ~50000
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_gfds.py -v -k anomaly` → FAIL

- [ ] **Step 3: Implement**

```python
def static_anomaly(signal, avg, sd, sd_floor=0.005):
    return (avg - signal) / xr.where(sd > sd_floor, sd, sd_floor)
```

- [ ] **Step 4: Run tests** — all PASS. **Step 5: Commit** — `git commit -am "feat(gfds): static-baseline anomaly (flood-positive sign)"`

---

### Task 6: Pakistan 2022 case study

**Files:**
- Create: `experiments/gfds_pakistan_2022.py`

**Interfaces:**
- Consumes: `fetch_stack`, `read_gfds`, `gfds_baseline_url`, `static_anomaly`.
- Produces: `data/gdacs_gfds/pak2022/` day cache; `outputs/gfds_validation/pak_anomaly_static.nc` (the anomaly stack, consumed by Task 7); three PNGs.

No unit tests — this is an analysis script; its "test" is the printed sanity block at the end.

- [ ] **Step 1: Write `experiments/gfds_pakistan_2022.py`**

```python
"""Pakistan 2022 floods seen through GFDS with the static JRC baseline.

Fetches Jun-Oct 2022 daily signal over the Indus bbox (~150 windowed
range-reads, ~3 min on first run, instant after), computes flood-positive
anomalies, writes plots + the anomaly stack for Task 7.
Run: uv run python experiments/gfds_pakistan_2022.py
"""
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ds_flood_gfm.datasources.gfds import (
    fetch_stack, gfds_baseline_url, read_gfds, static_anomaly,
)

BBOX = (66.0, 24.0, 71.0, 29.0)
START, END = datetime.date(2022, 6, 1), datetime.date(2022, 10, 31)
PEAK = "2022-08-29"
CACHE = Path("data/gdacs_gfds/pak2022")
OUT = Path("outputs/gfds_validation")
OUT.mkdir(parents=True, exist_ok=True)

signal = fetch_stack(START, END, BBOX, CACHE, kind="signal")
avg = read_gfds(gfds_baseline_url("avg"), kind="signal", bbox=BBOX)
sd = read_gfds(gfds_baseline_url("sd"), kind="signal", bbox=BBOX)
anom = static_anomaly(signal, avg, sd)
anom.to_netcdf(OUT / "pak_anomaly_static.nc")

# -- plots ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
anom.sel(time="2022-06-15").plot(ax=axes[0], vmin=-3, vmax=6, cmap="RdBu_r")
axes[0].set_title("pre-flood (2022-06-15)")
anom.sel(time=PEAK).plot(ax=axes[1], vmin=-3, vmax=6, cmap="RdBu_r")
axes[1].set_title(f"peak ({PEAK})")
anom.mean(["y", "x"]).plot(ax=axes[2])
axes[2].set_title("bbox-mean anomaly over time")
fig.savefig(OUT / "pak_static_anomaly_overview.png", dpi=150,
            bbox_inches="tight")

# -- sanity block (this is the acceptance check) -------------------------
peak_mean = float(anom.sel(time=PEAK).mean())
june_mean = float(anom.sel(time="2022-06-15").mean())
print(f"bbox-mean anomaly  pre-flood: {june_mean:.2f} sigma   "
      f"peak: {peak_mean:.2f} sigma")
assert peak_mean > june_mean + 0.5, (
    "Peak anomaly should clearly exceed pre-flood — if not, the sign "
    "convention or baseline alignment is wrong. STOP and debug."
)
print("OK: flood peak is clearly visible in the DIY anomaly.")
```

- [ ] **Step 2: Run it** — `uv run python experiments/gfds_pakistan_2022.py`
Expected: `OK: flood peak is clearly visible...` and a PNG where the Indus corridor lights up red at the peak panel. Open the PNG and *look at it*.

- [ ] **Step 3: Commit** — `git add experiments/gfds_pakistan_2022.py && git commit -m "feat(gfds): Pakistan 2022 static-baseline case study"`

---

### Task 7: FloodScan comparison

**Files:**
- Create: `experiments/gfds_vs_floodscan.py`

**Interfaces:**
- Consumes: `outputs/gfds_validation/pak_anomaly_static.nc` (Task 6); FloodScan SFED COGs from team blob (`ocha-stratus`); Pakistan CODAB admin2.
- Produces: `outputs/gfds_validation/adm2_comparison.csv` (per-admin2 Pearson r + peak-timing lag) and `adm2_comparison.png`.

Before writing code, read `scripts/01_download_codab_to_blob.py` in this repo (the team CODAB pattern) and invoke the `/ocha-stratus` skill to confirm the exact signatures of the CODAB loader and `open_blob_cog` — do not guess them. Per the team KB, daily FloodScan COGs live in the dev blob `global` container at `raster/cogs/aer_area_300s_{YYYYMMDD}_v05r01.tif` (band 1 = SFED, fractional flooded area 0–1).

- [ ] **Step 1: Write `experiments/gfds_vs_floodscan.py`**

```python
"""Compare GFDS DIY anomaly vs FloodScan SFED at admin2, Pakistan 2022.
Run: uv run python experiments/gfds_vs_floodscan.py
"""
from pathlib import Path

import numpy as np
import ocha_stratus as stratus
import pandas as pd
import rioxarray  # noqa: F401 (registers .rio accessor)
import xarray as xr
from rasterstats import zonal_stats

OUT = Path("outputs/gfds_validation")
anom = xr.open_dataarray(OUT / "pak_anomaly_static.nc")
anom = anom.rio.write_crs("EPSG:4326").rename({"y": "y", "x": "x"})

# CODAB adm2, clipped to the study bbox — confirm loader signature
# via the /ocha-stratus skill before running.
adm2 = stratus.codab.load_codab_from_blob("pak", admin_level=2)
adm2 = adm2.cx[66.0:71.0, 24.0:29.0].reset_index(drop=True)

dates = pd.to_datetime(anom.time.values)
rows = []
for ts in dates:
    day = anom.sel(time=ts)
    gf = [s["mean"] for s in zonal_stats(
        adm2, day.values, affine=day.rio.transform(),
        nodata=np.nan, all_touched=True, stats="mean")]
    sfed_name = f"raster/cogs/aer_area_300s_{ts:%Y%m%d}_v05r01.tif"
    sfed = stratus.open_blob_cog(sfed_name, container_name="global")
    sfed = sfed.sel(band=1).rio.clip_box(66, 24, 71, 29)
    fs = [s["mean"] for s in zonal_stats(
        adm2, sfed.values, affine=sfed.rio.transform(),
        nodata=sfed.rio.nodata, all_touched=True, stats="mean")]
    for i, (g, f) in enumerate(zip(gf, fs)):
        rows.append({"date": ts, "adm2_idx": i, "gfds": g, "sfed": f})

df = pd.DataFrame(rows)
per_adm = []
for i, sub in df.groupby("adm2_idx"):
    sub = sub.dropna()
    if len(sub) < 30:
        continue
    r = sub["gfds"].corr(sub["sfed"])
    lag = (sub.loc[sub["gfds"].idxmax(), "date"]
           - sub.loc[sub["sfed"].idxmax(), "date"]).days
    flooded = sub["sfed"].max() > 0.05  # SFED peak >5% => really flooded
    per_adm.append({"adm2_idx": i, "pearson_r": r,
                    "peak_lag_days": lag, "flooded": flooded})
res = pd.DataFrame(per_adm)
res.to_csv(OUT / "adm2_comparison.csv", index=False)

fl = res[res.flooded]
print(res.to_string())
print(f"\nflooded adm2s: {len(fl)} | median r: {fl.pearson_r.median():.2f} "
      f"| median |peak lag|: {fl.peak_lag_days.abs().median():.0f} days")
```

- [ ] **Step 2: Run it** — `uv run python experiments/gfds_vs_floodscan.py` (needs `.env` with team blob credentials — see repo README / ocha-stratus skill).

- [ ] **Step 3: Record the gate numbers.** The decision gate (used in Task 9): over flooded adm2s, **median Pearson r ≥ 0.6** and **median |peak lag| ≤ 4 days** = promising; r < 0.4 = stop, GFDS is not tracking FloodScan.

- [ ] **Step 4: Commit** — `git add experiments/gfds_vs_floodscan.py && git commit -m "feat(gfds): admin2 validation against FloodScan SFED"`

---

### Task 8: Day-of-year climatology (our own baseline)

**Files:**
- Create: `experiments/gfds_climatology.py`

**Interfaces:**
- Consumes: `fetch_stack` (this is the big pull: 2015-01-01 → 2024-12-31 over the same bbox ≈ 3,650 windowed reads ≈ 1–2 h first run, resumable, ~50 MB cached).
- Produces: `data/gdacs_gfds/pak_doy_clim.nc` with `mean` and `sd` per (dayofyear, y, x); reruns Task 6/7's comparison with the DOY baseline → `adm2_comparison_doy.csv`.

- [ ] **Step 1: Write `experiments/gfds_climatology.py`**

```python
"""Per-pixel day-of-year climatology from the 2015+ AMSR2/GPM era,
then re-run the FloodScan comparison with it.
Run: uv run python experiments/gfds_climatology.py
"""
import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ds_flood_gfm.datasources.gfds import fetch_stack

BBOX = (66.0, 24.0, 71.0, 29.0)
CACHE = Path("data/gdacs_gfds/pak_hist")
OUT = Path("outputs/gfds_validation")
WINDOW = 10  # +/- days around each DOY

stack = fetch_stack(datetime.date(2015, 1, 1), datetime.date(2024, 12, 31),
                    BBOX, CACHE, kind="signal")
# Exclude the validation event itself from its own baseline:
stack = stack.sel(time=~((stack.time.dt.year == 2022)
                         & stack.time.dt.month.isin([6, 7, 8, 9, 10])))

vals = stack.values                       # (time, y, x) ~ (3500, 56, 56)
doys = stack.time.dt.dayofyear.values
mean = np.full((366,) + vals.shape[1:], np.nan)
sd = np.full_like(mean, np.nan)
for d in range(1, 367):
    # circular +/- WINDOW day membership (wraps around new year)
    sel = np.abs(((doys - d + 183) % 366) - 183) <= WINDOW
    sub = vals[sel]
    n_valid = np.isfinite(sub).sum(axis=0)
    m = np.nanmean(sub, axis=0)
    s = np.nanstd(sub, axis=0)
    enough = n_valid >= 30  # do not fabricate stats from thin samples
    mean[d - 1] = np.where(enough, m, np.nan)
    sd[d - 1] = np.where(enough, s, np.nan)

clim = xr.Dataset(
    {"mean": (("dayofyear", "y", "x"), mean),
     "sd": (("dayofyear", "y", "x"), sd)},
    coords={"dayofyear": np.arange(1, 367),
            "y": stack.y.values, "x": stack.x.values},
)
clim.to_netcdf("data/gdacs_gfds/pak_doy_clim.nc")
print("climatology written; NaN share in mean:",
      float(np.isnan(mean).mean()).__round__(3))

# DOY-baseline anomaly for the 2022 event window
event = fetch_stack(datetime.date(2022, 6, 1), datetime.date(2022, 10, 31),
                    BBOX, Path("data/gdacs_gfds/pak2022"), kind="signal")
doy = event.time.dt.dayofyear
mu = clim["mean"].sel(dayofyear=doy)
sg = clim["sd"].sel(dayofyear=doy).clip(min=0.005)
anom_doy = ((mu - event) / sg).rename("anomaly_doy")
anom_doy.to_netcdf(OUT / "pak_anomaly_doy.nc")
print("DOY anomaly written:", OUT / "pak_anomaly_doy.nc")
```

- [ ] **Step 2: Run it** (first run is the long one; it is resumable — rerun after any crash and it continues from cache).

- [ ] **Step 3: Re-run the Task 7 comparison against `pak_anomaly_doy.nc`.** Copy `experiments/gfds_vs_floodscan.py`'s logic, swapping the input file and writing `adm2_comparison_doy.csv`. Print both result tables side by side: the DOY baseline should beat the static 2009 baseline (higher median r over flooded adm2s). If it doesn't, that finding goes into the memo verbatim.

- [ ] **Step 4: Commit** — `git add experiments/gfds_climatology.py && git commit -m "feat(gfds): DOY climatology baseline and re-validation"`

---

### Task 9: Validation memo + decision gate

**Files:**
- Create: `docs/gfds-validation-memo.md`
- Modify: `docs/gdacs-gfds-assessment.md` (add a one-line link to the memo at the top)

- [ ] **Step 1: Write the memo** with exactly these sections, filling in your measured numbers — no adjectives without a number next to them:
  1. **What was tested** — event, bbox, dates, both baselines.
  2. **Results table** — per-admin2 median r and |peak lag| for static vs DOY baseline, flooded vs non-flooded adm2s.
  3. **Gate verdict** — against the Task 7 thresholds (median r ≥ 0.6 and |lag| ≤ 4 d = go; r < 0.4 = no-go; between = discuss).
  4. **Surprises / data quality notes** — missing days encountered, artifacts seen in plots.
  5. **Recommendation** — if GO: next plan is the backfill pipeline (2015+ signal → uint16 ZSTD COGs → blob, ~55 GB, mirroring `floodscan-ingest`), plus an ADR proposing GFDS adoption. If NO-GO: close with the numbers that killed it.

- [ ] **Step 2: Commit and open a PR**

```bash
git add docs/
git commit -m "docs(gfds): validation memo and go/no-go verdict"
git push -u origin feat/gfds-prototype
gh pr create --title "GFDS prototype: access layer + Pakistan 2022 validation" --fill
```

---

## Self-Review (done at plan-writing time)

- **Spec coverage:** assessment doc's recommendation (prototype ingest, own climatology, FloodScan comparison over a known event, gate before backfill) → Tasks 1–5 (access), 6 (event), 7 (comparison + gate), 8 (climatology), 9 (gate memo). Backfill intentionally deferred. ✓
- **Placeholder scan:** all code blocks complete; the two "confirm via /ocha-stratus skill" notes are deliberate verification instructions (external API owned by another package), not placeholders. ✓
- **Type consistency:** `read_gfds(path_or_url, kind, bbox)` used identically in Tasks 3, 4, 6; `fetch_stack(start, end, bbox, cache_dir, kind)` identical in Tasks 4, 6, 8; anomaly sign convention (flood-positive) consistent across Tasks 5, 6, 8. ✓
