"""GFDS-style M/C signal from AMSR2 L3 36 GHz H-pol Tb (0.1 deg, one file per pass).

Method (JRC GFDS tech note v2015, applied on the L3 grid instead of in swath geometry):
  per pass:  C = 95th percentile of Tb_H in the 9x9 window centred on each cell
             s = Tb_H(M) / C
  per day:   s = mean of the valid passes (ascending, descending)

Choices the tech note does not make (validated in 05_compare_jrc.py):
- the window wraps in longitude and is NaN-padded at the poles;
- C needs >= MIN_VALID of the 81 window cells observed (single-pass grids have inter-orbit gaps),
  else NaN;
- the percentile is numpy's default linear interpolation over the observed cells.

L3 grid (checked against GFDS oceanmask, 0.979 land/ocean agreement within 55N-55S):
north-up, row 0 centred at 89.95N, column 0 centred at 0.05E. Output is rolled to start at
180W: GeoTIFF float32, transform (0.1, 0, -180, 0, -0.1, 90), NaN = not observed.

    uv run --no-project --with h5py --with numpy --with rasterio \
        python experiments/gfds_rebuild/04_signal_from_l3.py 2022-10-01 2022-10-31
"""

import datetime as dt
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.transform import Affine

SRC = Path("data/gdacs_gfds/rebuild/amsr2_l3_tb36")
OUT = Path("data/gdacs_gfds/rebuild/signal_l3")
WIN, P, MIN_VALID, CHUNK = 9, 0.95, 41, 100
FILL = 65534  # 65534 and above are no-data
TRANSFORM = Affine(0.1, 0, -180, 0, -0.1, 90)


def read_tb_h(path: Path) -> np.ndarray:
    with h5py.File(path) as h:
        ds = h["Brightness Temperature (H)"]
        scale = float(ds.attrs["SCALE FACTOR"][0])
        raw = ds[:]
    if raw.shape != (1800, 3600):
        raise ValueError(f"unexpected L3 grid {raw.shape} in {path}")
    tb = raw.astype("float32") * scale
    tb[raw >= FILL] = np.nan
    return np.roll(tb, 1800, axis=1)  # 0..360E -> -180..180


def window_p95(tb: np.ndarray) -> np.ndarray:
    """95th percentile over the observed cells of each 9x9 window; NaN if < MIN_VALID observed."""
    h = WIN // 2
    padded = np.concatenate([tb[:, -h:], tb, tb[:, :h]], axis=1)  # wrap longitude
    padded = np.pad(padded, ((h, h), (0, 0)), constant_values=np.nan)  # poles
    out = np.full(tb.shape, np.nan, "float32")
    for r0 in range(0, tb.shape[0], CHUNK):
        r1 = min(r0 + CHUNK, tb.shape[0])
        win = sliding_window_view(padded[r0 : r1 + 2 * h], (WIN, WIN)).reshape(r1 - r0, tb.shape[1], WIN * WIN)
        n = np.isfinite(win).sum(-1)
        v = np.sort(np.where(np.isfinite(win), win, -np.inf), axis=-1)  # observed cells sort to the end
        pos = (WIN * WIN - n) + P * (n - 1)  # numpy 'linear' index among the n observed values
        lo = np.clip(np.floor(pos).astype(int), 0, WIN * WIN - 1)
        hi = np.clip(np.ceil(pos).astype(int), 0, WIN * WIN - 1)
        frac = (pos - np.floor(pos)).astype("float32")
        vlo = np.take_along_axis(v, lo[..., None], -1)[..., 0]
        vhi = np.take_along_axis(v, hi[..., None], -1)[..., 0]
        c = vlo + frac * (vhi - vlo)
        c[n < MIN_VALID] = np.nan
        out[r0:r1] = c
    return out


def day_signal(day: dt.date) -> tuple[np.ndarray, int]:
    files = sorted((SRC / f"{day:%Y}" / f"{day:%m}").glob(f"GW1AM2_{day:%Y%m%d}_01D_EQM[AD]_*.h5"))
    if len(files) != 2:
        raise FileNotFoundError(f"{day}: expected 2 pass files, found {len(files)}")
    sigs = []
    for f in files:
        tb = read_tb_h(f)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigs.append(tb / window_p95(tb))
    stack = np.stack(sigs)
    n = np.isfinite(stack).sum(0)
    s = np.where(n > 0, np.nansum(stack, 0) / np.maximum(n, 1), np.nan).astype("float32")
    return s, len(files)


start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
for k in range((end - start).days + 1):
    day = start + dt.timedelta(k)
    out = OUT / f"{day:%Y}" / f"s_{day:%Y%m%d}.tif"
    if out.exists():
        continue
    s, _ = day_signal(day)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=1800, width=3600, count=1, dtype="float32",
                       crs="EPSG:4326", transform=TRANSFORM, nodata=np.nan, compress="deflate", tiled=True) as dst:
        dst.write(s, 1)
    tmp.rename(out)
    print(f"{day}  valid {np.isfinite(s).mean():.3f}  median s {np.nanmedian(s):.4f}", flush=True)
