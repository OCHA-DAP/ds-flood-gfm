# GDACS Global Flood Detection System (GFDS) — Data Source Assessment

*Assessed 2026-08-17. All access patterns below were verified hands-on that day
(files downloaded, rasters opened, API queried).*

## What it is

The [GDACS flood detection product](https://www.gdacs.org/flooddetection/download.aspx)
is **GFDS** — the JRC / Dartmouth Flood Observatory *Global Flood Detection
System*, operational since 2006. It is **not** flood-extent mapping: it infers
surface-water anomalies from **passive microwave brightness temperature**
(36.5 GHz, H-pol). For each monitored "measurement" pixel over a river, the
brightness temperature is ratioed against a nearby dry "calibration" pixel
(the *M/C* ratio, or *signal* `s`); flood **magnitude** `m` is the number of
standard deviations of `s` above that pixel's long-term mean.

- Reference: [Technical Note — GFDS Data Product Specifications, v2015](https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf)
  (De Groeve, Brakenridge, Paris; JRC97421)
- Sensors: AMSR-E (2002–2011, dead), TRMM-TMI (1997–2015, dead),
  **AMSR2 (2013–ongoing)**, **GPM-GMI (2015–ongoing)**
- Latency: ~3 h (AMSR2) to ~24 h (GPM); site updates every 3 hours

## Products and access (verified)

### 1. Daily global rasters

`https://www.gdacs.org/flooddetection/DATA/{PRODUCT}/{FOLDER}/{YYYY}/{MM}/{name}_{YYYYMMDD}_{PRODUCT}.tif`

- `PRODUCT`: `ALL` (multi-sensor merge — recommended operationally), `SINGLE`,
  `GPM`, `AMSR2` (+ dead `TRMM`, `AMSR-E`)
- Folders / names: `SignalTiffs/signal_*` (s ×1,000,000),
  `MagTiffs/mag_signal_*` (m ×1,000, clipped ±20σ), plus `SourceTiffs`,
  `CalibrationTiffs`, `PositionTiffs` and 4-day smoothed `Avg*` variants
- Format (inspected `signal_20260816_ALL.tif` / `mag_signal_20260816_ALL.tif`):
  GeoTIFF, **4000×2000 @ 0.09° (~10 km)**, EPSG:4326, int32, LZW, ~19–23 MB/day
- Archive: December 1997 → **current** (today's file existed, written intraday)
- ⚠️ Data hygiene: declared nodata is −32000, but rasters *also* contain
  int32-min (−2147483648) sentinels — mask both

### 2. Virtual-gauge time-series API (~11,000 fixed river sites)

- v3: `flooddetection/data.aspx`, v4: `flooddetection/data_v2.aspx?source=DFO|DFOMERGE|GPM`
- Params: `areaid`, `from`/`to`, `datatype=DAILY|4DAYS`,
  `alertlevel=GREEN|ORANGE|RED`, `type=txt|html|rss|kml`
- Returns per-site signal/magnitude time series; RED ≈ m > 4. Verified live,
  no auth, current through yesterday.

## Empirical check: the raw magnitude raster is noisy

On 2026-08-16 (`ALL` merge), **150,189 pixels (~2 % of valid land) exceeded the
RED threshold (m > 4σ)**. Their distribution:

| Region | Share of m>4 pixels |
|---|---|
| High latitude N (>55°N) | 33 % |
| High latitude S (<50°S) | 19 % |
| Sahara / Arabia (arid) | 6 % |
| S Asia + SE Asia (monsoon, mid-Aug) | 2 % |
| Other | 40 % |

More than half of global "red alert" pixels are snow/ice and arid-land
artifacts. The raster is unusable as an alert layer without aggressive masking
— which is exactly why JRC's operational product is the *curated ~11k site
network*, not the raw grid. The 2002–2008 statistical baseline (per the 2015
tech note) has never been re-documented against the post-2015 AMSR2+GPM sensor
mix, which likely contributes to the drift.

## How it compares to what we already use

| | **GFM (this repo)** | **FloodScan (team ingest)** | **GFDS** |
|---|---|---|---|
| Basis | Sentinel-1 SAR | Passive microwave (AMSR2/SSMIS) | Passive microwave (AMSR2/GPM) |
| Output | 20 m flood extent | ~8.3 km fractional flooded area (SFED/MFED) | ~10 km signal/σ-anomaly |
| Cadence | 1–3 day revisit gaps | Daily | Daily, 3 h latency |
| Archive | 2015→ | 1998→ | 1997→ |
| Cost | Free | Licensed (AER) | Free |
| Usable for exposure? | Yes (core use) | Yes, coarsely | **No** — anomaly, not extent |

GFDS is essentially a free, rawer cousin of FloodScan from the same sensor
family: same ~10 km scale, but it delivers a *statistical anomaly* rather than
a calibrated *fractional flooded area*, with no documented recalibration since
2015 and visible baseline noise.

## Verdict

**Not promising as a flood-extent or exposure source** — 10 km pixels cannot
support admin-level population exposure, which is this repo's purpose. GFM
stays.

**Marginally promising in exactly one role: a free, observation-based,
3-hour-latency global trigger.** The virtual-gauge API is lightweight
(semicolon-delimited text, no auth) and could flag *where* to prioritize
running the heavier GFM/Sentinel-1 pipeline, or corroborate FloodScan
anomalies with an independent observation. Even there, honest caveats:

1. GDACS itself already publishes curated flood *events* (which `ocha-lens`
   wraps) — for most triggering purposes that's the better-vetted entry point
   than raw GFDS sites.
2. The team already pays for FloodScan, which covers the same physical signal
   with better calibration. GFDS's edge is only cost (free), latency (3 h),
   and archive homogeneity for anomaly baselines.
3. Maintenance risk: docs frozen in 2015, `about.aspx` dead, several API
   fields marked deprecated, and the live feed depends on a single aging
   sensor pair (AMSR2 launched 2012; GPM 2014).

**Recommendation:** do not build on GFDS rasters. If an NRT trigger signal is
wanted for this pipeline, prototype against the virtual-gauge API (or GDACS
flood events via `ocha-lens`) before considering any raster ingestion.
