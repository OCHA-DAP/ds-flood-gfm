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

- References: [Technical Note — GFDS Data Product Specifications, v2015](https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf)
  (De Groeve, Brakenridge, Paris; JRC97421) and the fuller 2007 system
  report with validation, [Kugler & De Groeve, EUR 23303](https://www.unisdr.org/files/9622_LBNA23303ENC002.pdf):
  against 58 major 2002–2007 flood events, 35–42% of monitored sites gave a
  clean reliable signal, ~a third detected through noise, 14–16% missed;
  the 4–6σ extreme-flood thresholds were set empirically ("needs to be
  confirmed with further research"); Sahel irrigation is a documented
  false-alarm source.
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

## Archive completeness (verified 2026-08-17)

Spot checks across the full record confirm the archive is genuinely complete,
not nominal:

- `signal_*_ALL.tif` and `mag_signal_*_ALL.tif` respond for 1998, 2003, 2010,
  2015, 2020, 2024, and today
- A sampled month (2010-08) has all 31 daily files with no gaps
- Era caveat: 1997–2002 files are ~10 MB vs ~25 MB later — the TRMM-only era
  covers only 40°N–40°S; global coverage starts with AMSR-E (June 2002)
- `DATA/ALL/AveragesAndSd/` exposes the baseline avg/sd rasters themselves
  (files dated 2009–2014 — i.e., the operational baseline really is that old)
- Volume estimate for ingestion: ~10,400 days × ~20 MB ≈ **~210 GB per
  variable** (signal or magnitude, daily); 4-day `Avg*` variants double that

## Verdict

**Not usable as a flood-extent or exposure source** — 10 km anomaly pixels
cannot support admin-level population exposure, which is this repo's purpose.
GFM stays.

**Promising as a free FloodScan analogue for anomaly monitoring — provided we
compute our own climatology.** The team's FloodScan use (floodexposure
monitoring, return-period baselines) is anomaly-shaped, and GFDS is the
closest free, open, observation-based equivalent: same passive-microwave
family, daily, 3 h latency, and a complete 1998→present archive on plain
predictable HTTP URLs. The key design decision:

- **Do not consume JRC's magnitude product** — its 2002–2008 baseline is
  visibly broken (2 % of land >4σ on a normal day, half of it snow/ice
  artifacts).
- **Ingest the raw `signal` rasters and derive our own per-pixel, day-of-year
  climatology/quantiles** from the 28-year archive — the same pattern as the
  FloodScan SFED baselines. Signal is monotonic in pixel water fraction
  (eq. 4 of the tech note), so per-pixel quantiles and return periods are
  meaningful even though the absolute value isn't a flooded fraction.

Honest caveats that survive this framing:

1. **Sensor inhomogeneity across eras** (TRMM-only → AMSR-E → TRMM-only gap
   2011–2013 → AMSR2+GPM) will put discontinuities in any long climatology;
   baselines should probably be computed on the 2015+ AMSR2+GPM mix, or per
   sensor via the `SINGLE` product.
2. Signal ≠ flooded fraction: cross-pixel comparison of magnitudes is fine,
   cross-pixel comparison of *area* is not. FloodScan SFED remains the better
   product where a calibrated flooded fraction is needed.
3. Arid/snow masking is mandatory; high latitudes and deserts dominate raw
   exceedances.
4. Maintenance risk: docs frozen in 2015, `about.aspx` dead, deprecated API
   fields, and the live feed rides on one aging sensor pair (AMSR2 2012,
   GPM 2014). Worth a lightweight liveness check in any ingest pipeline.
5. Licensing: standard EC/JRC reuse policy (expected CC BY 4.0 with
   attribution) — verify before redistributing derived products.

**Recommendation:** worth a prototype. Natural shape: a `gfds-ingest` pipeline
mirroring `floodscan-ingest` (daily signal GeoTIFF → COG → blob), plus a
one-time archive backfill (~210 GB) and a climatology job producing per-pixel
day-of-year quantiles. Compare the resulting anomalies against FloodScan SFED
over known events (e.g. Pakistan 2022, Nigeria 2022/2024) before committing —
if agreement is good, this is a credible free fallback/replacement for the
FloodScan subscription. The virtual-gauge API and GDACS flood events
(`ocha-lens`) remain the quicker entry points for pure triggering.
