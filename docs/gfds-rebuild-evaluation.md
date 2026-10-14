# Rebuilding GFDS ourselves: what it would take

*2026-10-14. This document evaluates options; nothing has been built or
decided. Companion docs:
[gdacs-gfds-assessment.md](gdacs-gfds-assessment.md) (status and evidence),
[gfds-nga2022-findings.md](gfds-nga2022-findings.md) and
[gfds-downscaling-poc.md](gfds-downscaling-poc.md).
The fusion v1 numbers below come from branch `feat/flood-fusion-v1`, which is
not on `main` yet: `docs/decisions.md` (D-07, D-19, D-37) and
`docs/final_report.md` (ablation).*

## Bottom line

- **"Maintaining GFDS" is not an option we have.**
  - We do not have JRC's code.
  - Every input GFDS was built on is gone or going:
    - AMSR-E and TRMM are dead.
    - NASA changed GMI's file format on 2026-03-04, within days of GFDS's
      last GPM file (2026-02-28).
    - AMSR2 is being retired in favour of AMSR3.
  - Any continuation is a rebuild on new inputs.
- **The algorithm is small.** JRC's 2015 tech note specifies it fully: one
  ratio per pixel, a 9×9 percentile, and daily averaging. The work and the
  risk sit in two places:
  - **inputs:** swath files from two agencies, two logins and two formats;
  - **operations:** catching upstream changes like the ones that ended GFDS.
- **The value case is narrow but real.**
  - Fusion v1 does not need a rebuild where FloodScan exists: GFDS added
    0.004 CSI over FloodScan alone, mostly as false alarms.
  - A rebuild matters for places FloodScan never covers (Europe), and as a
    hedge if FloodScan itself is disrupted. FloodScan rides the same retiring
    sensor, AMSR2.
  - It also matters as a free FloodScan analogue for admin-level monitoring
    (the book's case).
- **Do now, whatever else we decide: archive the JRC record.** It is still
  downloadable. It is also the only reference for validating any rebuild.

## 1. What happened

| date | event | source |
|---|---|---|
| ~2015 | Development ended; the pipeline ran unmaintained since | JRC, 2026-09-04 ([assessment](gdacs-gfds-assessment.md)) |
| 2025-08-11 | AMSR3 (GOSAT-GW) data record starts | [JAXA TB catalogue](https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/) |
| 2026-02-28 | Last GFDS GPM file (`signal_20260228_GPM_HDF5.tif`) | server listing |
| 2026-03-04 | NASA GMI L1B/1CR to V08, "format change from HD5 to NetCDF4 and corresponding filename change" | [NOAA OSPO](https://ospo.noaa.gov/data/messages/2026/02/MSG_20260220_2002.html) |
| Mar–Sep 2026 | GFDS `ALL` = AMSR2 only (byte-identical file sizes) | server listing |
| 2026-06-30 | AMSR3 L1 ver. 01A public on G-Portal | [JAXA](https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/) |
| 2026-09-02 | NOAA ends AMSR2 data and product generation (AMSR3 transition) | [NOAA OSPO](https://www.ospo.noaa.gov/products/suites/gaasp/maps.html) |
| 2026-09-14 | Last GFDS file (partial day, 16 MB instead of ~23 MB) | server listing |
| 2026-09 | DMSP SSMIS program end planned | [NSIDC](https://nsidc.org/data/user-resources/data-announcements/user-notice-ssmis-processing-will-now-continue-through-september-2026) |

The timing of the two stops is consistent with the two upstream changes, but
the cause is not confirmed. A one-line question to JRC would settle it.

## 2. What GFDS computes

From the [2015 Technical Note](https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf)
(De Groeve, Brakenridge, Paris; JRC97421):

1. **Input.** 36.5 GHz H-pol brightness temperature (Tb), from ascending and
   descending swaths, per sensor. TMI and GMI are listed as "37 GHz".
2. **Per swath, per measurement pixel M:**
   - The calibration value C is the **95th percentile of Tb in the 9×9 pixel
     window centred on M**, in swath geometry. The note says it uses the 95th
     percentile "not the hottest pixel", to exclude outliers.
   - The position P (1–81) records which window cell supplied C.
3. **Signal.** s = T_M / T_C. Eq. 4 gives s ≈ 1 − w + w·ε_water/ε_land, which
   is monotone in the water fraction w. Because M and C come from the same
   swath at the same moment, physical temperature and atmosphere cancel to
   first order. That cancellation is why the ratio, not Tb, is the product.
4. **Gridding.** Swath values are projected onto the 4000×2000 global grid
   (0.09°).
   - Per day: s and m are averaged over all samples; M, C and P keep the
     last sample.
5. **Merge (`ALL`).** Per-sensor daily s values are averaged. M, C and P are
   not stored for the merge.
6. **Magnitude.**
   - m = (s − avg)/sd, against a per-pixel 2002–2008 baseline. The files are
     dated 2009.
   - The sd in use is floored at 0.005 (see the assessment's 2026-10-14
     status).
7. **4-day product.** A trailing mean of the daily values (`Avg*` folders).

**What we would actually need to rebuild.** Only step 1–5's daily `s`.
- The Nigeria work showed that the signal plus our own baseline is the
  product ([findings](gfds-nga2022-findings.md) §"What this settles").
- Fusion's R1 rule turns `s` into a flooded fraction with no JRC baseline:
  `f = (s_ref − s)/(s_ref·K)` (D-07).
- JRC's magnitude and its frozen 2009 baseline are things we already chose
  not to use.

**What the note leaves unspecified.** We would have to choose each of these
and validate the choice:
- how swath footprints are assigned to grid cells (its Fig. 6 shows
  footprints overlapping cells, but no rule);
- how the 9×9 window behaves at swath edges and coasts;
- the "½ pixel systematic shift" among sensors that JRC itself flags;
- whether the merge weights sensors.

## 3. Inputs available now

| sensor | 36 GHz channel | coverage | status (2026-10-14) | access |
|---|---|---|---|---|
| **AMSR3** (GOSAT-GW, JAXA) | 36.42 GHz (L3 label `TL7`) | global | Data from 2025-08-11; L1 ver. 01A public since 2026-06-30. The natural AMSR2 successor. | JAXA G-Portal, free registration; NetCDF. The team KB records no G-Portal account. |
| **GMI** (GPM, NASA/JAXA) | 37 GHz class (tech note) | 65°S–65°N orbit (swath edge reaches further, see below) | Operating; V08 since 2026-03-04, NetCDF4. No end of mission announced. | NASA PPS (near real time) or GES DISC `GPM_1CGMI`; Earthdata login |
| **AMSR2** (GCOM-W, JAXA) | 36.5 GHz | global | NOAA stopped AMSR2 products on 2026-09-02. WMO OSCAR lists GCOM-W end of life in 2026; JAXA's catalogue still lists the data as open-ended. Plan as ending. | G-Portal L1B/L1R (HDF5). GES DISC `GPM_1CGCOMW1AMSR2` V08 (intercalibrated L1C, Earthdata) |
| SSMIS (DMSP) | 37 GHz | global | Program end planned for September 2026; gaps since 2025 | Not a foundation |
| WSF-M, FY-3 MWRI | 36–37 GHz | — | Not assessed | — |

**Credentials.**
- The team's `dsci` Earthdata account is used by the IMERG pipeline. Fusion
  v1 found that NASA's token API rejects it (HTTP 401, D-37), so a scheduled
  job needs that fixed.
- G-Portal would be a new account.

**Measured: what a single sensor loses.** We compared one day,
2025-08-15, from JRC's own per-sensor signal files. Land pixels come from
GFDS's `oceanmask.tif`. "Valid" means not a sentinel and signal > 0.

| land pixels valid | AMSR2 | GMI | `ALL` (merge) |
|---|---|---|---|
| north of 65°N | 100% | 48% | 100% |
| 65°S–65°N | 91.5% | 79.8% | 98.4% |
| south of 65°S | 96% | 5% | 96% |

Where both sensors observed the same land pixel (65°S–65°N, n = 1.15 M):
- median difference GMI − AMSR2 = +0.0001;
- IQR −0.0054 to +0.0062;
- r = 0.75.

For scale, flooded cells in Nigeria 2022 sat 0.08–0.12 below the baseline
(D-07). So the ratio really is close to sensor-independent in level, which is
what makes a sensor switch survivable. This is **one day and one sensor
pair**. Repeat it on several seasons, and on AMSR3 against AMSR2, before
relying on it. Reproduce with
`experiments/gfds_rebuild/01_sensor_coverage.py`.

## 4. Options

### A. Archive only — do this now

- **What.** Mirror the JRC server to blob:
  - `ALL/SignalTiffs` (1997-12 → 2026-09-14);
  - per-sensor `AMSR2/` and `GPM/` `SignalTiffs` and `SourceTiffs`, plus
    `CalibrationTiffs` and `PositionTiffs` where they exist (GPM 2015–2026,
    AMSR2 2013 → 2018-09);
  - `SINGLE/` too, if cheap: it continued alongside AMSR2 after GPM stopped.
  - `ALL/AveragesAndSd`, `oceanmask.tif`, `SpecialTiffs`, and the tech note.
- **Why the per-sensor folders matter.** They are the answer key for a
  rebuild. They record, per pixel and day, JRC's T_M, T_C and which of the 81
  cells it picked. With them, a rebuild can be checked step by step, not
  only on the final ratio.
- **Size and time.** These are estimates from listed file sizes; size the job
  from the listings before starting.
  - `ALL` signal is about 210 GB (assessment).
  - AMSR2 signal is about 115 GB (~5,000 days × 23 MB).
  - GPM signal is about 70 GB (~4,000 days × 17 MB).
  - Source, Calibration and Position files are of the same order each, so
    a full mirror is probably under ~1 TB.
  - Fusion measured about 0.3 MB/s per connection (D-19). At 4 connections
    that is about 100 GB a day: about 4 days for the signal sets, and 1–2
    weeks for everything.
- **Effort.** 1–2 days to write a resumable mirror; the prefetch and
  `phase6_package.py` patterns already exist. Then a polite download.
- **Licensing.** The prior docs disagree. The assessment expected CC BY. The
  fusion inventory found only an "as is" disclaimer. The tech note says
  "Reproduction is authorised provided the source is acknowledged." An
  internal archive is fine; check before redistributing.
- **What it does not do.** It produces no new days. Fusion's GFDS input stays
  NoData after 2026-09-14, as it is now.
- **Note.** Fusion's 5.5 GB GFDS row-band store covers event windows only.
  It is not an archive.

### B. Rebuild the signal from AMSR3

- **What.** A daily job:
  1. Fetch AMSR3 L1B/L1R swaths from G-Portal.
  2. Take the 36 GHz H channel.
  3. Per swath, compute C as the 95th percentile of the 9×9 window, then
     s = M/C.
  4. Grid to the same 4000×2000 0.09° grid. Fusion then takes it unchanged:
     the 9×9 nesting and the R1 rule (D-03, D-07).
  5. Average the day and write a COG to blob, following the
     `floodscan-ingest` pattern.
  6. Build our own day-of-year climatology from the start, not JRC's 2009
     baseline.
- **Validation is the real work, and the archive makes it possible.**
  1. Run the same code on **AMSR2** L1 for months where JRC's AMSR2 files
     exist. For 2013–2018, compare M, C, P and s pixel by pixel; for later
     years, compare M and s (`SourceTiffs` run to 2026). This tests our
     reading of the tech note against JRC's output.
  2. Run it on **AMSR3** over the AMSR2/AMSR3 overlap (2025-08-11 to at least
     2026-09-01) and measure the step at the sensor switch. AMSR3 is
     36.42 GHz and a new instrument (ver. 01A), so a small level shift is
     plausible. The ratio should absorb most of it; measure it rather than
     assume it.
  3. Re-run the Nigeria 2022 LGA comparison and the fusion GFDS-only
     ablation on the rebuilt signal.
- **Effort** (judgement, not measured):
  - about 4–8 person-weeks to a validated prototype; input plumbing and
    swath gridding are most of it;
  - then 2–4 weeks to put it into operation (Databricks job, liveness and
    absence checks).
- **Ongoing cost.** Each upstream change (format, version, endpoint, shortname)
  needs a fix. 2026 alone had two that matter here: GMI's V08 NetCDF4 change,
  and NOAA's AMSR shortname changes. Budget a few days per change and alert on
  absence. The fusion daily job's per-day absence record is the right pattern.
  Silent staleness is how GFDS ended.
- **Loses against JRC.** Multi-sensor sampling. On the measured day the merge
  lifted land coverage in 65°S–65°N from 91.5% to 98.4%. Single-sensor days
  will have swath gaps: either use the 4-day mean, or accept NoData, which
  fusion already handles.

### C. Rebuild with AMSR3 + GMI merged (closest to JRC's `ALL`)

- **Adds.**
  - More samples per day across 65°S–65°N.
  - A fallback if one sensor fails. GMI alone is not a fallback north of
    ~70°N.
  - A second validation target: JRC's GPM files with full M/C/P for
    2015 to 2026-02.
- **Costs.**
  - A second agency, login and format.
  - Footprint and frequency differences, including the half-pixel shift JRC
    never fixed.
  - Validating the merge itself against JRC `ALL`.
  - Roughly double the surface for upstream changes.
- **Effort.** B plus about 3–6 person-weeks. Do it only after B validates, and
  only if the coverage gain matters for the use.

### D. Don't build; rely on others

- **DFO / University of Colorado.**
  - JRC said a handover to DFO (Brakenridge, Kettner) has been discussed for
    years.
  - The DFO River Watch station pages we found show readings only to October
    2025, and it is not clear whether DFO runs its own M/C processing.
  - Ask Kettner directly whether they run, or plan, an AMSR3/GMI
    continuation. If they do, B and C may be redundant, and archiving (A)
    still stands.
- **FloodScan (AER).**
  - FloodScan uses the same sensor family. Book chapter 05 lists AMSR2 and
    GMI; the assessment table says AMSR2/SSMIS. Either way it depends on
    AMSR2, which is ending.
  - We found no AER statement about an AMSR3 transition. Ask AER.
  - Our blob's FloodScan outside Africa stops at 2026-03-31, but that is an
    **ingest gap, not a product end**. Later data exist as AER 90-day zips on
    the dev blob, or through the AER API; they are just not integrated
    (fusion `final_report.md` §7). Closing that gap is far cheaper than
    rebuilding GFDS.

## 5. Is it worth it?

**For fusion v1.** Held-out gold, from the fusion branch ablation:

| microwave input | CSI |
|---|---|
| FloodScan OR GFDS (v1 as built) | 0.285 |
| FloodScan only | 0.281 |
| GFDS only | 0.261 |
| no microwave | 0.244 |

- Where FloodScan exists, GFDS adds 0.004 CSI: 6,656 hits for 17,246 false
  alarms.
- Where it is the only microwave input, it is worth about +0.017 CSI over
  nothing.
- So the fusion value of a rebuild is confined to regions FloodScan never
  covers, which essentially means **Europe**. Outside Africa after
  2026-03-31, the cheaper fix is ingesting the FloodScan data that already
  exist (§4 D).
- Europe is also where CEMS gold is densest, and where fusion v1 found GFM
  alone beats fusion.

**For admin-level monitoring** (the book's framing):
- Nigeria 2022: LGA r = 0.88–0.92 by severity band, and 0.89–0.97 at the
  reference LGAs.
- The physics R1 route matched SFED at r = 0.84, with 99% of peak area.
- That is promising, but it is one event in one basin, and not yet shown
  elsewhere.

**Recommendation.**
1. **A now.** It is cheap, urgent and has no regret: the server is
   unmaintained and its viewer pages already return 404.
2. **Ask DFO and AER** this week. Their answers change B's value more than
   anything we can measure.
3. **B as a time-boxed feasibility test** (about 2 weeks) before any
   commitment to operations:
   1. Get a G-Portal account.
   2. Pull one month of AMSR2 L1R and AMSR3 L1R from the overlap.
   3. Implement M/C.
   4. Score against JRC's AMSR2 files for the same month.

   The pass criteria and what "good enough" means should be agreed before
   running it.
4. **C only if B passes** and the coverage gain matters for the use.
5. Record the choice as an ADR once made. It has real rejected alternatives,
   and `main` has no `docs/decisions/` yet; the fusion branch has one.

## Open questions

- Did the GMI format change and the AMSR2 retirement cause the stops? Ask JRC.
- What is the AMSR3 near-real-time latency on G-Portal? Not checked.
- Is AMSR2 L1 still flowing on G-Portal after 2026-09-02? JAXA lists it as
  open-ended; check the latest granule date.
- Does a GPM-constellation intercalibrated AMSR3 L1C exist, or is one
  planned? None was found. It would give one login and one format for both
  sensors.
- Virtual-gauge API: is it empty because the feed stopped, or broken? We
  have no known-good query to tell.
- Licence for redistributing derived products.

## Sources

- JRC, *Global Flood Detection System: Data Product Specifications*, v2015:
  <https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf>
- GDACS GFDS data server and listings: <https://www.gdacs.org/flooddetection/DATA/>
- NOAA OSPO, GPM V08 / NetCDF4 change (2026-02-20 notice):
  <https://ospo.noaa.gov/data/messages/2026/02/MSG_20260220_2002.html>
- NOAA OSPO, AMSR2 to AMSR3 transition: <https://www.ospo.noaa.gov/products/suites/gaasp/maps.html>,
  <https://ospo.noaa.gov/data/messages/2026/07/MSG_20260728_1045.html>,
  <https://www.nesdis.noaa.gov/news/decommissioning-of-advanced-microwave-scanning-radiometer-amsr2-products-and-transition-amsr3-products>
- JAXA AMSR brightness temperature catalogue (AMSR2, AMSR3): <https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/>
- WMO OSCAR, GCOM-W: <https://space.oscar.wmo.int/satellites/view/gcom_w>
- NASA GES DISC, `GPM_1CGCOMW1AMSR2` V08: <https://www.earthdata.nasa.gov/data/catalog/ges-disc-gpm-1cgcomw1amsr2-08>
- NSIDC, SSMIS processing through September 2026:
  <https://nsidc.org/data/user-resources/data-announcements/user-notice-ssmis-processing-will-now-continue-through-september-2026>
- DFO River and Reservoir Watch: <https://floodobservatory.colorado.edu/>
