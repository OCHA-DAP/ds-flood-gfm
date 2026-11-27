# An open GFDS-style flood signal: what it would take

*2026-10-14; revised 2026-10-15. **Goal:** an open, FloodScan-like
product, meaning a daily ~10 km flooded fraction with a long archive for
baselines and return periods. It would be built on our own passive-microwave
signal, computed from public satellite data. Fusion v1 is one user of it, not
the reason for it. Nothing has been decided.*

*Companion docs:
- [gdacs-gfds-assessment.md](gdacs-gfds-assessment.md): JRC GFDS status and
  evidence;
- [gfds-nga2022-findings.md](gfds-nga2022-findings.md) and
  [gfds-downscaling-poc.md](gfds-downscaling-poc.md).

Fusion v1 numbers come from branch `feat/flood-fusion-v1`, which is not on
`main` yet: `docs/decisions.md` (D-07, D-19, D-37), `docs/final_report.md`
and `results/phase5_*.csv`.*

## Bottom line

- **We rebuild from source; we don't maintain JRC's GFDS.**
  - JRC's feed stopped on 2026-09-14, and we do not have their code.
  - The method is small and fully specified (§2).
  - The raw satellite data are public at JAXA.
- **The source is concrete.** JAXA's G-Portal serves 36 GHz data for two
  sensors. Both come as raw swaths and as a 0.1° daily grid (§3). A G-Portal
  account exists as of 2026-10-15.
  - **AMSR2:** July 2012 to now. It is still flowing; only NOAA's copy ended.
  - **AMSR3:** August 2025 to now. Swaths appear within about 2 hours.
- **JRC's archive is a reference, not a dependency.**
  - A few GB of JRC files are enough to check that our implementation
    matches theirs.
  - The full archive (~216 GB for the combined signal) is optional insurance
    (§4 A).
- **Feasibility splits three ways** (§5):
  - *Can we build it?* Yes.
  - *Can it match FloodScan's skill?* Partly shown. Even the crude version
    catches as many floods as FloodScan. The gap is false alarms, which is
    what AER's documented fixes address.
  - *Can we keep it running?* This is the deciding risk. It is what killed
    GFDS.
- **"Open" constrains calibration.** FloodScan can be a benchmark, never an
  input: products derived from it need AER's approval. Calibrate against GFM
  and the CEMS/UNOSAT flood maps instead.

## 1. What happened

| date | event | source |
|---|---|---|
| ~2015 | Development ended; the pipeline ran unmaintained after that | JRC, 2026-09-04 ([assessment](gdacs-gfds-assessment.md)) |
| 2025-08-11 | AMSR3 (GOSAT-GW) data record starts | [JAXA TB catalogue](https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/); G-Portal listing |
| 2026-02-28 | Last GFDS GPM file (`signal_20260228_GPM_HDF5.tif`) | GDACS listing |
| 2026-03-04 | NASA GMI L1B/1CR move to V08, "format change from HD5 to NetCDF4 and corresponding filename change" | [NOAA OSPO](https://ospo.noaa.gov/data/messages/2026/02/MSG_20260220_2002.html) |
| Mar–Sep 2026 | GFDS `ALL` = AMSR2 only (byte-identical file sizes) | GDACS listing |
| 2026-06-30 | AMSR3 L1 ver. 01A public on G-Portal | [JAXA](https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/) |
| 2026-09-02 | **NOAA** ends *its own* AMSR2 processing (AMSR3 transition) | [NOAA OSPO](https://www.ospo.noaa.gov/products/suites/gaasp/maps.html) |
| 2026-09-14 | Last GFDS file (partial day, 16 MB instead of ~23 MB) | GDACS listing |
| 2026-10-15 | **AMSR2 L1B/L1R and L3 still published by JAXA**, through the current day | G-Portal SFTP listing |

The GPM stop lines up closely with NASA's format change. The AMSR2 stop does
not line up with any end of AMSR2 data, because JAXA still publishes it. We
do not know where JRC took AMSR2 from. The cause of both stops is
unconfirmed; a one-line question to JRC would settle it.

## 2. What GFDS computes

From the [2015 Technical Note](https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf)
(De Groeve, Brakenridge, Paris; JRC97421):

1. **Input.** 36.5 GHz H-pol brightness temperature (Tb), from ascending and
   descending swaths, per sensor. TMI and GMI are listed as "37 GHz".
2. **Per swath, per measurement pixel M:**
   - The calibration value C is the **95th percentile of Tb in the 9×9 pixel
     window centred on M**, in swath geometry. The note uses the 95th
     percentile, "not the hottest pixel", to exclude outliers.
   - The position P (1–81) records which window cell supplied C.
3. **Signal.** s = T_M / T_C.
   - Eq. 4 gives s ≈ 1 − w + w·ε_water/ε_land, which is monotone in the water
     fraction w.
   - M and C come from the same swath at the same moment, so physical
     temperature and atmosphere cancel to first order. That cancellation is
     why the ratio, not Tb, is the product.
4. **Gridding.** Swath values are projected onto the 4000×2000 global grid
   (0.09°).
   - Per day: s and m are averaged over all samples; M, C and P keep the last
     sample.
5. **Merge (`ALL`).** Per-sensor daily s values are averaged.
6. **Magnitude.**
   - m = (avg − s)/sd, against a per-pixel 2002–2008 baseline. The files are
     dated 2009. (The tech note writes the difference the other way round; the
     exactness test below shows JRC's files use avg − s, so a flood, which
     lowers s, gives a positive magnitude.)
   - The sd is floored at 0.005.
7. **4-day product.** A trailing mean (`Avg*` folders).

**What we need.** Only the daily `s` from steps 1–5.
- Baselines we build ourselves.
- Flooded fraction comes from our own physics route,
  `f = (s_ref − s)/(s_ref·K)` (D-07), to be refined (§6 gate 2).

**What the note leaves unspecified,** which we choose and validate:
- how footprints map to grid cells;
- edge and coast behaviour of the 9×9 window;
- the "½ pixel systematic shift" between sensors;
- merge weights.

## 3. Inputs available now

### JAXA G-Portal

These facts come from the listing probe of 2026-10-15, made on the old
system: SFTP to `ftp.gportal.jaxa.jp:2051` with user ID and password.

**G-Portal system renewal (notice of 2026-10-01, read 2026-10-27).** JAXA
replaced the platform on 2026-10-01; the old system is planned to shut down
on 2026-11-04 after a month of parallel operation. In practice the old SFTP
host reset every connection from 2026-10-26 (the website was down for a
JAXA-wide maintenance from 2026-10-22 and returned on 2026-10-27). The new
addresses are `https://www.gportal.jaxa.jp/`, SFTP `sftp.gportal.jaxa.jp`
on **port 22** (an AWS Transfer Family endpoint in ap-northeast-1), and
CSW `https://csw.gportal.jaxa.jp/csw`. SFTP is **public-key only**: the
site generates the key pair and the private key (OpenSSH format) is
downloaded under "User account services"; SFTP becomes available up to an
hour after the account is set up. Users are asked to "reconfigure your
user account on the new system" under new terms of use, so the account
may need to be re-registered. The web bulk-download function is gone;
JAXA supplies CSW + SFTP sample scripts instead. The GCOM-C request
procedure for data older than 2.5 years does not apply to GCOM-W/AMSR2.
`experiments/gfds_rebuild/10_signal_from_jaxa_l1b.py` reads
`GPORTAL_HOST`/`GPORTAL_PORT`/`GPORTAL_KEY` from `.env` (defaults: the new
host, 22, key auth).

| product | path under `/standard` | dates | size | delay |
|---|---|---|---|---|
| AMSR2 L1B / L1R swaths (HDF5) | `GCOM-W/GCOM-W.AMSR2/L1B/2/`, `…/L1R/2/` | 2012 → now | ~1.0 / ~1.5 GB per day | still flowing |
| AMSR2 L3 Tb 36 GHz, 0.1°, daily per pass (HDF5) | `GCOM-W/GCOM-W.AMSR2/L3.TB36GHz_10/2/` | 2012 → now | ~33 MB per day for the global grids; **~165 GB whole record** | ~1 day |
| AMSR3 L1R swaths (NetCDF) | `GGW/GGW.AMSR3/L1.L1R/01A/` | 2025-08 → now | ~1.2 GB per day | **~2 h** (newest file 14:44 UTC on 2026-10-15) |
| AMSR3 L3 Tb 36.42 GHz (`TL7`, NetCDF) | `GGW/GGW.AMSR3/L3.TL7/01A/` | 2025-08 → now | ~28 MB per day for the global grids | ~1 day |

- **Near real time:** the `/nrt/` folders for AMSR2 and AMSR3 exist, but they
  return *permission denied* for a standard registered account. The standard
  AMSR3 swaths already arrive within about 2 hours, so this may not matter.
- **History before AMSR2:** G-Portal also has `AQUA` (AMSR-E, 2002–2011) and
  `TRMM` folders. Their contents are not checked yet.
- **Credentials:** the account is personal. A scheduled job needs team-held
  credentials in Databricks secrets. Whether JAXA allows that is not checked.
  The `dsci` Earthdata account's 401 error (D-37) is the cautionary example.
- **Terms:** G-Portal data are free to use, commercial use included, provided
  derived products credit the source. The terms warn that bulk scraping can
  lead to account suspension, so downloads must be throttled.

### Other sensors

| sensor | status (2026-10-15) | access |
|---|---|---|
| **GMI** (GPM) | Operating; V08 and NetCDF4 since 2026-03-04. Covers 65°S–65°N. | NASA PPS or GES DISC `GPM_1CGMI`; Earthdata login |
| AMSR2 intercalibrated L1C | GES DISC `GPM_1CGCOMW1AMSR2` V08 | Earthdata |
| SSMIS (DMSP) | Program end planned for September 2026 | Not a foundation |
| WSF-M, FY-3 MWRI | Not assessed | — |

AMSR2's satellite (GCOM-W) is past its design life. WMO OSCAR lists its end
of life as 2026. So we should plan on AMSR3 going forward and use AMSR2 for
history and for cross-calibration.

### Measured: sensors agree in level

On 2025-08-15 we compared JRC's own per-sensor files, using land pixels only
(`experiments/gfds_rebuild/01_sensor_coverage.py`).

| land pixels valid | AMSR2 | GMI | `ALL` (merge) |
|---|---|---|---|
| north of 65°N | 100% | 48% | 100% |
| 65°S–65°N | 91.5% | 79.8% | 98.4% |
| south of 65°S | 96% | 5% | 96% |

Where both sensors observed the same land pixel (n = 1.15 M):
- median difference GMI − AMSR2 = +0.0001;
- IQR −0.0054 to +0.0062;
- r = 0.75.

Flooded cells in Nigeria 2022 sat 0.08–0.12 below the baseline (D-07), so the
ratio is close to sensor-independent in level. That is what makes switching
sensors survivable. This is one day and one sensor pair; the AMSR2→AMSR3 test
is gate 1c.

## 4. Options

### A. JRC reference set, plus optional full archive

The objective is narrow:
- check that our implementation reproduces JRC's (gate 1);
- reproduce past results that were computed on JRC's signal (Nigeria, fusion).

It is not a dependency.

| tier | what | size (from listings, ±10%) | needed? |
|---|---|---|---|
| reference set | Sample months of `AMSR2/` and `GPM/` `SignalTiffs`, `SourceTiffs` (T_M), `CalibrationTiffs` (T_C) and `PositionTiffs` (P). AMSR2 Calibration/Position exist only for 2013 to 2018-09. Plus `ALL/AveragesAndSd`, `oceanmask.tif`, `SpecialTiffs` and the tech note. | a few GB | **yes** |
| insurance | `ALL/SignalTiffs`, 1997-12 → 2026-09-14 | ~216 GB, ~2–3 days at the measured ~100 GB/day | optional |
| not needed | full per-sensor folders (AMSR2 ~263 GB, GPM ~191 GB), `SINGLE` (~201 GB), magnitude and 4-day folders | — | no; we rebuild these from source |

The JRC licence is unresolved:
- The assessment expected CC BY.
- The fusion inventory found only an "as is" disclaimer.
- The tech note says "Reproduction is authorised provided the source is
  acknowledged."

Building from source avoids the question for anything we publish.

### B. Rebuild the signal from JAXA: AMSR2 history, AMSR3 forward

There are two routes. Start with the cheap one and escalate only if it fails
gate 1.

1. **Grid route (L3, 0.1°).**
   - Per pass, compute C as the 95th percentile of the 9×9 *grid* window,
     then s = M/C. The day is the mean of the passes.
   - The whole AMSR2 record is ~165 GB.
   - It approximates JRC's swath-geometry computation. Gate 1 measures how
     closely.
2. **Swath route (L1R).** Faithful to the tech note: the percentile is taken
   in swath geometry, then gridded to 0.09°. It is ~1.2–1.5 GB per day, so
   AMSR2 history is on the order of 7 TB.

Either way the output goes on the 4000×2000 0.09° grid, so fusion takes it
unchanged (D-03).

**Losses against JRC.** We lose multi-sensor sampling. On 2025-08-15 the
merge lifted land coverage in 65°S–65°N from 91.5% to 98.4%. Single-sensor
days have swath gaps: use the 4-day mean or accept NoData.

### C. Add GMI

- **Gains:** more samples between 65°S and 65°N, and a fallback sensor.
- **Costs:** a second agency, login and format, plus inter-sensor footprint
  differences.
- **When:** only after B passes, and only if the coverage gain matters.

### D. Rely on others

- **DFO/Colorado:** ask Kettner whether they run or plan an AMSR3/GMI
  continuation of GFDS. The DFO station pages we found end in October 2025.
- **AER:** ask how FloodScan handles the AMSR2→AMSR3 transition. We found no
  public statement. FloodScan's current inputs are AMSR2 and GMI, from AER's
  Algorithm Description Document
  ([floodscan-algorithm-notes.md](floodscan-algorithm-notes.md)).
- **Our FloodScan outside Africa** stops at 2026-03-31 only because of an
  ingest gap. AER 90-day zips and the AER API exist; fusion's `final_report.md`
  §7 covers this.

## 5. Feasibility of an open FloodScan-like product

**1. Can we build it? Yes.**

| component | status |
|---|---|
| Daily signal | Method specified; source data reachable (§3); prototype under way (§6 gate 1) |
| Flooded fraction | R1 physics route exists, but crude: one global K, JRC's 2009 baseline as dry reference, fixed 0.30 threshold, no filtering |
| History and baselines | AMSR2 from 2012 at JAXA. AMSR-E and TRMM folders unchecked. JRC's archive is a shortcut. |
| Validation | Fusion's gold harness: CEMS/UNOSAT maps, frozen train/holdout split, scorers. FloodScan SFED as benchmark. |
| 90 m downscaling | Optional. The POC showed it is mostly the static prior. |

**2. Can it match FloodScan? Partly shown.**

Scores are on held-out gold, with each product scored on the cells it
observed (`results/phase5_skill_overall.csv`, `mode = conditional`). The
GFDS column is the crude R1 rule on JRC's signal.

| | GFDS (crude) | FloodScan |
|---|---|---|
| recall | **0.35** | 0.32 |
| false-alarm ratio | 0.62 | 0.47 |
| CSI | 0.220 | 0.251 |
| CSI, vegetated land (no snow) | 0.079 | 0.186 |

- **Recall is level, and the whole gap is false alarms.** That is where AER's
  documented fixes live: a dynamic dry reference, an MDFF-style minimum,
  temporal filtering and masks.
- **Vegetated land is the clearest weakness.** This is the single global K
  failing.
- **By continent** (`mode = all`): both products are equal but poor in Africa
  and South America (~0.10–0.12). The gaps are in Oceania (0.22 vs 0.35) and
  North America (0.08 vs 0.15). Neither can see Europe's small floods at
  10 km.
- **At admin level,** Nigeria 2022 LGA series reach r ≈ 0.9, and the R1
  fraction matched SFED with r = 0.84 and 99% of peak area. That is one
  event.
- **All of this used JRC's signal.** A rebuilt signal has not been through the
  retrieval yet.

**3. Can we keep it running? This is the deciding risk.**
- It needs two agencies' format changes absorbed, and missing days recorded
  and alerted on.
- That is a standing commitment, roughly 0.1–0.2 FTE.
- GFDS ended exactly here.

**Openness.**
- **FloodScan is benchmark only.** The KB `floodscan-ingest` page says sharing
  derived products requires AER approval. R0, which is fitted on SFED,
  therefore cannot ship.
- **JAXA (with credit) and NASA inputs are open.**
- **JRC's archive licence is unresolved,** which is another reason to build
  from source.

**Fusion, briefly.** Where FloodScan exists, GFDS added 0.004 CSI to fusion
(6,656 hits for 17,246 false alarms). Its fusion value is mostly Europe. Our
own product is the main case.

## 6. Plan, gates and effort

The thresholds are proposals; agree them before running each gate. Effort is
measured against this repo's record: fusion v1's build ran 2026-10-12 to
10-14. The pacing items are downloads, run-in time and review, not code.

| gate | test | AI work | calendar |
|---|---|---|---|
| **1a. Signal, grid route** | Rebuild AMSR2 s from L3 and test it against JRC's AMSR2 files: pixelwise r ≥ 0.95 and median \|diff\| < 0.005. **Done 2026-10-15** (§7). The pixel replication bar **fails** (r 0.69). Against FloodScan, the flood-tracking test is on par with JRC `ALL` and better than JRC AMSR2-only. Proposal: redefine this gate as flood-tracking and repeat it on a second event. | hours | ~1 day (3.6 GB took ~2 h 45 min) |
| 1b. Swath route | Only if 1a fails: the same test from L1R | ~1 day | ~1 week (~45 GB per month) |
| 1c. AMSR3 | AMSR3 vs AMSR2 over the overlap (2025-08 →): level step < 0.005 | hours | ~1 day |
| **2. Flooded fraction** | Per-pixel K, dynamic dry reference, minimum fraction, masks. Fitted on `train` events only. Held-out goals: false-alarm ratio ≤ 0.50 at recall ≥ 0.32, and vegetated CSI ≥ 0.15. | ~1–2 days | ~1 week, plus your review |
| 3. History | 2012→ AMSR2 (plus AMSR-E if it is reachable). No level step at sensor changes. Return periods. | ~1 day | days (~165 GB of L3) |
| 4. Operations | Daily AMSR3 job with absence records and alerts | ~1 day | weeks of run-in |

Alongside:
- ask JRC (cause of the stop), DFO (any continuation) and AER (AMSR3);
- record the choice as an ADR once made.

## 7. Prototype result: gate 1a (2026-10-15)

**Setup**
- Input: AMSR2 L3 36 GHz H-pol, 0.1°, daily per pass, from G-Portal,
  2022-08-15 → 2022-11-30 (216 files, 3.6 GB).
- Signal: the tech-note method on the grid (`04_signal_from_l3.py`).
- Reference: JRC's AMSR2-only signal for the same 108 days, none missing
  upstream.
- Scripts: `experiments/gfds_rebuild/02`–`06`.

**A. Against JRC, October 2022, land** (`05_compare_jrc.py`):

| measure | result |
|---|---|
| level | identical: median difference −0.00002 |
| land coverage | 95.0% ours vs 94.9% JRC |
| flood contrast (inland pixel-days) | sd ratio 0.971; orthogonal slope 0.961 |
| pixel-day r | 0.69; 0.85 at 0.45°; 0.90 at 0.9° |
| per-pixel temporal r (median, same land pixels) | 0.29 daily; 0.41 for 4-day means |
| Niger–Benue confluence, daily AOI mean | r 0.95 |

All of these numbers come from `05_compare_jrc.py 2022-10-01 2022-10-31`.

- The pre-set bar (pixelwise r ≥ 0.95, median |diff| < 0.005) **fails**.
- The two processing chains carry independent day-to-day noise. They differ
  in time sampling (JRC averages per orbit; the L3 grid is a day mean) and in
  window geometry.
- An early reading that our grid version "flattens" wet pixels was a
  regression-to-the-mean artefact. Binned on the mean of both signals, the
  wet classes differ by at most +0.0034.

**B. Against FloodScan SFED, Nigeria 2022** (`06_nigeria_lga_vs_floodscan.py`):
- Window: 15 Aug → 30 Nov, the book chapter 05 window.
- Each source is scored the same way: the book's anomaly against JRC's
  static baseline, a 4-day trailing mean, LGA means, and zero-lag r against
  SFED.

| SFED seasonal max | LGAs | JRC `ALL` (AMSR2+GPM) | JRC AMSR2 only | **ours (AMSR2)** |
|---|---|---|---|---|
| 0.02–0.05 | 202 | 0.30 | 0.29 | 0.34 |
| 0.05–0.10 | 118 | 0.45 | 0.36 | 0.46 |
| 0.10–0.20 | 58 | 0.88 | 0.86 | 0.90 |
| > 0.20 | 28 | 0.93 | 0.90 | 0.93 |

- **Reference LGAs** (ours vs JRC `ALL` / JRC AMSR2):
  - Ours ≥ JRC `ALL` in all seven; Ajaokuta is a tie (0.862 each).
  - Ours ≥ JRC AMSR2 in six of seven. At Ibaji, JRC AMSR2 scores 0.979
    against our 0.970.
  - Examples: Kogi 0.95 vs 0.92/0.91; Lokoja 0.93 vs 0.90/0.89; Ekeremor
    (coastal) 0.58 vs 0.25/0.46.
- **Flooded pixels** (SFED max > 0.2, n = 853), median r: ours 0.70, JRC
  `ALL` 0.71, JRC AMSR2 0.60.
- Per-LGA results: `results/gfds_rebuild/nga2022_lga_r_by_source.csv`.

**Reading**
- **Against JRC's two-sensor product: indistinguishable.** The band
  differences are 0.00–0.04 on 28–202 LGAs, which is noise.
- **Against JRC's own AMSR2-only signal: consistently better.** Ours leads by
  0.03–0.10 in every band, at 6 of 7 reference LGAs, and at pixel level
  (0.70 vs 0.60).
- What we need from a rebuild is flood tracking. Replicating JRC's
  pixel-day values is not the goal.
- **Part of the lead may be smoothness, not more information.**
  - A 0.1° cell is ~23% larger than a 0.09° cell, and the L3 day mean
    smooths further.
  - A smoother signal correlates better with a smooth target (SFED works at
    ~22 km), especially after LGA averaging.
  - The pixel-level result is the least affected by this, which makes it the
    strongest single line.
  - The untested alternative is that the day mean averages every overlapping
    observation and so damps noise.

**Caveats**
- **One event, one basin, one season.** Repeat on Pakistan 2022 before
  relying on this.
- **FloodScan is not independent truth:** it shares the AMSR2 radiometer.
  This is fair for comparing the three sources against each other, not as
  absolute skill.
- All three anomalies used JRC's 2009 baseline.
- **The flood-tracking criterion was chosen after the pre-set replication bar
  failed.** Treat it as a proposal to redefine gate 1a, not as a passed
  pre-registered test.

**Operational, measured**
- G-Portal delivers ~0.3–0.4 MB/s on one session.
- With two concurrent sessions on one account, one session was dropped.
- One session stalled without closing. The fetcher now sets a 120 s read
  timeout and reconnects at most 3 times per file.

## Open questions

- Swath route: needed only if a later gate shows the grid route loses skill.
  On Nigeria 2022 it did not (§7).
- Can a team or service account hold G-Portal credentials for a scheduled job?
- Do the `AQUA` (AMSR-E) and `TRMM` folders give a usable pre-2012 record?
- Does a GPM-constellation intercalibrated AMSR3 L1C exist?
- Virtual-gauge API: empty because the feed stopped, or broken?
- What caused the JRC stops? Ask JRC.

## Sources

- JRC, *Global Flood Detection System: Data Product Specifications*, v2015:
  <https://www.gdacs.org/flooddetection/Download/Technical_Note_GFDS_Data_Products_v1.pdf>
- GDACS GFDS data server and listings: <https://www.gdacs.org/flooddetection/DATA/>
- JAXA G-Portal: <https://gportal.jaxa.jp/gpr/>; terms of use:
  <https://gportal.jaxa.jp/gpr/index/eula>; AMSR3 data provision guideline:
  <https://gportal.jaxa.jp/gpr/assets/mng_upload/GGW/AMSR3_data_prov_guideline_en.pdf>
- JAXA AMSR brightness temperature catalogue: <https://www.eorc.jaxa.jp/AMSR/datacatalog/tb/>
- NOAA OSPO, GPM V08 / NetCDF4 change:
  <https://ospo.noaa.gov/data/messages/2026/02/MSG_20260220_2002.html>
- NOAA OSPO, AMSR2 to AMSR3 transition: <https://www.ospo.noaa.gov/products/suites/gaasp/maps.html>,
  <https://ospo.noaa.gov/data/messages/2026/07/MSG_20260728_1045.html>
- WMO OSCAR, GCOM-W: <https://space.oscar.wmo.int/satellites/view/gcom_w>
- NASA GES DISC, `GPM_1CGCOMW1AMSR2` V08: <https://www.earthdata.nasa.gov/data/catalog/ges-disc-gpm-1cgcomw1amsr2-08>
- NSIDC, SSMIS through September 2026:
  <https://nsidc.org/data/user-resources/data-announcements/user-notice-ssmis-processing-will-now-continue-through-september-2026>
- DFO River and Reservoir Watch: <https://floodobservatory.colorado.edu/>

## Exactness test (2026-10-26): can GFDS be replicated pixel for pixel?

Scripts: `experiments/gfds_rebuild/07_signal_from_swath.py` (raw orbit
footprints → JRC's 0.09° grid), `08_gridding_schemes.py` (which footprints
go in a cell and a day), `09_window_convention.py` (is C taken on the grid).
JRC's archive was fully served on 2026-10-26, including the per-sensor
"answer key" files and the baseline rasters; the per-sensor calibration and
position files are populated for 2013–2016 (`*_HD5_GZ.tif`) and empty in
2018.

**Magnitude: exact.** With JRC's own `AMSR2/AveragesAndSd/bt_signal_avg.tif`
and `bt_signal_sd_sf_min_to_0.5.tif`, m = (avg − s)/sd (note the sign: a
flood *lowers* the signal) reproduces JRC's MagTiff on 2018-08-20 for 99.9%
of land cells to within the file's 0.001 rounding; the rest is the ±20 clip.
So anyone holding their baseline rasters and a signal file can produce
their magnitude product exactly. The rasters are ~24 MB each and should be
backed up.

**Signal: close, not exact, from NOAA's AMSR2 data.** One day each in 2016
and 2018, West Africa box, land cells:

| | 2016-08-20 | 2018-08-20 |
|---|---|---|
| our T_M vs JRC SourceTiff, UTC-day mean | r 0.77, bias −1.6 K, MAD 1.8 K | r 0.75, bias −1.0 K, MAD 1.5 K |
| … daytime pass only, else night pass | r 0.84, bias −0.3 K, MAD 1.1 K | r 0.80, bias −0.4 K, MAD 1.2 K |
| our C vs JRC CalibrationTiff | r 0.90, bias −1.6 K, MAD 0.65 K | (empty file) |
| our s vs JRC SignalTiff | r 0.74, bias 0.0000, MAD 0.0033 | r 0.56, bias +0.001, MAD 0.0042 |
| JRC's own SourceTiff / CalibrationTiff vs its SignalTiff | r 0.90 | – |

What the answer keys settle:
- C is **not** a 9×9 window of grid cells: recomputing it from JRC's own
  source grid matches their calibration grid in only 19% of cells under any
  percentile convention. It is taken among raw footprints, as the tech
  note says and as our code does. The position file (values 1–81) is a
  footprint-window offset, not decodable from the grid.
- JRC's source grid follows the **daytime (ascending) pass** where it
  exists (95% of cells with both passes in 2016, 87% in 2018) and is filled
  from the night pass elsewhere; a per-cell maximum over the day's samples
  fits about as well. A plain UTC-day mean of both passes is 1–2.5 K too
  cold.
- JRC's signal is not the ratio of its own source and calibration grids
  (r 0.90), so the signal is a per-footprint ratio averaged separately.
- The residual after the pass rule, ~1.1 K MAD at r 0.8–0.84, is the size
  of the calibration difference between NOAA's CSU record (tied to GMI) and
  JAXA's L1B, which JRC read. The test for that is one day from JAXA's L1B
  via G-Portal (~300 MB for the orbits over the box).

**Bottom line.** Exact magnitude, yes, from their baseline. Exact signal
needs JAXA's calibration and the pass rule above; it is plausible but
unproven until the L1B test runs. From NOAA data alone the signal is
r 0.8 pixel-for-pixel, unbiased.
