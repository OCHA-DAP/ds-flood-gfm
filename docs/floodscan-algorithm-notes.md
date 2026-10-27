# FloodScan's algorithm: notes from AER's published documents

*2026-10-16. Purpose: an internal reproduction of FloodScan, to check that an
open FloodScan-like product is on the right track. These are notes and
citations; AER's documents are © AER and are not copied into this repo.*

## Sources

AER publishes three documents. The FloodScan support page links to them
(<https://published.aer.com/floodscan/aer_floodscan_support.html>), on a mirror
whose TLS certificate does not match its hostname. `www.aer.com` now redirects
to Janus Research. We read archived copies:

| document | pages | archived copy |
|---|---|---|
| ARC Flood Extent Depiction **Algorithm Description Document** (ADD), AFED V05R00, Galantowicz, Picton & Root, 31 Aug 2018 | 67 | <https://web.archive.org/web/2024id_/https://www.aer.com/contentassets/22663ebfdb7c467599f363466b32770f/afm_algo_descr_doc_v05r00_r00.pdf> |
| AFED **Algorithm Performance Document**, V05R00 | 201 | <https://web.archive.org/web/2024id_/https://www.aer.com/contentassets/22663ebfdb7c467599f363466b32770f/afm_algo_perf_doc_v05r00_r00.pdf> |
| FloodScan **Data Users Guide**, V05R01 (© 2023) | 14 | <https://web.archive.org/web/2024id_/https://www.aer.com/siteassets/files/floodscan_data_users_guide_v05r01_r03.pdf> |

The ADD describes the Africa (ARC) system, V05R00. Our FloodScan files are
V05R01, which covers Africa, South America, North America south of 55°N and
Asia-Australia. Regional static tables beyond Africa are not documented.

## What FloodScan computes

Section numbers below refer to the ADD.

1. **Inputs (§2.3, §3.1)**
   - Current sensors: AMSR2 (JAXA L1B/L1R) and GMI (NASA 1B).
   - Historical sensors: TMI (1998–2014), AMSR-E (2002–2011) and SSM/I
     (1992–1997 only).
   - The repo's earlier "AMSR2/SSMIS" description is wrong; book chapter 05's
     AMSR2 + GMI is right.
   - Channels: **37 GHz V and H** drive the flooded fraction. 19 GHz V/H,
     22/24 GHz V and 85/89 GHz V/H are used for flags and false-positive
     detection.
2. **Footprint matching (§3.3)**
   - A Backus–Gilbert-type linear resampling (eq. 1) to a 22 km
     modified-Gaussian composite footprint for 37 GHz (AMSR2, GMI).
   - A 27 km Gaussian composite for the flag channels.
   - The coefficient sets are not published.
3. **Geolocation correction (§3.4)**
   - A per-swath shift (distance, azimuth) that minimises a cost between Q
     and persistent-water fraction at coastal control points (eq. 4).
   - Below 50 control points, coastal samples are dropped.
4. **Regridding (§3.5)**
   - 2-D cubic spline from swath coordinates onto a **5-arcminute (300″)
     grid**.
   - One grid per pass direction per day. Where swaths overlap, the earliest
     one wins.
5. **Atmospheric correction (§3.6)**
   - Earth-leaving Tb = (Tb − upwelling term) / transmittance, from AER's
     radiative transfer model (Moncet et al. 2011) driven by NCEP or GFS.
   - Surface temperature is deliberately not corrected: the polarization
     ratio cancels it.
6. **Rain and frozen flags (§3.7)**
   - The Ferraro et al. (1996) decision tree, with documented modifications:
     a local scattering-index threshold SI0 = 10 + 2·max(ΔSI61, 0) +
     max(SI61 − 7, −1), revised snow, desert and arid tests.
   - Then dilation over 7–9 cells, with a re-test at SI > SI0 − 1.
7. **Flooded fraction (§3.8)**
   - PRI = (V − H)/(V + H) at 37 GHz (eq. 9).
   - **Q = √(PRI + 0.11) − 0.58·PRI** (eq. 10).
   - Linear mix of persistent water, dry land and flood (eq. 11). With
     Q_flood = Q_pow:
     **f_flood = (Q_obs − Q_pow·f_pow − Q_dry·(1 − f_pow)) / (Q_flood − Q_dry)** (eq. 13).
   - **Q_dry** (eqs. 14–17) is a weighted blend of:
     - a dry-land climatology Q_dc(day, x), weight 1/var(year-to-year
       61-day median);
     - the expected 61-day time-centred median, weight 1/0.0012. It is
       predicted from 31- and 61-day retrospective medians with linear
       time-shift models, in point-wise and spatial (11×11) variants.
     - In seasonal wetlands, Q_dry is the climatology alone.
   - **Flash false-positive detection** (§3.8.2):
     - newly flooded means 19 GHz FF > 0.05 while the previous 37 GHz FF
       was < 0.05;
     - flagged where more than 75% of a 10-pixel circle is newly flooded;
     - screened by t_flash = 1.5 + 4.75·f_hop, the seasonal-wetland mask, a
       noise threshold and an isolation test;
     - then dilated over contiguous areas with FF > 0.025.
   - **Final FF** (§3.8.3):
     - weighted mean over today and the two previous days, with weights
       0.67 − 0.33·f_hop, 0.33 and 0.33·f_hop; outliers get half weight;
     - point-wise and spatial variants merge after 12 of 15 days above
       MDFF (eq. 18);
     - gaps are filled by decaying the last value;
     - FF is clipped to [0, 1 − f_pow];
     - FF below **MDFF** is set to 0, with a coastal exception.
8. **Downscaling (§3.9, §4.21)**
   - FF is interpolated to a 3″ grid. A 3″ cell is flooded where
     FF ≥ ff0(cell).
   - The result is upscaled, the residual computed, and the step repeated
     3–6 times until the residual converges.
   - ff0 is the footprint-weighted share of cells whose relative
     floodability (RF) is at least the cell's own, over about 4,000 km²
     (AER computed it on a GPU).
   - **RF** combines:
     - MERIT elevation above the nearest channel along the flow path;
     - stream networks at > 5,000 km² and > 100 km² accumulation, weighted
       0.75 and 0.25;
     - a 150-point flow-wise median filter and a 5×5 minimum filter;
     - a 21×21 local linear fit;
     - an override by GSW annual recurrence wherever it is above 0.

**Static tables, all built from history (§4)**
- Built from AMSR-E 2003–2010 and AMSR2 2013–2016:
  - Q 31- and 61-day median climatologies;
  - year-to-year variance;
  - time-shift parameters;
  - a Q lookup by NDVI and tree cover (75th percentile in bins of 0.05);
  - seasonal-wetland detection (seven rule groups);
  - the dry-land climatology Q_dc (eq. 25 with stated weights);
  - f_hop, from the kurtosis of FF time derivatives:
    f_hop = (log10 k − 0.75)/0.75;
  - **MDFF = 3σ-equivalent FF**: the average of MDFF0 and its annual
    maximum, capped at 0.25 where f_pow < 0.49.
- Other inputs:
  - land-water mask from Hansen datamask OR SRTM water bodies, with
    documented exceptions;
  - Hansen 2000 tree cover;
  - VIP Lab 2000–2010 NDVI;
  - MERIT DEM, flow direction and accumulation;
  - GSW recurrence.

**Published constants**
- Q transform: a = 0.11, b = −0.58.
- Q_pow (AMSR-E): 37 GHz 0.457 ascending, 0.455 descending; 19 GHz 0.465
  ascending, 0.463 descending.
- AMSR2→AMSR-E cross-calibration, Q' = A·Q + B, at 37 GHz: 1.0466/−0.0154
  ascending, 1.0438/−0.0141 descending.
- GMI→AMSR-E at 37 GHz: 1.0835/−0.0276.
- Weight on the expected median in Q_dry: 1/0.0012.

## Products (Users Guide V05R01)

- **SFED**: standard extent, binary at 3″.
- **MFED**: maximum daily extent; skips the false-positive filter and the
  multi-day averaging.
- **NDT-SFED / NDT-MFED**: "no detection threshold", meaning no MDFF.
  - Our blob's `*_no_ndt.nc` files are therefore the standard, thresholded
    product.
  - The fusion inventory listed `_no_ndt` as unexplained; this answers it.
- **AREA**: "0–1 fraction of unmasked area with flooding aggregated at
  specified spatial resolution".
  - The team's `aer_sfed_area_300s` is the SFED share of unmasked (non-water)
    3″ cells in each 300″ cell.
  - It sits on the same 5′ grid as the flooded fraction.

## What this means for reproducing it

- **FloodScan's signal is not GFDS's.**
  - FloodScan uses the 37 GHz *polarization ratio* (V − H)/(V + H) at about
    22 km.
  - GFDS uses the 36.5 GHz *H-pol* ratio of measurement to calibration pixel
    (M/C).
  - Our rebuilt GFDS signal is not FloodScan's input.
- **Downscaling conserves FF.**
  - AER's Performance Document (§4): "the effective flooded fraction of the
    AFED product itself closely matches the input microwave flooded
    fractions".
  - So at aggregates of ≥ 22 km, SFED_area ≈ FF / (1 − f_pow). The
    microwave core can be tested against SFED_area at 0.25° and admin level
    without building the downscaling.

**How much of it we can rebuild:**

| exactly specified | we must approximate | not specified |
|---|---|---|
| Q transform; mixing eq. 13; Q_pow; AMSR2 and GMI cross-calibration; Q_dry blend; MDFF rule; 3-day weights; f_hop; flash thresholds; RF recipe | 22 km footprint matching (no coefficients: Gaussian-smooth instead); atmospheric correction (AER's model: ERA5 + a simple model, or skip at first, since PRI cancels most of it); static climatologies (different years and inputs); earliest swath vs the L3 day mean; NDVI source | the seven seasonal-wetland rule groups; the outlier test in 3-day averaging; V05R01 regional tables outside Africa |

**Expectation:** close is realistic; bit-exact is not. The largest residuals
should appear in seasonal floodplains and wetlands (the unspecified rules),
along coasts, and in arid rainy seasons, where MDFF is high.

## Step 0 result (2026-10-16): the core retrieval, Nigeria 2022

**Code**
- `experiments/floodscan_repro/01_core_l3.py`: the retrieval.
- `experiments/floodscan_repro/02_compare_floodscan.py`: scoring.
- Per-LGA table: `results/floodscan_repro/nga2022_core_l3_lga.csv`.

**Setup**
- Input: AMSR2 L3 36 GHz V and H, 2022-08-15 → 11-30.
- From AER, as documented:
  - 22D smoothing, then PRI → Q;
  - AMSR2 → AMSR-E cross-calibration;
  - eq. 13 with Q_pow;
  - 2-day weighting.
- **Placeholders**, so this is not yet a reproduction:
  - Q_dry is the in-window 10th percentile of Q per pixel and pass;
  - f_pow = 0;
  - no atmospheric correction;
  - no rain, frozen or flash flags;
  - no per-pixel MDFF.

**Scored against FloodScan SFED area (300″)**, each product on its own grid.
LGA bands are the same LGAs as the GFDS test (`results/gfds_rebuild/`).

| SFED seasonal max (LGA) | LGAs | FloodScan core r | GFDS-style r | ours / SFED at the LGA's SFED peak |
|---|---|---|---|---|
| 0.02–0.05 | 202 | 0.19 | 0.34 | 0.68 |
| 0.05–0.10 | 118 | 0.39 | 0.46 | 0.54 |
| 0.10–0.20 | 58 | 0.86 | 0.90 | 0.54 |
| > 0.20 | 28 | 0.91 | 0.93 | 0.38 |

- **Timing is right where floods are large.**
  - Median temporal r is 0.93 in 0.5° blocks with SFED max > 0.2.
  - Reference LGAs score 0.89–0.94; the exceptions are Ajaokuta (0.73) and
    coastal Ekeremor (0.28).
  - This is slightly below the GFDS-style M/C route.
- **Magnitude and location are wrong.**
  - Pooled over 0.5° block-days, r = 0.17.
  - Our total flooded fraction is 3.4× SFED's, but only 0.4–0.7× SFED's at
    flood peaks.
  - 63% of our mass sits in blocks SFED never floods (max < 0.02).
  - The south (below 8°N: forest and coast) holds 52% of our mass against
    27% of SFED's.
  - November, as floods recede, has our highest mean (0.049 vs SFED 0.007).
    That is seasonal dry-land drift, the job of AER's day-of-year Q_dc
    climatology built with NDVI.
- A uniform MDFF of 0.06 lowers the excess (2.3×) but removes the small
  floods. AER's MDFF is per pixel and per day of year, from 3σ noise.

**Reading.** The published core physics tracks real floods over time. What it
gets wrong is the dry reference and noise, the components that need history:
- the Q_dc climatology (needs NDVI and tree cover);
- the expected 61-day median model;
- a per-pixel MDFF;
- persistent water f_pow;
- the rain and flash flags.

That is step 1.

## Step 1 result (2026-10-21): AER's history tables, West Africa 2024

**Code**
- `experiments/floodscan_repro/00`, `03`–`06`.
- Per-LGA table: `results/floodscan_repro/nga2024_lga_by_variant.csv`.

**Inputs**
- **AMSR2:** NOAA's CSU microwave CDR on AWS, daily 0.25° grids, 36 and
  18 GHz V/H, June–December of 2023, 2024 and 2025.
  - 638 days; 50 MB kept on disk.
  - 2024-12-17 to 12-20 are absent upstream.
  - CSU calibrates to GMI, not to JAXA.
- **Box:** West Africa, 0–20°E, 0–20°N.
- **Earth Engine layers:** NDVI climatology (MODIS 2013–2022), tree cover
  (MODIS VCF), persistent water (MOD44W 2011–2015), ocean (LSIB).
  Footprint-weighted at 27 km.
- **Training years** for the static tables: 2023 and 2025. **Test year:**
  2024, Aug–Nov.
- **Reference:** FloodScan SFED, from the fusion work's local cache.

**Departures from AER**, beyond step 0's:
- **Q_pow is the median Q over open sea.** The box has no inland water the
  radiometer sees as water. Lake Chad, which MOD44W calls 90% water, shows
  dry-land Q, so the inland fit extrapolated below dry land.
- **Cells with f_pow > 0.5 are excluded** (essentially Lake Chad).
- **var61** is the 9×9 spatial median, with a 1e-6 floor; 61–72% of cells sit
  at the floor. Dry-land Q61 really is stable from 2023 to 2025.
- **Wetlands** use our heuristic (Q61c − QdLUT > 0.02), not AER's unpublished
  rules.
- **Time-shift model:** a61 is fixed at 1.
- **No rain or frozen flags**, because the stage-1 data has no 23 or 89 GHz.
- **Flash detection is simplified**, with no t_flash/f_hop screen.
- **No atmospheric correction**, as in step 0.

**Scores against SFED, 2024-08-01 → 11-30** (ours as ff / (1 − f_pow)):

| | placeholder (in-window Q_dry) | full chain, no flash | full chain |
|---|---|---|---|
| total vs SFED | 11.9× | 1.29× | 1.34× |
| share of ours where SFED never floods | 0.81 | 0.40 | 0.47 |
| pooled r, 0.25° cell-days | 0.27 | 0.51 | 0.47 |
| temporal r, cells with SFED max > 0.2 | 0.85 | 0.71 | 0.56 |
| temporal r, 0.5° blocks with SFED max > 0.2 | 0.93 | 0.81 | 0.79 |
| LGA r, SFED max > 0.2 (8 LGAs) | 0.90 | 0.80 | 0.80 |
| ours / SFED at LGA peaks (SFED max > 0.2) | 0.78 | 0.56 | 0.54 |

**Reading**
- **The history tables fix magnitude and location:** the total falls from
  11.9× to 1.3× SFED, and pooled r roughly doubles.
- **They cost temporal skill where floods are large.**
  - Our simplified flash filter is net harmful: the full chain is worse than
    no-flash on every measure.
  - MDFF (~0.10 in mid-July) is probably inflated by rain noise, since nothing
    is rain-flagged. That would zero moderate flooding, and it fits peaks at
    ~0.55× SFED.
- **November is still too high** (0.0076 vs SFED 0.0040): seasonal drift that
  two training years do not fully capture.

### Step 1 diagnostics (2026-10-21): what each part costs

**Setup**
- All variants are scored on one common population: 659,328 cell-days valid
  in every variant.
- Code: `05_ff_chain.py` (variants) and `06_score_2024.py`.
- Tables: `results/floodscan_repro/nga2024_variant_summary*.csv`.

**Column key**
- ratio: total ours / total SFED.
- never: share of our total in cells SFED never floods.
- r_pool: pooled r over cell-days.
- r_cell: median per-cell temporal r, 86 cells with SFED max > 0.2.
- r_blk: the same for 16 blocks at 0.5°.
- LGA r: for SFED max bands 0.10–0.20 (43 LGAs) and > 0.20 (8 LGAs).
- peak: ours / SFED at LGA peaks.

| variant | ratio | never | r_pool | r_cell | r_blk | LGA r 0.10–0.20 / > 0.20 | peak |
|---|---|---|---|---|---|---|---|
| climatology Q_dry only (no MDFF, no flash) | 2.81 | 0.58 | 0.48 | 0.70 | 0.81 | 0.81 / 0.84 | 0.57 |
| + MDFF ×0.5 | 1.91 | 0.50 | 0.48 | 0.71 | 0.80 | 0.77 / 0.83 | 0.56 |
| + MDFF capped at 0.06 | 1.83 | 0.53 | 0.47 | 0.67 | 0.79 | 0.71 / 0.81 | 0.52 |
| + MDFF (AER rule) | 1.36 | 0.47 | 0.48 | 0.66 | 0.77 | 0.64 / 0.80 | 0.38 |
| + MDFF + simplified flash filter | 1.34 | 0.47 | 0.47 | 0.58 | 0.75 | 0.55 / 0.80 | 0.38 |
| var61 floored at 0.005² | ≈ as without the floor | | | | | | |
| placeholder (in-window minimum) | 14.5 | 0.84 | 0.25 | 0.84 | 0.90 | 0.84 / 0.90 | 0.86 |

**Inland cells only** (no ocean in the footprint; 86% of cells):

| variant | ratio | never | r_pool | r_cell |
|---|---|---|---|---|
| + MDFF (AER rule) | 0.83 | 0.15 | 0.68 | 0.66 |
| + MDFF ×0.5 | 1.24 | 0.25 | 0.69 | 0.71 |
| climatology only | 2.11 | 0.45 | 0.67 | 0.70 |
| placeholder | 13.8 | 0.83 | 0.27 | 0.84 |

**Reading**
- **The coast is the main residual: 63% of the false mass sits in the 14% of
  cells with ocean in the footprint.**
  - This is the gap between the 0.25° grid and the footprint-weighted f_pow,
    plus geolocation error.
  - AER addresses it with its geolocation correction and coastal MDFF logic.
    FloodScan itself says floods "within 5–10 km of coastlines" are usually
    not depicted.
- **Inland, the chain matches FloodScan in size** (0.83–1.24×) **and spatial
  pattern** (pooled r ≈ 0.68). Its false mass there is 15–25%.
- **MDFF as built costs timing and peaks:** LGA r 0.81 → 0.64, peak
  0.57 → 0.38. Halving it recovers most of that. This fits MDFF being
  inflated by unflagged rain noise, which is the case for the 23/89 GHz rain
  flags.
- **The simplified flash filter is net harmful.** Leave it off until it is
  implemented properly.
- **Peaks stay at ~0.55× SFED even without MDFF.** The likely causes are the
  0.25° grid (wider than AER's 22 km composite, so narrow floodplains are
  diluted) and the missing atmospheric correction. Neither is tested.
- **The placeholder's temporal scores are not a fair benchmark.** Its
  in-window-minimum reference makes anything above the driest moment read as
  flood. That is also why it is 14× too large with 84% false mass.
- **Small n:** the top LGA band has 8 LGAs and the 0.5° test 16 blocks. Rely
  on r_pool and r_cell.

**AER's own skill statement** (Performance Document §4):
- Best when flooding covers a substantial part of more than one 22 km
  footprint.
- Smallest detectable flood ≈ MDFF × the area of a circle 1.25 footprints
  across. That is about 36 km² at MDFF 0.06 and 22 km.
- False positives are most common in marginally vegetated areas in rainy
  seasons (Sahel, southern Africa, Horn).
- Floods shorter than 2–3 days are mostly missed.
- AER validated with MODIS case studies; the document gives no pooled
  POD/FAR. Fusion v1 measured FloodScan at a false-alarm ratio of 0.47
  against CEMS/UNOSAT gold (branch `feat/flood-fusion-v1`).

## Step 2 (2026-10-22): the rest of the specified chain, before results

Pre-checks that changed the plan (all in the session transcript, numbers from
`06` and scratch scripts):
- **The spatial windows were never rescaled.** AER counts 9×9, 7×7 and 11×11 in
  5′ cells; the same counts had been applied on the 0.25° grid, so each window
  covered three times AER's ground distance. Rescaled to 3, 3 and 5 cells, the
  near-coast test added (§4.20), and the 3×3 mean dropped (one cell here).
  Rescoring moved the LGA peak ratio 0.38 → 0.48 and little else.
- **MDFF is not a two-year sampling artefact:** tables from 2023 alone or 2025
  alone give the same Aug–Nov median (0.075–0.081) as both years together.
- **The peak deficit is half timing, half height:** max(ours)/max(SFED) is 0.75
  against 0.57 at FloodScan's peak day, with our peak a median 3 days late.
  The shortfall is regional (Benue valley 0.5–0.6, Niger lowlands 0.97, Sahel
  1.3–1.7), so it is a dry-reference and rain problem, not one gain factor.
- **Q_pow:** ours 0.416 vs AER's 0.451 on the AMSR2 scale. The gap is in the
  direction that makes our floods larger, so it does not explain low peaks;
  it is the atmospheric correction AER applies and we do not.
- **The NOAA CSU bucket** holds AMSR2 daily grids only for 2023–2026 and no GMI
  at all; AMSR2 swath files exist from July 2012 (~74 MB per orbit). More
  training years therefore mean the swath route, not more grid years.

What was built (`rainflags.py`, `aer_chain.py`, `04`, `05`):
- **Rain flags (§3.7):** Ferraro's land scattering index
  SI = 451.9 − 0.44·T19V − 1.775·T22V + 0.00575·T22V² − T85V with AER's four
  changes (local SI0 = 10 + 2·max(ΔSI61, 0) + max(SI61 − 7, −1); snow test with
  "and" and 264 K; desert > 22 K; two-part arid test), dilation by 2 cells
  (AER: 7 at 5′) and re-test at SI0 − 1. SI0 from the training years' 15th
  percentile of the 61-day SI median. No frozen flags at 0–20°N. Channels:
  AMSR2 18.7 V/H, 23.8 V, 89 V (A-scan), added to the CSU fetch (≈ 6 GB
  streamed, < 0.1 GB kept).
- **Spatial Q61Esp (§3.8.1):** median of (Q61Epw − Q61c) over the 5×5 window
  minus its 3×3 centre, over cells where Q61Epw > Q61c; three initial FF
  forms (point-wise, 61-day, spatial).
- **f_hop (§4.19):** in-sample initial FF per pass, 15-day running mean
  removed, negatives to 0, derivative between successive *valid* observations
  (a pass revisits a cell every ~2 days here, so consecutive-day differences
  discard most of the record), Pearson kurtosis per day-of-year over ±30 days
  across years and passes; f_hop = clip((log10 k − 0.75)/0.75, 0, 1); near
  coast 5·f_pow. Cells with < 8 samples get 0 and are counted in the log.
- **Finishing (§3.8.3):** weights 0.67 − 0.33·f_hop / 0.33 / 0.33·f_hop;
  outliers at half weight; the last earlier instantaneous value at half the
  prior-day weight when the prior day is empty; the eq. 18 merge with the
  15-day record (trigger ≥ 12 of 15 above MDFF, region where ff_sp and ff_61
  exceed MDFF, w_sp = clip((N − 11)/3, 0, 1)); fill-in decay; clip; MDFF with
  the coastal exception.
- **Flash filter (§3.8.2):** now with the t_flash = 1.5 + 4.75·f_hop screen
  and dilation across rain-flagged cells whose last test was flash or rain.

Our readings where the ADD is silent or its text is garbled (marked in code):
- Outlier test: more than 0.10 from the group median.
- Fill-in: f0 = f−1 − a·max(T0 − T−1, 0)·(f−1 − f_floor). The extracted
  formula divides by (f−1 − f_floor); the stated behaviour ("the larger the
  difference … the faster f0 decreases") needs a product.
- Coastal exception: min(FF, half the lowest neighbour above MDFF).
- Running medians need ≥ 25% of the window valid (AER states no minimum;
  AMSR2 sees ~55% of the box's cells per pass per day, and 50% left many
  cells without a 61-day median).

### Swath-route prototype (2026-10-22): one day, `08_swath_prototype.py`

- CSU AMSR2 orbit files for 2024-09-01 (ICDR; 2023 and earlier are FCDR):
  15 listed, 5 cross the box. Per-variable range reads: **46 MB streamed**
  for 36 GHz V/H with geolocation and time. All channels for the chain would
  be ~2.5× that, so ≈ 25 GB per June–December season.
- Footprints split by spacecraft latitude trend, Gaussian footprint matching
  (FWHM 22 km, 3σ radius) onto the 5′ grid, earliest swath wins.
- Check: the 5′ Q aggregated 3×3 against the CSU 0.25° grid's Q, same day
  and pass: r 0.991 (asc, n 3324) and 0.990 (dsc, n 4813), bias ≤ 0.0003,
  median absolute difference 0.0008–0.0009. Coverage 0.55 / 0.94 of cells
  (asc / dsc) against the grid's 0.52 / 0.75.
- So the swath route reproduces the grid where both exist, at 9× finer
  cells, with no account needed. What it costs is bandwidth (above) and the
  chain's run time at 240×240 cells.

### Atmospheric correction (2026-10-22): built, pending results (`atmos.py`, `09_gee_merra2.py`)

- AER (§3.6) corrects two terms, transmittance and upwelling emission, from
  an RT model on NCEP/GFS profiles; it deliberately leaves reflected
  downwelling and surface temperature alone. Same two terms here from the RSS
  AMSR model (Wentz & Meissner 2000, ATBD v2, eqs. 22–29, Table 4): oxygen
  absorption A_O = aO1 + aO2·(T_D − 270), vapour A_V = aV1·V + aV2·V²,
  τ = exp(−secθ·(A_O + A_V)), effective temperatures T_D(V, T_S) and
  T_U = T_D + b6 + b7·V; TB_EL = (TB − (1 − τ)·T_U)/τ at θ = 55°.
- Inputs: MERRA-2 hourly TQV, T2M, PS from Earth Engine
  (`NASA/GSFC/MERRA/slv/2`), sampled at the 00 UTC (descending) and 12 UTC
  (ascending) hours on the 0.25° box. No new account. Over land T2M stands in
  for the sea-surface temperature in the small b5 term (our choice).
- Why it should matter: at 36.5 GHz a 50 mm atmosphere has τ 0.79 and
  TB_up 59 K, and compresses a land PRI of 0.071 to 0.056; a 15 mm Sahel
  atmosphere compresses it to 0.063. The flood signal is the PRI rise, and
  the flood season is the humid season, so uncorrected Tb damp the peaks in
  exactly the Benue-valley months. October TQV over the box is ~45 mm,
  December ~18 mm, so the dry reference drifts with humidity as well.
- Switch: `04 default <out> atmos` builds corrected tables;
  `05` with `{"atmos": true}` runs on them. The rain-flag channels stay
  top-of-atmosphere, as in AER.

## Step 2 results (2026-10-23): rain flags, finishing step, atmospheric correction

Setup: tables from 2023 + 2025 with rain-flagged passes removed; test Aug–Nov
2024; 677,401 common cell-days; `results/floodscan_repro/nga2024_variant_summary*.csv`
(baseline) and `results/floodscan_repro/atmos/` (earth-leaving Tb).
Columns as before, plus peak_mm = max(ours)/max(SFED) per LGA regardless of
day, and lag = our peak day minus FloodScan's (median over LGAs with
SFED max ≥ 0.10).

| variant | ratio | never | r_pool | r_cell | r_blk | LGA r 0.10–0.20 / > 0.20 | peak | peak_mm | lag |
|---|---|---|---|---|---|---|---|---|---|
| step 1 (rescaled windows, no flash) | 1.36 | 0.46 | 0.48 | 0.65 | 0.76 | 0.67 / 0.83 | 0.48 | – | – |
| **step 2 full** (rain flags, f_hop, merge, fill-in, coastal rule, flash) | 1.41 | 0.41 | 0.50 | 0.70 | 0.82 | 0.73 / 0.85 | 0.51 | 0.68 | 2 |
| step 2, no daily rain flags | 1.32 | 0.40 | 0.51 | 0.68 | 0.78 | 0.68 / 0.80 | 0.46 | 0.66 | 5 |
| step 2, no flash filter | 1.45 | 0.41 | 0.51 | 0.72 | 0.82 | 0.73 / 0.84 | 0.51 | 0.68 | 2 |
| step 2, no spatial merge | 1.41 | 0.41 | 0.50 | 0.70 | 0.82 | 0.73 / 0.85 | 0.51 | 0.68 | 2 |
| step 2, f_hop = 0 | 1.42 | 0.41 | 0.50 | 0.69 | 0.82 | 0.75 / 0.85 | 0.49 | 0.72 | 3 |
| step 2, no coastal exception | 1.35 | 0.39 | 0.51 | 0.70 | 0.82 | 0.73 / 0.85 | 0.51 | 0.67 | 2 |
| step 2, MDFF × 0.5 | 1.84 | 0.43 | 0.51 | 0.74 | 0.85 | 0.80 / 0.85 | 0.53 | 0.72 | 3 |
| step 2, no MDFF | 2.48 | 0.51 | 0.51 | 0.75 | 0.86 | 0.82 / 0.86 | 0.58 | 0.72 | 2 |
| **step 2 + atmospheric correction**, full | 1.47 | 0.42 | 0.49 | 0.71 | 0.85 | 0.74 / 0.86 | 0.49 | 0.68 | 2 |
| + atmos, no MDFF | 2.46 | 0.51 | 0.49 | 0.75 | 0.87 | 0.85 / 0.87 | 0.59 | 0.74 | 1 |

Inland cells (86%): step 2 full ratio 0.92, never 0.12, r_pool 0.70,
r_cell 0.70; with the atmospheric correction 0.95 / 0.13 / 0.70 / 0.71.

**The four pre-registered checks**
1. *Rain flags vs the noise floor.* Aug–Nov inland median MDFF (36 GHz asc)
   0.078 → 0.064, p75 0.125 → 0.116, p90 0.174 → 0.169. Rain explains part
   of the median and almost none of the upper tail. The high-MDFF cells are
   where the thresholding loses floods, so the tail is what matters.
2. *Spatial merge.* It fires (11,389 triggered cell-days, 9,703 raised) but
   moves no score: ff_sp sits barely above ff_pw here.
3. *Rain flags on their own population.* Same as on the common population
   (the flags remove passes, and the 3-day averaging refills nearly every
   cell-day). Daily rain flags are worth +0.05 in LGA r, +0.05 in peak and
   3 days of lag.
4. *Peak height and lag.* 0.51 at FloodScan's day, 0.68 regardless of day,
   2 days late. Unchanged by the atmospheric correction.

**Reading**
- Steps 2–3 together: cell r 0.65 → 0.70, block r 0.76 → 0.82, LGA r
  0.67/0.83 → 0.73/0.85, inland false mass 0.15 → 0.12, lag 3 → 2. Real,
  modest. Rain flags are most of it; f_hop, the merge, the fill-in and the
  coastal exception are each within ±0.02 of neutral. The flash filter now
  costs 0.02 in cell r and nothing else.
- **The atmospheric correction is neutral on scores** (±0.02 everywhere)
  while moving Q_pow to 0.439 (AER 0.451). It scales Q_obs and Q_dry
  together, so the flooded fraction barely changes. Kept for fidelity, not
  for skill; it is not the cause of the low peaks.
- **The peak deficit is regional and sits in narrow floodplains:** by state,
  max/max is 1.0 in Anambra (wide Niger lowland) and 1.3–1.5 in the Sahel,
  but 0.6 in Kogi, 0.5 in Taraba and 0.3–0.4 in Adamawa, where the Benue
  valley is a few km wide; Adamawa's peak is a month late. A 28 km cell
  dilutes such a floodplain below MDFF until the flood is broad. Removing
  MDFF lifts peak_mm only to 0.72–0.74, so part of the deficit is in the
  retrieval itself at this cell size.
- **What is left on the 0.25° grid:** the coast (all-cells never 0.41 vs
  inland 0.12) and the MDFF tail, both tied to the grid. The next structural
  step is the swath route (5′ cells, fixed 22 km footprints), tested first on
  two weeks for its effect on the 7-day Q noise (`10_swath_noise_test.py`).

### Parameter sweep (2026-10-23): `07_fit_params.py`, corrected tables, fit year 2024

One knob at a time around AER's values (`results/floodscan_repro/fit_sweep_atmos.csv`):

| knob | effect |
|---|---|
| weight on the expected median (W_Q61E, AER 1/0.0012), ×¼ … ×4 | nothing moves (all scores within 0.005) |
| Q_pow offset −0.02 | peak 0.49 → 0.74, peak_mm 0.68 → 0.89, LGA r 0.74/0.86 → 0.78/0.86, but ratio 1.47 → 2.17, inland ratio 0.95 → 1.42 |
| Q_pow offset +0.02 | ratio 1.03, inland 0.68, peak 0.35 |
| MDFF × 0.25 | LGA r 0.84/0.87, peak_mm 0.74, but inland never 0.13 → 0.27 and ratio 1.5 |
| MDFF × 1.5 | r_cell 0.60, peak 0.37 |
| outlier test off (deviation 1.0) | r_cell 0.71 → 0.73, LGA r 0.75/0.87, peak 0.53; our 0.10 test costs ~0.02 |

Reading: no knob raises the peaks without inflating the total, because the
inland total is already right (0.95×) while the peaks are 0.68×. The error is
in the *distribution* of flooded fraction, too much low-level mass and too
little at the peaks, which is what cell dilution plus a per-cell threshold
produces. Tuning on this grid cannot fix it. Our outlier test is slightly
harmful and AER's is unspecified; left at 0.10 pending a hold-out year.

### Gridding-noise test (2026-10-25): `10_swath_noise_test.py`, 1–14 Sep 2024

Does the 0.25° daily grid add day-to-day Q noise that fixed 22 km footprints
on a 5′ grid do not? MDFF is 3σ of the 7-day Q variance (§4.20), so this
decides whether the swath route lowers the threshold tail.

Same inland land cells, same days, 36 GHz, two 7-day windows
(239 MB streamed for 14 days; 8 days had been cached before a DNS outage):

| | asc | dsc |
|---|---|---|
| cells × weeks compared | 10,686 | 10,745 |
| σ7 at 0.25°, CSU daily grid | 0.0012 | 0.0011 |
| σ7 at 0.25°, our 5′ grid aggregated 3×3 | 0.0008 | 0.0007 |
| ratio of median 7-day variances, grid / ours | 2.09 | 2.33 |
| ratio at the 75th percentile | 1.83 | 1.72 |
| σ7 at 5′ itself | 0.0008 | 0.0008 |
| same-day Q at 0.25°, r / bias | 0.974 / −0.0001 | 0.973 / +0.0002 |

**Reading**
- The daily grid carries about twice the 7-day Q variance of footprint
  matching, for the same cells and days, with no level bias. The daily grid
  averages a different mix of footprints into each cell every day; fixed
  Gaussian footprints on a fixed grid do not. This is the noise MDFF is
  built from.
- MDFF scales with σ, so the swath route should cut it by ~30% (1/√2.1)
  across the distribution, tail included, which the rain flags did not do.
  Our Aug–Nov median 0.064 would land near 0.045; AER's worked example is
  0.06 at 22 km.
- Even at the native 5′ cells, σ7 (0.0008) is below the 0.25° grid's. So the
  swath route gives 9× finer cells *and* a lower noise floor. It addresses
  the three residuals at once: the MDFF tail, narrow-floodplain dilution,
  and (with AER's geolocation correction later) the coast.
- Cost: ~25 GB streamed per June–December season for 36/18/23/89 GHz, three
  seasons for tables plus test, < 1 GB kept; chain runtime at 240×240 cells
  about 9× the current few minutes.
- To fix before the full run: day assignment. Descending passes over the
  box come from orbits starting ~23:20 UTC the previous day, so grouping by
  file date leaves some days with 2–20% descending coverage (seen on 7, 9,
  12, 14 Sep). AER starts its day at 21:00 UTC the day before (§3.5).

## Hold-out result (2026-10-27): test year 2023, tables from 2024+2025

Every score above tested 2024 with tables from 2023 and 2025. Swapping roles
(`04 2024,2025 …static_tables_2425.npz`, `05 … {"year": 2023}`, `06`)
answers whether the method or the year was doing the work.

| | 2024 test | 2023 hold-out |
|---|---|---|
| ratio (our total / FloodScan total, box) | 1.41 | 4.79 |
| never (share of our flooded cell-days where FloodScan never floods) | 0.41 | 0.79 |
| r_cell | 0.70 | 0.39 |
| r_lga1 | 0.73 | 0.43 |
| peak / peak_mm | 0.51 / 0.68 | 0.08 / 0.47 |
| our mean ff, Aug–Nov land | 0.0066 | 0.0122 |
| FloodScan: inland cells with SFED max > 0.2 | 100 | 41 |

2023 was a low-flood year in FloodScan (half of 2024's LGA peaks) and our
chain produced twice as much flooding as in 2024. Diagnostics
(scratchpad `q_level_by_year.py`, `sahel_tables_diag.py`):

- Not calibration: monthly land medians of Q and of the 18 GHz Tb agree
  across 2023 (FCDR), 2024 (ICDR) and 2025 (FCDR) to 0.001 / 1 K.
- Where: the Sahel band 12.5–17.5°N (rows 10–29; mean of cell max 0.12 in
  2023 against 0.02–0.05 in 2024), in October–November (mean ff per 10-day
  block rises to 0.022 from 0.007 in August). Mostly Niger and Chad, outside
  the LGA scores. South of 12.5°N the two years behave alike.
- Why: in that band Q has a soil/vegetation seasonal cycle (band median
  0.362 in June, 0.348 in late August, 0.365 in November). The dry-down of
  2023 reached 0.364–0.367 in Oct–Nov; 2024 and 2025 reached 0.356–0.362.
  The 2024+2025 Qdc/Q61c at DOY 288–319 is 0.355–0.360, so 2023 sits
  ~0.005 above it, which eq. 13 turns into ff ≈ 0.09 with Q_pow − Q_dry ≈
  0.06. The band's MDFF is 0.10–0.11, so the excess straddles the
  threshold. South of 12.5°N the year-to-year spread is < 0.001.
- The 2024 result is partly luck: 2024's Sahel dry-down sat inside the
  2023+2025 range, so no false floods there. The southern (Nigeria) scores
  do not depend on this and stand.

**Expectation weight test** (`wexp_test.sh`; `w_exp` is the eq. 17 weight
of the in-season 61-day expectation; the table weight 1/var61 is ~1e6 with
two training years, so at AER's 1/0.0012 the climatology has 99.9% of the
weight):

| w_exp | 2023 ratio / never / r_cell / r_lga1 | 2024 ratio / never / r_cell / r_lga1 / peak |
|---|---|---|
| 833 (AER) | 4.79 / 0.79 / 0.39 / 0.43 | 1.41 / 0.41 / 0.70 / 0.73 / 0.51 |
| 1e5 | 3.34 / 0.72 / 0.38 / 0.44 | 1.34 / 0.42 / 0.68 / 0.75 / 0.46 |
| 1e6 | 1.99 / 0.63 / 0.44 / 0.43 | 1.19 / 0.46 / 0.64 / 0.79 / 0.46 |
| 1e7 | 1.74 / 0.61 / 0.41 / 0.46 | 1.11 / 0.50 / 0.59 / 0.76 / 0.42 |

Giving the in-season expectation real weight halves the 2023 false total
but does not restore skill (r_cell stays ~0.4, 61% of our flooding still
where FloodScan has none) and it erodes 2024 (r_cell 0.70 → 0.59), because
a 61-day median also absorbs real slow floods. It is a partial patch, not
the fix.

**Reading.** The dry reference built from two years cannot absorb normal
inter-annual variation of the semi-arid surface, which is the size of the
flood threshold there. AER's tables come from 10+ years. The CSU 0.25°
daily grid exists only from 2023, so the grid route is capped at three
or four training years; the CSU swath files run from 2012-07 and are the
only public source of a 13-year AMSR2 climatology. The swath route is
therefore needed for the climatology, not only for the noise floor
(previous section). Until then, scores from a single test year should be
quoted with this caveat.
