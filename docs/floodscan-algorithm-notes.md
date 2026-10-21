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
