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
