# Phase 0 inventory: what can be reached

Checked 2026-10-12 from this machine with the team's credentials. Each row was verified by reading the data or the official documentation; sources are listed under the table. "Reachable" means that the v1 pipeline actually reads it.

## Flood inputs

| product | version used | resolution | revisit | years reachable | access route (v1) | licence | known gaps / caveats |
|---|---|---|---|---|---|---|---|
| **Copernicus GFM** (Sentinel-1 SAR), `ensemble_flood_extent`, `exclusion_mask`, `advisory_flags` | GFM product v3.x (PDD); PUM `gfm4.0_pum_2025` (version stamps inside disagree: D 6.1 V20251021 vs D5.2 V20251201). STAC has no version field. | 20 m Equi7 (read at the 80 m overview, D-08) | 6–12 days per orbit; S1B lost Dec 2021; S1C from 2025 | 2015-01-01 → present | EODC STAC `https://stac.eodc.eu/api/v1`, collection `GFM`, COG range reads, no auth | CEMS terms: "freely and fully open … as long as the source (CEMS) is acknowledged" (GFM wiki). STAC `license` field says `proprietary`. | Exclusion mask covers HAND ≥ 10 m, radar shadow, look-alikes, and dense vegetation/urban (for non-flood pixels only). Snow/frozen ground is **not** excluded, only flagged in `advisory_flags` (1/3), which v1 treats as NoData (D-17). STAC returns items outside the query bbox; each item's bbox is re-checked. |
| **FloodScan** SFED (AER) | v05r01 (from file names; AER's user guide could not be retrieved) | 300″ (~9 km) fraction | daily | Africa 1998-01-12 → 2026-10-10 (current). Asia-Australia, N. America, S. America 1998-01-12 → **2026-03-31** | Africa: prod `raster/floodscan/daily/v5/processed/aer_area_300s_v{date}_v05r01.tif`. Other regions: **dev** `raster/floodscan/daily/v5/raw/aer_sfed_area_300s_19980112_20260331_v05r01_{aa,na,sa}_no_ndt.nc` via ocha-stratus | Commercial. "Sharing derived products requires AER approval" (KB `floodscan-ingest.md`). | **Europe is not covered.** The KB says the prod `processed/global/` folder holds the global merge; it exists but is empty. The regional NetCDFs are on the **dev** account, not prod. `_no_ndt` is not explained anywhere. The 90-day rolling zips after 2026-03-31 are not used in v1. The files have no NaN and no mask: FloodScan is "observed" everywhere it covers. |
| **GFDS** (JRC/GDACS passive microwave) `signal` | "Global Flood Detection System – Version 2"; method Kugler & De Groeve (JRC, EUR 23303) | 0.09° (4000×2000) | daily | 1997-12 → **2026-09-14** | `https://www.gdacs.org/flooddetection/DATA/ALL/SignalTiffs/YYYY/MM/signal_YYYYMMDD_ALL.tif`, range reads, ≤ 1 req/s; static baseline `AveragesAndSd/bt_signal_avg.tif` | No reuse licence found, only an "as is" disclaimer (gdacs.org terms). | **The feed appears to have stopped.** Every day from 2026-09-15 on returns HTTP 403, and 2026-09-07/08 are also missing. JRC confirmed the system is end-of-life with no SLA (`docs/gdacs-gfds-assessment.md`). The server returns 403, not 404, for missing days. There are no snow/frozen masks, and snow/arid artefacts are documented. |
| **VIIRS flood map** (NOAA/GMU) | v2r0, 1-day global composite (`VIIRS-Flood-1day-GLBnnn_v2r0_blend_*`) | 375 m (0.00337°), 136 AOI tiles of 15° | daily (cloud-limited) | 2012–2020 and 2023-08 → present; **nothing for 2021–2022 or early 2023** | `s3://noaa-jpss/JPSS_Blended_Products/VFM_1day_GLB/TIF/` (public, no credentials), range reads | NOAA Open Data Dissemination: "open to the public and can be used as desired", attribution requested | The GMU archive is offline. Cloud, shadow and snow/ice codes become NoData. It enters as the optical source because MODIS cannot be reached (D-16). |
| **MODIS NRT flood** (MCDWD) | — | 250 m | daily | `MCDWD_L3_NRT`: 2026 → present; reprocessed `MCDWD_L3`: 2003–2025 | LAADS DAAC; downloads need a **NASA Earthdata login** | CC0 (NASA data policy) | **Not reachable in v1.** No Earthdata login or token is configured on this machine, and the KB has no mechanism for one. LAADS `MCDWD_L3_NRT/` lists only 2025 (days 346–365) and 2026. The Phase 5 ablation states the cost (D-16). |

## Gold labels

| product | version used | resolution | years | access route | licence | known gaps |
|---|---|---|---|---|---|---|
| **Copernicus EMS rapid mapping** flood delineations | CEMS gold v1 (ds-geospatial-impact-estimates, ADR-0036's v2 not built) | vector | 2012 → 2026 | dev `global/copernicus_ems/flood/gold/` (`label_index.parquet`, `labels/code=*/data.parquet`) | CEMS On-Demand Mapping terms: free access for reproduction, distribution and adaptation, with the source credited | 283 codes and 2,715 label sets, 71% in Europe. Geometries are 3-D with line fragments (v1 forces them to 2-D polygons). There is no `sensor_class`; v1 derives SAR/optical from the `sensor` string, and 771 sets have no sensor. |
| **UNOSAT** flood maps | UNOSAT gold v2 | vector | 2013 → 2026 | dev `global/unosat/gold/` | Varies per HDX dataset: mostly CC BY-SA, some CC BY-NC-SA 3.0, CC BY-IGO | 180 codes and 867 sets. 221 sets have no valid mask. Six sets have impossible dates (1476/1900). Five giant events were never built. |

After the v1 filters, 2,742 label sets in 353 events remain usable. 187 of those events are used in v1 (`results/event_split.csv`). Every exclusion and its reason is recorded in `results/events_excluded.csv` and `results/label_sets_excluded_summary.csv`.

## Static and supporting layers

| layer | version | resolution | access | licence | notes |
|---|---|---|---|---|---|
| HAND | ASF GLO-30 HAND v1 (2021 DEM release) | 30 m, 1° tiles | `https://glo-30-hand.s3.amazonaws.com/v1/2021/` | CC0 (DEM © DLR/Airbus) | 26,450 tiles; a missing tile means ocean. Overviews 2/4/8. |
| Permanent water | JRC Global Surface Water **v1.4** occurrence (2021) | 30 m, 10° tiles | `https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/` | "free of charge, without restriction of use" (Copernicus regulation), source EC JRC/Google | v1.5 (1984–2024) exists, and JRC states that v1.2–1.4 occurrence has consistency errors. v1 uses v1.4 (open issue). 504 tiles, no overviews. |
| Land cover | ESA WorldCover 2021 **v200** | 10 m, 3° tiles | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/` | CC BY 4.0 | 2,651 tiles. Class 10 = tree cover, 50 = built-up. |
| Snow season | **ERA5-Land** monthly means `snow_cover` (%), 2014–2023, regridded by CDS to 0.25° | 0.25° | CDS API `reanalysis-era5-land-monthly-means`, `~/.cdsapirc` | CC BY 4.0 | **Stands in for the MODIS snow climatology (MOD10CM), which needs an Earthdata login (D-05).** The documentation contradicts itself on units ("%" vs a 0–1 fraction); the values are 0–100 (D-05). |
| Basins | HydroBASINS v1c level 4 | vector | `https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_*_lev04_v1c.zip` | HydroSHEDS licence: free for non-commercial and commercial use | Used only to group events for the split. |
| Continents | Natural Earth 1:50m admin-0 | vector | naciscdn.org | public domain | Continent of each event centroid (nearest country offshore). |
| GloFAS | v4.0 forecast; v4/v5 historical | 0.05° | EWDS `cems-glofas-historical` / `cems-glofas-forecast` | CEMS-FLOODS licence | **Not used in v1.** Prior v2 is not built because access plus the build would exceed the 5% budget (D-21). |

## What the gaps cost

- **No MODIS.** The optical slot is VIIRS only, and VIIRS has no data for 2021-01 → 2023-07. For 46 of the 187 v1 events (36 train, 10 held out), every gold day falls in that gap, and 59 event windows overlap it. For those, fusion runs on two sources.
- **No FloodScan in Europe**, and none outside Africa after 2026-03-31. In Europe the microwave vote is GFDS alone.
- **GFDS has ended (2026-09-14).** The daily operational job has no microwave input outside Africa from that date. The job records GFDS absence per day; it does not invent zeros.
- **Phase 5 ablation** (`results/phase5_ablation.csv`) shows what dropping each available source costs. A source that is absent (MODIS) cannot be ablated; its absence is a limitation, stated in the final report.

## Sources read

KB (`ds-knowledge-base`): `infrastructure/storage.md`, `conventions.md`, `databricks.md`, `pipeline-registry.md`, `libs/ocha-stratus.md`, `pipelines/floodscan-ingest.md`, `raster-pipelines.md`, `cems-flood-archive.md`, `flood-gfm.md`, `datasets/README.md`.

Official pages: GFM PUM (`extwiki.eodc.eu/gfm_assets/gfm4.0_pum_2025.pdf`), GFM PDD output layers, GFM terms (`extwiki.eodc.eu/GFM/Terms_and_conditions`), CEMS mapping terms and citation guidelines, HDX licences FAQ, GDACS terms of use and the GFDS 2015 data specification, `floodscan.aer.com`, NASA `MCDWD_LAADS_Desc.pdf` and the data-use policy, the SSEC VIIRS flood archive page, the AWS NOAA JPSS registry, the GLO-30 HAND readme, the WorldCover PUM v2.0, the GSW download page, the CDS ERA5-Land catalogue entry, the HydroBASINS technical document, the EWDS CEMS-FLOODS licence.

Probes, all run 2026-10-12:
- STAC search and COG headers.
- GDACS HEAD requests across 2026-09-01 … 2026-10-10.
- LAADS archive JSON listing.
- Blob listings of `raster/floodscan` on both accounts.
- Bucket listings for HAND, WorldCover, GSW and `noaa-jpss`.
