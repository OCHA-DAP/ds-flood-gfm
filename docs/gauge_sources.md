# Observed river-gauge sources for the gauge-exceedance test (candidates, not chosen)

Checked 2026-10-13 for plan Phase 5 (`docs/flood_fusion_plan.md`, ADR 0002). Every fact below is from the official page cited. GRDC's data policy and the USGS WaterServices shutdown were re-read directly on 2026-10-13.

**The answer: no single open, global source of observed gauge data with flood thresholds exists.**
- **USGS gauges with NWS flood categories** are the only source found that has official numeric flood thresholds, public-domain terms and sub-daily data across 2015–2026.
- **Everywhere else**, thresholds would have to be derived from the record (for example a 2-year return level). Some sources have thresholds but no archive we could find.
- **Outside the US**, no open bulk source covering 2024–2025 was verified.
- **Modelled discharge is out of scope.** GloFAS, and the Google Flood Forecasting feed the team's Nigeria gauge monitor uses, are model output, not observations.

| source | measured | coverage | flood thresholds | latency | licence | access | refs |
|---|---|---|---|---|---|---|---|
| USGS + NWS NWPS | stage and discharge, typically 15 min | US: 12,165 USGS gauges; ~4,000 NWS forecast points | official action/minor/moderate/major, in stage and flow, per gauge | about 1 h | public domain | USGS OGC API (key optional); NWPS API for thresholds, joined through `usgsId` | [1]–[10], [32] |
| GRDC | daily discharge only | global (station count not verified) | none; derive from the record | not verified | free, identified access; no redistribution; no commercial use | portal request | [11], [12] |
| GRDC-Caravan | daily discharge | 5,356 catchments, 25 countries | none; derive | static release, to 2023 | CC BY 4.0 | Zenodo | [13], [14] |
| Caravan | daily discharge | 6,830+ basins | none; derive | static, 1981–2020 at release | CC BY 4.0 | Zenodo | [15], [16] |
| UK Environment Agency | level (and flow at some stations), 15 min | ~8,000 stations | only a proxy (`typicalRangeHigh`, 5% exceedance) | near real time; API keeps 4 weeks, daily CSV archive | OGL v3 | REST, no registration | [19], [20] |
| Australia BoM Water Data Online | level and discharge | ~3,500 stations (2015 figure) | minor/moderate/major exist per gauge but are not in this service | mostly supplied daily | mostly CC BY, varies by data owner | web and web service | [21], [22], [30] |
| Bangladesh FFWC | level | Bangladesh | Danger Level | daily; the web shows only the last 2 days | not found | web tables; annual PDF reports 2010–2021 | [23], [24] |
| Mekong MRC | level | Mekong | Normal / Alarm / Flooded states | near real time, no archive found | all rights reserved | web | [25] |
| GSIM | monthly, seasonal and yearly indices only | >30,000 stations | none | published 2018 | CC BY-SA 3.0 | PANGAEA | [17], [18] |

**Not usable or not verified:**
- **India (CWC, India-WRIS):** the official sites refused automated access, so nothing was verified. Check in a browser.
- **South Africa:** the water department's site blocked access.
- **EFAS:** collects partner observations under licence; no public release of them was found.
- **WHOS, HydroSOS:** a data broker and a status product, respectively.
- **Google Flood Hub API:** documents gauge metadata and forecasts, but no observations.
- **Not checked:** Brazil (ANA), Canada (HYDAT).

**Watch-outs:**
- The legacy USGS WaterServices APIs are decommissioned in Q1 2027 [6]. Build on the new OGC API.
- GRDC's terms forbid redistribution and commercial use [12]. Ask GRDC before relying on it for an OCHA product.
- Compare NWS flood stage with USGS gage height only after checking the two vertical datums [8].

**Method notes (judgement, not from the sources):**
- A gauge exceedance means "flooding somewhere along the gauged reach", not truth for one 1 km cell. So the test scores events and timing, not cells.
- Compare stage with stage thresholds using the daily maximum, not the mean.
- US floods differ from those in OCHA's priority countries, so a US-only test answers a narrower question.

**Choice pending (user).** The obvious anchor is USGS + NWS, possibly with GRDC-Caravan for breadth to 2023.

## Sources

1. https://www.usgs.gov/mission-areas/water-resources/science/usgs-streamgaging-network
2. https://waterdata.usgs.gov/iv-data-availability-statement/
3. https://api.waterdata.usgs.gov/docs/ogcapi/
4. https://api.waterdata.usgs.gov/docs/ogcapi/keys/
5. https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits
6. https://waterdata.usgs.gov/blog/api-waterservices-decom/
7. https://water.noaa.gov/about/api
8. https://api.water.noaa.gov/nwps/v1/docs/swagger.json
9. https://www.weather.gov/news/241101-national-water-prediction
10. https://www.weather.gov/aprfc/terminology
11. https://grdc.bafg.de/data/data_portal/
12. https://grdc.bafg.de/about/data_policy/
13. https://grdc.bafg.de/data/project_datasets/grdc-caravan/index.html
14. https://zenodo.org/records/10074416/latest
15. https://www.nature.com/articles/s41597-023-01975-w
16. https://zenodo.org/records/15530021
17. https://essd.copernicus.org/articles/10/787/2018/
18. https://doi.pangaea.de/10.1594/PANGAEA.887470
19. https://environment.data.gov.uk/flood-monitoring/doc/reference
20. https://environment.data.gov.uk/hydrology/doc/reference
21. https://www.bom.gov.au/water/about/publications/document/InfoSheet_18.pdf
22. https://www.bom.gov.au/resources/learn-and-explore/flood-knowledge-centre/flood-classifications-and-river-heights
23. http://old.ffwc.gov.bd/ffwc_charts/waterlevelobs.php
24. http://old.ffwc.gov.bd/index.php/reports/annual-flood-reports
25. https://ffw.mrcmekong.org/
30. https://www.bom.gov.au/waterdata/services
32. https://www.weather.gov/disclaimer
