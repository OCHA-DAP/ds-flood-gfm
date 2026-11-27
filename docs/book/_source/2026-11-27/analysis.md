# Flood fusion: analysis and results

Oct 15, 2026 · @Zack Arno

Flood fusion combines GFM (radar), FloodScan and GFDS (microwave) and VIIRS (optical) into one daily probability of flooding on a 0.01° grid. This page sets out how it is tested and what the test found: scope, inputs, how each cell is classified, how scoring works, how events are split for testing, and the results. All 170 usable flood events in Africa, Asia, South America and Oceania are each scored once, by a model fitted only on events in other river basins. Every term is defined in the Glossary tab. `docs/methods.md` in the repo describes the same mechanics with the code that does each step, and docs/log.md is the ledger: every step so far in order, with the command that reproduces it, its output files and its commit.

## Status

Done: all 170 events are built and scored (15 October). Two rules were decided for you on 14 October, before any result existed, because you asked for the work to be finished without further questions: the rule for the days before and after each event, and the event-level rules. Both can be reversed in `configs/fusion.yaml`.

| part | status |
| --- | --- |
| Scope: Africa, Asia, South America, Oceania (ADR 0003) | done |
| 170 events | all built (15 October); 148 have enough gold flood for the event-level scores |
| Three land types: open, vegetated, urban; snow split dropped (ADR 0005) | done |
| Days before and after each event: cells flooded on any map day of their window get no label (ADR 0006) | decided for you on 14 October; the old rule's scores are shown alongside |
| Five folds by river basin (ADR 0004) | done; one basin holds 40% of the gold flood, so the folds are uneven (Testing) |
| Scores: recall, precision, F1; all cells and observed cells; per continent and land type | done (Results) |
| Event level: caught, false events, days to detection (ADR 0007) | decided for you on 14 October; done (Results). False events away from floods deferred |

## The basics: what is classified, and how it is scored

**The unit is one cell on one day.** A cell is a 0.01° square, about 1.1 km across. For every cell on every day, each satellite product says one of three things: **flood**, **dry**, or **couldn't see**. Fusion turns the three products' reports into a probability P, and calls the cell flooded when P ≥ 0.5. A gold map, on the day it was made, labels each cell it examined **flooded** or **dry**; cells it did not examine have **no label**.

**Only labelled cells on map days are scored.** A map day is a day with a gold map. Scoring compares the product's call with the gold label, cell by cell, on those days only:

|  | gold: flooded | gold: dry |
| --- | --- | --- |
| product: flood | hit | false alarm |
| product: dry or couldn't see | miss | correct dry |

**Cell level: counts, summed.** Hits, misses and false alarms are counted over every labelled cell, every map day and every event, then:

- **recall** = hits ÷ (hits + misses): the share of gold-flooded cells the product flagged. The denominator is all 404,014 gold flooded cell-days in the 170 events.
- **precision** = hits ÷ (hits + false alarms): the share of the product's flood calls that gold confirms. The denominator is the product's own flood calls on labelled cells.
- **F1** combines the two. Because counts are summed, a flood of 80,000 cells weighs 800 times a flood of 100.

**Event level: one verdict per event.** An event is **caught** if, on some day from 3 days before its first map to 3 days after its last, the product flags at least 10% of the cells gold flooded (and at least 5). It is a **false event** if the same rule fires on days 30 to 10 before the first map, when no flood was mapped. Both are reported as shares of the 148 events with at least 5 flooded cells. Here every event counts once, whatever its size.

**Which days are used for what:**

| days | what they give |
| --- | --- |
| map days | the gold labels; all recall, precision and F1 |
| 30 days before and 60 after each event | dry labels for fitting each product's false-alarm rate (cells flooded on a map day get no label); the false-event check |
| ordinary days, anywhere else | no gold, so not scored. Only the base flood rate comes from them, from a random sample of places with no mapped event |

**Two rates that come up often, with their denominators:**

- **12% of open-land cells are flooded on map days** = 355,927 gold-flooded ÷ 2,856,112 labelled floodable open-land cell-days on map days (all land types: 404,014 ÷ 4,387,050 = 9.2%). This is the flood rate inside mapped floods, on the days they were mapped.
- **0.2% on ordinary days** = 1,493 ÷ 647,443: in 30 random places with no mapped event, over every day of 2023–2025, the share of open-land cell-days GFM saw on which GFM called flood. There are no gold maps there, so GFM stands in. Fusion uses this as the starting probability for every cell (the base rate).

**Why the gap between them matters.** Precision depends on how common flood is where you score. With the same false-alarm rate, a flood 60 times rarer gives far lower precision. Every precision on this page is from map days inside mapped floods; precision on ordinary days has not been measured yet (Next steps, 6).

## From pixels to cells: how every layer lands on one grid

Nothing is compared at a product's own resolution. Every layer, from 20 m radar to 9 km microwave, is first put onto the same grid of 0.01° **cells** (about 1.1 km). "Pixel" on this page always means a product's own native pixel; "cell" always means our grid square. The rules below are the whole resampling; each lives in `configs/fusion.yaml` under the key shown.

| layer | native pixel | how it reaches a cell | cell is observed when | cell is positive when | config |
| --- | --- | --- | --- | --- | --- |
| GFM flood | 20 m, read at 80 m (every 4th pixel) | each pixel goes to the cell holding its centre; about 190 pixels per cell | at least 50% of the expected pixels were evaluated (not excluded, not flagged) | at least 30% of the observed pixels are flood | `sources.gfm` |
| VIIRS flood map | 375 m | same: pixel centre to cell; about 9 pixels per cell | at least 50% of its pixels observed (not cloud, shadow, snow or no data) | at least 30% of the observed pixels are flood | `sources.viirs` |
| FloodScan SFED | about 9 km (300 arcsec): one value, the flooded fraction | each cell takes the FloodScan pixel that contains the cell's centre; about 81 cells share one pixel | the region's file exists for that day | that pixel's flooded fraction is at least 30% | `sources.floodscan` |
| GFDS | about 9 km | same containing-pixel rule; the signal is turned into a flooded fraction against GFDS's long-term mean | the day's file exists | that fraction is at least 30% | `sources.gfds` |
| CYGNSS watermask (experiment) | 0.01°, but offset by half a cell | each cell is the average of the four CYGNSS pixels at its corners, a quarter each | at least 50% of that area has data | at least 30% of the observed area is water | `experiment_cygnss.py` |
| gold map (flood polygons and examined area) | vector | drawn onto a 0.001° grid: 10 × 10 = 100 sub-pixels per cell | at least 50% of the sub-pixels are inside the examined area (otherwise no label) | at least 30% of the examined sub-pixels are flood; "flood possible" above 30% gives no label | `gold` |
| HAND (floodable) | 30 m, read at every 4th pixel | fraction of samples per cell | always | at least 30% of the cell is under 15 m above the nearest stream | `static.hand` |
| permanent water (JRC occurrence) | 30 m, every 4th pixel | fraction per cell | always | removed from the domain if at least 50% of the cell is water at least 90% of the time | `static.gsw` |
| land type (ESA WorldCover) | 10 m, every 4th pixel | fraction per cell | always | urban if built-up is at least 20%; else vegetated if trees at least 30%; else open | `static.worldcover` |
| normally wet (JRC monthly recurrence) | 30 m: exactly 40 × 40 pixels per cell | fraction per cell and calendar month | always | at least 30% of the pixels were water in that month in at least half the years | `baseline` |

**Why the same 50% and 30% everywhere.** Half observed before a cell counts as seen, and 30% flooded before it counts as flood, are applied to every product and to gold alike, so no product is held to a different standard than the answer key. The cost is that a flood covering less than 30% of a cell is invisible to everyone, gold included.

**What this means for the coarse products.** One FloodScan or GFDS pixel sets about 81 of our cells at once, so their hits and false alarms come in 9 km blocks (visible as squares in the maps on this page). They are scored cell by cell like everything else, which is fair to them where a flood is large and harsh where it is narrow.

**Days.** Every layer is daily except the static ones (HAND, permanent water, land type: fixed) and normally wet (one map per calendar month). A product that has no file for a day gives "couldn't see" on every cell, never dry.

## How the pieces fit together

1. **Inputs.** Three kinds of satellite product give each 0.01° cell, each day, a value: flood, dry or "couldn't see" (Inputs).
2. **Fixed layers.** Elevation, permanent water and land cover decide which cells can flood, and whether each is open, vegetated or urban (Grid, domain and strata).
3. **Gold labels.** Expert flood maps label each examined cell flooded, dry or none on their map days. The 30 days before and 60 days after each event add dry labels (Gold labels; Events, days and how counts add up).
4. **Fitting.** For each source and land type, the events used for fitting give two numbers: how often it says flood where gold says flooded (hit rate), and where gold says dry (false-alarm rate).
5. **Fusion.** Each day, fusion turns the sources' reports into a probability of flooding using those rates. P ≥ 0.5 counts as a flood call (How fusion and scoring work).
6. **Scoring.** Fusion and each product are compared with gold cell by cell, on the map days of events not used for fitting. The counts are summed into recall, precision and F1, and per event into "caught or not" (Testing; From counts to totals; Was the event caught?).

The project plan judges fusion on three criteria:

- **(a) coverage:** a value on more floodable cells each day than any single product;
- **(b) mapping skill:** in each land type, CSI at least that of the best single product;
- **(c) reliability:** its probabilities mean what they say. Of cells given P 0.6, about 60% should be flooded; the slope between predicted and observed must be 0.8–1.2.

## Scope and events

170 flood events are in scope, in Africa, Asia, South America and Oceania. Europe and North America are set aside for now.

An **event** is one activation code in a gold catalog. Each of its gold maps covers an area on one date. Areas within 0.05° of each other are merged into one **window**, and every window covers every day from 30 days before the event's first map to 60 days after its last.

An event is **built** once its data is prepared. For each window and each of its 91 or more days, the gold labels and every source's flood, dry or "couldn't see" value are put on the 0.01° grid. The result is saved as one file per window (an *event cube*). Fitting and scoring read only these files, which is why re-running them is cheap and building new events is not.

| step | events |
| --- | --- |
| In the gold catalogs: Copernicus EMS rapid mapping (283) and UNOSAT (180) | 463 |
| Pass the gold rules: one acquisition day, 2015 or later (GFM starts in 2015), a map of the examined area, a flood layer, not drawn from coarse optical imagery | 353 |
| In Africa, Asia, South America or Oceania | **170** |
| &#32;with at least 5 floodable cells flooded on a map day (used for the event-level scores) | 148 |

| continent | events | gold flooded cell-days |
| --- | --- | --- |
| Asia | 61 | 270,263 (67%) |
| Africa | 80 | 109,025 (27%) |
| Oceania | 13 | 19,553 (5%) |
| South America | 16 | 5,173 (1%) |
| all | 170 | 404,014 |

Gold flooded cell-days count floodable cells flooded on a map day, summed over map days; they are the cells scored for recall. Asia holds two thirds of them, and one river basin in Bangladesh 40% (four UNOSAT floods, 2017–2022), so every result is also shown per continent and with that basin left out. The five largest events hold 47%: Bangladesh 2020, 2017 and 2022, Thailand 2021 and Myanmar (EMSR130). 13 events have no gold flood on a floodable cell, because their maps' flood areas never cover 30% of one; they are scored, but add only dry cells. Five very large UNOSAT events were never built in the gold catalog and are not available.

## Events, days and how counts add up

Every score on this page is a sum of one kind of count: one labelled cell on one day. A window-day, used below, is one window on one day. Each event contributes three kinds of days, and only its map days are scored. The full list of events is in the Events tab.

&#91;image: One event window, every day: labelled cells per day before, during and after the Punjab floods (EMSR838)\]

*How to read it.* One of the five windows of the Punjab floods (EMSR838). Each bar is one day; its height is the number of this window's examined cells with a label that day.

- This is one of the event's 5 windows. The event's maps run from 27 August to 3 September 2025, so every window covers 28 July to 2 November: 98 days.
- This window has 3 of the event's 7 map days: 28 and 31 August and 1 September. Only those days carry flood labels, and only they are scored. The bar on 1 September is shorter (2,265 cells) because that map examined fewer cells; a cell is labelled only where that day's map examined it.
- Event-period days with no map for this window (27, 29 and 30 August, 2 and 3 September) have no bar: no cell is labelled.
- Every other day labels all 4,078 cells this window's maps examined as dry. The hatched part, 1,176 cells, was flooded on at least one of the three map days (1,152 on 28 August alone). Under the adopted rule, those cells get no label on those days.
- The Events tab lists 46,191 cells for EMSR838: every cell in the rectangles of all 5 windows, examined or not.

| kind of day | which days | labels | used for | in the 170 events (floodable cells) |
| --- | --- | --- | --- | --- |
| **before** | the 30 days before the event's first map | every examined cell dry (except cells flooded on a map day in this window) | fitting false-alarm rates (every day); reliability | 11,280 window-days; 63.0 M dry cell-days |
| **map days** | days in the event period on which this window has a map | flooded, dry or none per cell (see Gold labels) | fitting hit and false-alarm rates; **every recall, precision and F1** | 779 window-days; 404,014 flooded and 3,983,036 dry cell-days |
| **between maps** | days in the event period without a map for this window | none | not used |  |
| **after** | the 60 days after the event's last map | as before | as before | 22,560 window-days; 126.0 M dry cell-days |

The event period runs from the first to the last map across all the event's windows. Each window is scored only on its own map days. The 170 events have 385 windows and 1,023 gold maps on 587 map days. 9 of the windows have no labelled cell at all, because their maps examined no floodable cell; the day counts above are for the other 376. Most events have one or two map days (55 and 35 events); the most is 19. A window covers at least 91 days; most cover 91–123 (median 98). One covers 824, because its activation was mapped repeatedly over months.

## Inputs

Fusion uses three daily satellite sources plus the gold maps to test against. Every source is turned into one value per 0.01° cell per day: flood, dry or "couldn't see". "Couldn't see" is never counted as dry.

| input | what it is | pixel | years | read from | cell is flood when | cell is "couldn't see" when |
| --- | --- | --- | --- | --- | --- | --- |
| **GFM** | Copernicus Global Flood Monitoring, Sentinel-1 radar | 20 m, read at 80 m | 2015 on | EODC STAC (`stac.eodc.eu`), collection GFM: layers `ensemble_flood_extent`, `exclusion_mask`, `advisory_flags` | at least 30% of its observed pixels are flood | under half its pixels observed: no Sentinel-1 pass that day, excluded pixels, or GFM's own snow/frozen-ground flag |
| **FloodScan** | AER flooded fraction (SFED) from passive microwave | about 9 km | 1998 on; outside Africa to 2026-03-31 | Africa: prod blob `raster/floodscan/daily/v5/processed/` (daily COGs). Asia-Australia and South America: dev blob `raster/floodscan/daily/v5/raw/` (one NetCDF per region) | the 9 km cell it sits in is at least 30% flooded | outside FloodScan's regions, or after its archive ends (2026-03-31 outside Africa). Inside, it has no cloud mask and always answers |
| **GFDS** | JRC/GDACS Global Flood Detection System, passive microwave | about 9 km (0.09°) | 1997 to 2026-09-14 (feed ended); used from 2015 | `gdacs.org/flooddetection/DATA`: daily signal GeoTIFF plus GFDS's long-term mean | the signal's drop below its long-term mean implies at least 30% water | days GDACS has no file for, and pixels GFDS masks itself |
| **microwave vote** | FloodScan and GFDS combined into one vote, since they measure the same thing |  |  |  | either says flood | both couldn't see (possible only outside FloodScan's regions) |
| **VIIRS** | NOAA VIIRS flood map, optical | 375 m | 2012–2020 and 2023-08 on (no files 2021-01 to 2023-07) | AWS `s3://noaa-jpss/`, `VFM_1day_GLB` | at least 30% of its observed pixels are flood | under half its pixels observed: cloud, shadow, snow or ice, no data |
| **gold maps** | analyst flood maps: Copernicus EMS rapid mapping and UNOSAT | vector, burned at 0.001° | 2015 on | dev blob `global/copernicus_ems/flood/gold/` and `global/unosat/gold/` | see Gold labels | not examined by the analyst |

**From pixels to cells.** Each product pixel goes to the cell that holds its centre. A cell is "couldn't see" if fewer than half its pixels were observed that day. Otherwise it is flood if at least 30% of the observed pixels are flood, and dry if not. FloodScan and GFDS pixels are larger than a cell (about 9 km), so every cell copies the value of the 9 km pixel it sits in.

**Why the sources have gaps.**

- **GFM:** Sentinel-1 passes over a given place only every 6 to 12 days, so GFM has no image on most days.
- **VIIRS:** it sees reflected sunlight like a camera, so cloud blocks it, and floods come with cloud.
- **Passive microwave** (FloodScan, GFDS): it measures the energy the ground itself gives off. It sees through cloud every day, but only at about 9 km.

The in-cell rules (at least half observed, at least 30% flood) are the same for every source and for gold. A river that floods 10% of a cell is therefore dry everywhere. MODIS was tested only as an experiment; see Other approaches tried. FloodScan is used under AER's commercial licence, and whether it covers the non-Africa regions is still to be confirmed.

## Grid, domain and strata

Every cell gets three fixed attributes before any satellite data is read: whether it can flood (the domain), its land type (the stratum), and, through its event, a continent and a river basin. Only the land type changes the model. Continent and basin are used for scope, for building the test folds and for reporting.

Land type matters because each source's accuracy depends on what covers the ground. Radar and optical sensors cannot see water under trees, and buildings confuse radar. So the model learns its hit and false-alarm rates separately for open, vegetated and urban cells. Each group of cells with one land type is a stratum.

| layer | data, file | rule | used for |
| --- | --- | --- | --- |
| grid |  | 0.01° cells (about 1.1 km at the equator), EPSG:4326 | everything |
| floodable | GLO-30 HAND (height above nearest drainage), 30 m; AWS `glo-30-hand`, 1° tiles | at least 30% of the cell lies less than 15 m above the nearest stream | domain |
| permanent water | JRC Global Surface Water v1.4 occurrence (1984–2021), 30 m; Google Cloud `global-surface-water`, 10° tiles | removed if at least half the cell is water at least 90% of the time | domain |
| **domain** |  | floodable and not permanent water; nothing outside it is predicted or scored | everything |
| land type | ESA WorldCover 2021 v200, 10 m; AWS `esa-worldcover`, 3° tiles | **urban** if built-up (class 50) covers at least 20% of the cell; else **vegetated** if tree cover (class 10) covers at least 30%; else **open**. Cropland, grassland, shrubs, bare ground, wetland and mangroves all count as open | **strata** |
| snow season (not used) | ERA5-Land monthly snow cover, 2014–2023, 0.25°; Copernicus CDS | a calendar month is snow season if its mean snow cover is at least 50% | dropped: 518 of the 520 snow-season gold flood cell-days are in Europe |
| continent | Natural Earth 1:50m countries, field `CONTINENT` | the country at the event's centre (nearest country if offshore) | scope, fold balance, reporting |
| river basin | HydroBASINS level 4 (v1c) | the basin nearest the event's centre | folds: a basin's events stay together |
| climate zone | Köppen main groups (Beck et al. 2018), 0.083° |  | only to draw the global random sample: 53 blocks with no gold maps, with every day of 2023–2025 prepared. 30 are random 0.25° blocks, one per climate zone × continent; 12 are 0.5° blocks centred on cities, because urban cells are rare; 11 are reserves added to reach enough cells in every land type. The sample gives the base flood rate (the prior), and ordinary-day precision will be measured on it |

The fine layers are read at every 4th pixel, so each cell's fractions rest on 80–900 samples. The rules and thresholds live in `configs/fusion.yaml` under `static`, and the code in `src/ds_flood_gfm/fusion/static.py`.

&#91;image: How each cell is classified: HAND, permanent water, WorldCover land type and the resulting domain and strata, for one window of Typhoon Yagi, northern Viet Nam, 2024\]

*How to read it.* One window of Typhoon Yagi (northern Viet Nam, September 2024):

1. The delta's low ground (dark purple) is floodable: at least 30% of each cell lies less than 15 m above the nearest stream (GLO-30 HAND). The hills to the west and north are not.
2. Permanent water is mostly the sea in the south-east, plus the river channels.
3. WorldCover sorts every cell into open, vegetated or urban, before any domain rule is applied.
4. Keeping only floodable cells that are not permanent water leaves 24,859 cells: 7,329 open, 13,206 vegetated and 4,324 urban.

The white corner at the bottom right of panels 2–4 is open sea with no data. Elsewhere in panel 2, white means the cell is almost never water. Drawn from the pipeline's cached layers by `scripts/fusion/figure_event_layers.py`.

## Water versus flood

Every product here detects **water**. A **flood** is water that is not normally there. Deciding what counts as normal is a separate step, done after the water is detected, and each product does it its own way. The gold maps make that call too, by eye. This section shows where they differ and what that does to the scores.

&#91;embedded content: Who counts which water as flood · three kinds of water; the gold maps, four products and fusion\]

**Three kinds of water.**

- **Permanent water:** river channels and lakes, water nearly all the time (JRC Global Surface Water occurrence of at least 90%). Every product and the gold maps leave it out of "flood", and cells that are mostly permanent water are removed from the domain.
- **Normally wet this month:** water that is usually there at this time of year: a wet-season floodplain, a wetland, a flooded rice paddy. A cell is normally wet in a month if at least 30% of its 30 m pixels were water in that calendar month in at least half the years from 1984 to 2021 (JRC monthly recurrence). This is where the products and gold disagree.
- **Unusual water:** water that is not normally there this month. This is the flood everyone agrees on.

**How each draws the line on normally-wet cells.**

- GFM subtracts its own reference water, so it reports only unusual water.
- GFDS compares each day's signal with its long-term mean, so wet-season water reads as above normal and counts as flood.
- VIIRS and FloodScan appear to count seasonal water as flood; this is not confirmed.
- The gold maps are split. On map days, of the labelled cells that are normally wet, 52% are labelled flooded and 48% dry. These cells are only 7% of the labelled cells, but hold 38% of the gold flood. One map can do both, as in the example below: the analyst left the main river channel out of the flood but counted a large seasonal wetland as flooded.

&#91;image: Bangladesh, 22 August 2017: normally-wet cells, the gold map, fusion's flood calls, and how each call is scored\]

*How to read it.* Bangladesh, 22 August 2017 (FL20170815BGD), one of the four Bangladesh floods that hold 40% of the gold flood. Fusion here is the cross-validated model, fitted without this event.

- **A:** cells normally wet in August. The Jamuna river and its side channels run north to south; the large area in the east is a seasonal wetland.
- **B:** the gold map. Inside the satellite footprint (the tilted rectangle) the analyst marked flooded cells (blue) and dry cells (sand). The river channel is mostly marked dry: the analyst left it out of the flood, presumably as normal water. The eastern wetland is marked flooded.
- **C:** fusion's flood calls. The square blocks are the microwave products' 9 km cells.
- **D:** how each of fusion's calls is scored. Green: a hit. Orange: fusion says flood on a cell gold calls dry but that is normally wet, mostly the river channel. Red: fusion says flood on a gold-dry cell that is normally dry, mostly the edges of the coarse microwave blocks. Grey: flood fusion missed.

**Is orange a false alarm?** Only if the target is flood. Fusion saw water there, and on a cell that is normally wet this month the water is probably really there; gold left it out of the flood, most likely as normal water. The scores count it as a false alarm because gold says dry. A product that reports water, and leaves "is this normal?" to a later, explicit step, is not wrong; that is the water-first design at the end of this section.

**How much this changes the scores.** Over all 170 events, on map days:

- fusion makes 202,714 flood calls on gold-dry cells, and 39,076 of them (19%) are on normally-wet cells, the orange kind;
- not counting those, fusion's precision would be 0.51 instead of 0.46;
- the other 81%, the red kind, are on cells that are normally dry. They are coarse microwave cells reaching past the flood edge, water the analysts did not map, water the JRC record misses (it cannot see through persistent cloud), or plain errors. These labels cannot tell them apart.

**Removing normal water after the fact does not work with these labels.** If fusion dropped every flood call on a normally-wet cell, as one more post-processing step, it would remove the 39,076 calls gold labels dry but also 82,037 calls gold labels flooded. Recall would fall from 0.43 to 0.22, precision from 0.46 to 0.36, and F1 from 0.44 to 0.27. Every single product lost the same way in the first run (VIIRS recall 0.41 to 0.35, microwave 0.47 to 0.39). The reason is that gold counts about half of that water as flood: a masked product cannot be scored fairly against unmasked gold. GFM's recall (all cells) is lower on normally-wet cells than elsewhere (0.11 against 0.16); whether that is its stricter definition or less coverage there has not been separated.

A second example, where little of the flood is normally wet:

&#91;image: Water versus flood on one day: gold map, normally-wet water in August, and GFM, VIIRS and FloodScan, Punjab, Pakistan, 28 August 2025\]

*How to read it.* Punjab, Pakistan, 28 August 2025 (EMSR838, one window).

- **Gold** maps 1,152 flooded cells along the river. Only 11% of them are normally wet in August: the river channel itself (top middle).
- **GFM** could not see a large patch (grey), which counts against its recall here (0.62, all cells). Where it saw, nearly every flood call was right: precision 0.97.
- **VIIRS** caught almost all of the mapped flood (recall 0.95). It also flagged water outside it (precision 0.63), some of which may be real water the analysts did not map.
- **FloodScan's** 9 km cells catch the flood only in coarse blocks: recall 0.37, precision 0.58.

Scores are for this one day, on gold-labelled cells.

**A cleaner design: detect water, then remove normal water once.**

&#91;embedded content: Where the normal-water step happens · today and water-first\]

Today each product removes normal water its own way before fusion sees it, so fusion mixes several definitions, and gold mixes them again. In a water-first design, each product gives fusion its water; fusion estimates the chance of water; and one normal-water baseline turns that into a chance of flood. The definition of flood becomes one setting that users can see and change. It would make the target clear. It would not by itself fix the over-confident probabilities, which persist on both kinds of cell (Results, Water versus flood: where the problems sit).

## Gold labels

A gold map gives each cell it examined one label for its map date: flooded, dry or none. Cells it did not examine are never scored.

Each map has two parts: the **flood polygons** and the **area the analyst examined**. Both are drawn onto a 0.001° grid, so each 0.01° cell holds 10 × 10 = 100 sub-pixels.

| label | rule |
| --- | --- |
| **flooded** | at least half the cell was examined, and at least 30% of the examined part is flood |
| **dry** | at least half examined, under 30% flood, and under 30% "flood possible" |
| **none** | under half examined; or examined, under 30% flood and at least 30% "flood possible" |

Maps of the same event on the same day are merged, and flood wins over dry.

**Where the thresholds come from.** The 30% flood share and the at-least-half rule are the same for gold and for every product, so a cell is "flooded" by one definition on both sides. As a result, a flood covering under 30% of a cell is invisible to scoring (see Not yet validated). These thresholds, the 30-day and 60-day periods, the 10% daily fading and the P ≥ 0.5 cut-off were all set in the project plan before any result was seen. None is tuned to the results.

### The days before and after each event

Gold maps exist only on flood days. To learn how often each source raises a **false alarm on ordinary days**, the model also needs dry labels away from the flood. These come from the 30 days before an event's first map and the 60 days after its last.

- In those days, every cell the maps examined is labelled **dry**,
- *except* cells that were flooded on any map day in their window. Those get **no label**, because floodwater takes weeks to drain. In open land in the first run (D-39), fusion still called flood on 48% of days 1–30 after the event in those cells, against 5% in cells that never flooded. This rule was decided on 14 October (ADR 0006); the old rule's scores are shown alongside.

These days are used to fit the false-alarm rates and to check reliability. Recall, precision and F1 are scored on gold-map days only.

**What the labels cannot tell us.** Analysts map what they saw; unmapped water inside the examined area counts as dry. Many Copernicus maps are drawn from Sentinel-1, the same satellite as GFM, which favours GFM. Results are therefore also reported for the events whose maps are not from Sentinel-1.

## How fusion and scoring work

### Fusion's probability

For each cell and day, fusion adds up evidence from the sources that observed it:

- a source saying **flood** raises the odds by its hit rate divided by its false-alarm rate, in that land type;
- a source saying **dry** lowers them by its miss rate (1 − hit rate) divided by its correct-dry rate (1 − false-alarm rate);
- **couldn't see** adds nothing.

**Odds, with an example.** Odds are P / (1 − P); a probability of 0.002 is odds of about 1 to 500. In open land, in the first run, the base flood rate is 0.16%, and VIIRS's hit rate is 0.66 with a false-alarm rate of 5.5%. One VIIRS flood report multiplies the odds by 0.66 / 0.055 = 12, which lifts P from 0.0016 to about 0.02. Reaching 0.5 takes several sources agreeing, or several days in a row.

Yesterday's value carries over, fading 10% a day back towards the base flood rate for that land type. The base rate comes from the global random sample: how often GFM sees flood on an ordinary day. Fusion counts as **flood when P ≥ 0.5**. The hit and false-alarm rates are fitted only on the events in the other four folds (see Testing).

**Fitting is counting.** For each source and land type, two shares are counted on the events used for fitting:

- **hit rate:** of the gold-flooded cells the source observed, the share it called flood;
- **false-alarm rate:** of the gold-dry cells it observed, the share it called flood.

`docs/methods.md` calls these sensitivity and 1 − specificity. Half a hit is added to each count and one to each total, so no rate is ever exactly 0 or 1. The daily fading works on the log of the odds, not on P.

### Scoring: one cell, one gold day, one count

&#91;image: A synthetic example: a gold map and a product become hits, misses and false alarms. Numbers come from the pipeline's own functions.\]

*The figure* uses a made-up gold map and product. The labels, values and counts in it come from the same code the pipeline runs. Flood drawn outside the examined area (the left column) is ignored. The product pixels here are about 230 m, but the rule is the same for any pixel size.

|  | gold: flooded | gold: dry |
| --- | --- | --- |
| product: flood | **hit** | **false alarm** |
| product: dry | **miss** | **correct dry** |
| product: couldn't see | miss (all cells) / dropped (observed cells) | correct dry (all cells) / dropped (observed cells) |

- **Recall** = hits ÷ (hits + misses): the share of gold-flooded cells the product flags.
- **Precision** = hits ÷ (hits + false alarms): the share of its flood calls that gold confirms.
- **F1** = 2 × hits ÷ (2 × hits + misses + false alarms): the balance of recall and precision, high only when both are. It is the headline single score.
- **CSI** = hits ÷ (hits + misses + false alarms): of all cells the product or gold calls flooded, the share where both do. It ranks products the same as F1 and appears only where criterion (b) needs it.

**A worked example.** One gold day, 100 labelled cells, 30 of them flooded. A product flags 25 cells, 20 of them among the 30 flooded. That is 20 hits, 5 false alarms, 10 misses and 65 correct dry: recall 0.67, precision 0.80, F1 0.73.

### Two views, and how totals are formed

- **All cells** answers "what does a user get on a given day?". A cell the product couldn't see counts against it as a miss, which lowers recall but never precision.
- **Observed cells** answers "how good is the product when it looks?". It scores only the cells a product saw, and fusion is scored on the same cells for a fair comparison.
- **Totals** are formed as described in From counts to totals: large floods weigh more, so every score is also reported per continent and per land type.

**Precision here is measured inside mapped flood areas on flood days,** where about a tenth of labelled cells on map days are flooded (12% in open land). On ordinary days far fewer are, so precision there is much lower. Measuring it needs non-event data: the global random sample, with GFM as stand-in truth. The event-level false-event check below covers part of this.

## Testing: five folds by river basin

Every event is tested once, by a model fitted without it. A model scored on the events it was fitted on looks better than it is, because it has already seen them. The events are split into five folds. For each fold, the hit and false-alarm rates are fitted on the other four folds, and the fold's events are scored. The scores are then pooled over all five folds.

- **Whole river basins stay in one fold** (HydroBASINS level 4), so a neighbouring flood on the same river never helps score its neighbour.
- **Folds are balanced on flooded area first:** the basins with the most gold flood are spread across folds, then the rest are placed to even out each continent's events. A basin cannot be split, and one dominates: four UNOSAT floods in Bangladesh (2017, 2018, 2020 and 2022) share one HydroBASINS level-4 basin and hold 40% of all gold flooded cell-days. Their fold holds 43% of the open-land flood and 62% of the vegetated flood; each other fold holds 8–17%. When that fold is scored, its rates are fitted on the remaining 57% of the open-land flood and 38% of the vegetated flood.
- **It is cheap.** Fitting is counting, so five fits cost little. Scoring all events once took about two minutes on the laptop, under both rules for the days before and after.
- **Anything tuned is tuned inside each fold,** never on the fold being scored.

Balance of the five folds over the 170 events (99 basins). `scripts/fusion/make_folds.py` wrote the fold file, `results/event_folds.csv`, once, and it is frozen. Flood columns are each fold's share of that land type's gold flooded cell-days:

| fold | Africa | Asia | Oceania | South America | open-land flood | vegetated flood | urban flood |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 12 | 4 | 4 | 43% | 62% | 14% |
| 2 | 16 | 13 | 2 | 3 | 17% | 11% | 26% |
| 3 | 16 | 12 | 3 | 3 | 14% | 10% | 12% |
| 4 | 16 | 12 | 2 | 3 | 15% | 10% | 32% |
| 5 | 16 | 12 | 2 | 3 | 11% | 8% | 16% |

No event set stays completely untouched: the first run's 49 test events have already been used to compare variants. To limit the effect, every rule on this page was fixed before the results were computed.

## From counts to totals

Every score on this page is built from the same counts, in these steps:

1. **One count per labelled cell, per day, per product:** hit, miss, false alarm or correct dry. Every product, and fusion, is scored on the same labelled cells.
2. **Per event.** Counts are summed over each window's cells and days, then over the event's windows, and kept per event, land type and kind of day. Every number below is a sum of these.
3. **Fitting, per fold.** The counts of the four folds not being scored, per land type, give each source's hit rate (map days, the only days with flood labels) and false-alarm rate (map days plus every day before and after), counted on floodable cells only, where fusion makes its calls.
4. **Scoring.** Each fold's events are scored with the rates fitted on the other four folds. The map-day counts of all 170 events, each tested once, are summed, and recall, precision and F1 are computed from the sums: overall, per continent, per land type and per product.
5. **Pooled, not averaged.** A large flood has more cell-days, so it weighs more; five events hold 47% of all gold flooded cell-days, and one basin's four Bangladesh floods 40% (named under Testing). That is why results are also shown per continent and land type, with intervals from resampling whole events (a bootstrap: the scores are recomputed on many random re-draws of the events, and their spread shows how much they depend on which events happened to be mapped).
6. **Event level.** One yes or no per event (caught or not) and one number (days to detection), reported as shares of events. There every event counts equally.

## Was the event caught?

Cell-level scores say how well a product maps a flood. Many uses only need to know whether it noticed the flood at all, and how early. These rules *were fixed on 14 October, before any result existed (ADR 0007)*:

| measure | rule | why |
| --- | --- | --- |
| **caught** | on some day from 3 days before the event's first map to 3 days after its last, the product flags at least 10% of the cells flooded on any map day of the window, and at least 5 cells | a product need not match the outline to have seen the flood; 10% with a floor of 5 cells is the rule the first run used |
| **days to detection** | the first day the "caught" rule holds, counted from the first gold map | early warning value |
| **false event** | the same rule applied to the same cells on ordinary days: days 30 to 10 before the first map, which leaves a buffer in case flooding began before mapping | without it, a product that always says flood would catch everything |
| **false events away from floods** | the same rule on the global random sample blocks (no gold, ordinary days). Deferred: a sample block has no gold footprint for the rule to count against, so it joins the ordinary-day precision work (Next steps, 6) | catches products that fire everywhere, not just near floods |

Reported per product and for fusion (P ≥ 0.5), per continent and per land type: the share of events caught, the false-event rate, and the median days to detection.

## Results

Every event is scored once, by rates fitted on the other four folds. All scores are on gold-map days in the 170 events: 404,014 gold flooded cell-days. Each score is pooled over cells, so large floods weigh more; brackets give the 95% interval from resampling events.

**In short.**

- Against the single products it combines, fusion has the highest F1 overall (0.44), in Asia (0.49), in open land (0.47) and in vegetated land (0.30). Its lead over the best single product, the microwave vote (0.38 overall), is within the uncertainty, and sits mostly on cells that are normally wet in that month (Water versus flood: where the problems sit).
- **A simple rule that needs no fitting matches fusion.** Take VIIRS where VIIRS saw the cell, else GFM, else the microwave vote: F1 0.46 against fusion's 0.44, a difference within the uncertainty. Weighting each river basin equally, the simple rule is clearly ahead (0.40 against 0.31). Fusion's one clear advantage is at event level: far fewer false events (34% of events against 64%). See A simpler bar: the best-available cascade.
- **Learning how the products' reports combine does not beat the cascade either.** The best version (no memory, threshold chosen in each fold) scores 0.49 pooled against the cascade's 0.46, but is level basin-weighted and on false events. Fusion's memory is what gives it few false events, and it costs F1: with these products, fusion can catch about as many floods as the cascade or raise far fewer false events, not both (Can fusion beat the cascade?).
- **On ordinary days, precision collapses for every product except GFM.** Fusion calls flood on about 1 in 100 floodable cells on days with no mapped flood, and only 1.7% of those calls agree with GFM where it looked; its ordinary-day precision is somewhere between 0.03 (judged by GFM, which misses most flood) and about 0.2 (the ceiling set by how often it calls flood), against 0.46 on map days. The noise in the microwave vote comes from GFDS (2.4% of cell-days flagged against FloodScan's 0.1%) (Ordinary days).
- **Dropping GFDS cuts fusion's ordinary-day false alarms by two thirds for 0.01 F1** (1.06 to 0.34 flood calls per 100 cells; false block-level events from 2.7% to 0.03% of days). Recommended (Dropping GFDS).
- **VIIRS is what puts fusion ahead.** Run without VIIRS, fusion scores 0.39, level with the microwave vote. Where VIIRS has files, it adds 0.08 F1 to fusion (0.06–0.10). VIIRS's own low score (0.22) is mostly missing files: its archive has none for 2021-01 to 2023-07, which holds 28% of the gold flood.
- **FloodScan does almost all of the microwave work.** With FloodScan alone in the microwave slot, fusion loses 0.01 F1; with GFDS alone, 0.06; with no microwave, 0.11.
- One river basin weighs heavily. Four UNOSAT floods in Bangladesh (2017, 2018, 2020 and 2022) share one HydroBASINS level-4 basin and hold 40% of all gold flooded cell-days. Without them, fusion scores F1 0.35, the microwave vote 0.34 and FloodScan 0.33, with overlapping intervals.
- Criteria: (a) fusion has a value on as many cells as the microwave vote, which covers almost every cell; (b) passes on the point estimate in open and vegetated land, with overlapping intervals, and fails in urban land, where GFM is better; (c) fails in every land type.
- Event level: fusion catches 68% of events, more than any single product except the microwave vote (74%), but raises a false event on 34% of them. GFM catches fewer (49%) and raises the fewest false events (13%).

### All cells

A product's gaps count as misses.

| product | recall | precision | F1 |
| --- | --- | --- | --- |
| fusion | 0.43 (0.30–0.51) | 0.46 (0.37–0.53) | 0.44 (0.34–0.50) |
| microwave vote | 0.40 (0.32–0.45) | 0.37 (0.31–0.43) | 0.38 (0.32–0.43) |
| FloodScan | 0.34 (0.25–0.39) | 0.42 (0.36–0.48) | 0.37 (0.31–0.42) |
| GFM | 0.14 (0.11–0.19) | 0.86 (0.78–0.91) | 0.24 (0.19–0.31) |
| GFDS | 0.17 (0.12–0.24) | 0.36 (0.29–0.44) | 0.23 (0.18–0.29) |
| VIIRS | 0.14 (0.10–0.22) | 0.52 (0.42–0.62) | 0.22 (0.16–0.31) |

VIIRS ranks last here mostly because its archive has no files for 2021-01 to 2023-07, which holds 28% of the gold flood; there every flooded cell counts as a VIIRS miss. Where it has files, VIIRS is what puts fusion ahead (VIIRS: what it adds, below).

**Without the Bangladesh basin** (166 events, 244,189 gold flooded cell-days):

| product | recall | precision | F1 |
| --- | --- | --- | --- |
| fusion | 0.34 | 0.38 | 0.35 (0.29–0.42) |
| microwave vote | 0.37 | 0.31 | 0.34 (0.28–0.39) |
| FloodScan | 0.29 | 0.38 | 0.33 (0.26–0.39) |
| VIIRS | 0.18 | 0.47 | 0.26 (0.17–0.37) |
| GFDS | 0.20 | 0.31 | 0.25 (0.19–0.29) |
| GFM | 0.14 | 0.82 | 0.24 (0.16–0.34) |

**With the old rule for the days before and after** (every examined cell dry): fusion 0.38 · 0.46 (F1 0.41), against 0.43 · 0.46 (F1 0.44) with the adopted rule. Precision is the same; recall is lower, because false-alarm rates fitted on draining floodwater give flood reports too little weight.

### A simpler bar: the best-available cascade

A fair test of fusion is whether it beats the simplest way of combining the same products, one that needs no fitting at all. The **cascade** takes, for each cell and day, the first product that saw it: VIIRS, else GFM, else the microwave vote. (A colleague's suggestion; the order was fixed before running. GFM first is shown only as a check that the order does not matter.)

The same comparison is also shown under views that stop one basin or the label problems from deciding the answer. All the views were fixed before any of them was computed for the cascade (`scripts/fusion/cv_baselines_and_views.py`):

- **pooled:** the reference score used everywhere on this page;
- **basin-weighted:** each river basin's flood counts once (the 72 basins with at least 50 gold flooded cell-days), so the Bangladesh basin no longer holds 40% of the weight;
- **by gold map sensor:** map days whose maps come from Sentinel-1, other radar, optical imagery, or a mix;
- **normally-wet cells not scored:** the cells where gold is split between flood and dry are left out (Water versus flood).

F1, gold-map days, all cells:

| view | fusion | cascade | fusion minus cascade (95% interval) | GFM-first cascade | best single product |
| --- | --- | --- | --- | --- | --- |
| pooled (the reference) | 0.44 | 0.46 | −0.015 (−0.092 to +0.045) | 0.44 | microwave vote 0.38 |
| basin-weighted (72 basins) | 0.31 | 0.40 | −0.083 (−0.124 to −0.041) | 0.39 | FloodScan 0.28 |
| gold map from Sentinel-1 | 0.43 | 0.42 | +0.008 (−0.140 to +0.087) | 0.40 | microwave vote 0.36 |
| gold map from other radar | 0.54 | 0.53 | +0.012 (−0.119 to +0.125) | 0.54 | microwave vote 0.48 |
| gold map from optical | 0.43 | 0.49 | −0.059 (−0.114 to +0.006) | 0.48 | FloodScan 0.41 |
| gold map from a mix or unknown | 0.41 | 0.47 | −0.057 (−0.185 to +0.015) | 0.46 | microwave vote 0.39 |
| normally-wet cells not scored | 0.36 | 0.46 | −0.097 (−0.145 to −0.057) | 0.46 | FloodScan 0.34 |
| Africa | 0.31 | 0.42 | −0.105 (−0.150 to −0.048) | 0.42 | FloodScan 0.34 |
| Asia | 0.49 | 0.46 | +0.025 (−0.067 to +0.084) | 0.44 | microwave vote 0.40 |
| Oceania | 0.45 | 0.58 | −0.130 (−0.286 to −0.054) | 0.60 | VIIRS 0.53 |
| South America | 0.34 | 0.38 | −0.037 (−0.193 to +0.070) | 0.37 | GFM 0.31 |
| open land | 0.47 | 0.48 | −0.010 (−0.096 to +0.070) | 0.46 | microwave vote 0.41 |
| vegetated land | 0.30 | 0.34 | −0.044 (−0.105 to +0.009) | 0.34 | FloodScan 0.27 |
| urban land | 0.16 | 0.22 | −0.059 (−0.129 to −0.004) | 0.22 | GFM 0.24 |

The interval compares the same resampled events (basins, for the basin-weighted view) for fusion and the cascade. Pooled recall and precision: fusion 0.43 and 0.46; cascade 0.41 and 0.52.

Was the event caught (same rules as everywhere on this page):

| product | events | caught | false event | caught, no false event | median days to detection |
| --- | --- | --- | --- | --- | --- |
| fusion | 148 | 68% | 34% | 36% | −2 |
| cascade: VIIRS, else GFM, else microwave | 148 | 86% | 64% | 24% | −2 |
| cascade: GFM first (check) | 148 | 86% | 64% | 25% | −2 |

**Reading.**

- **On the reference score, the cascade and fusion are level:** 0.46 against 0.44. The cascade has higher precision (0.52 against 0.46) at similar recall.
- **Wherever one basin or the label problems stop deciding the answer, the cascade is ahead:** basins weighted equally (0.40 against 0.31), normally-wet cells left out (0.46 against 0.36), and in Africa, Oceania and urban land. Fusion is ahead only in Asia, home of the Bangladesh basin, and within the uncertainty.
- **By gold sensor, they are level on radar maps;** the cascade is ahead on optical maps (0.49 against 0.43), which are drawn from imagery like VIIRS's.
- **Fusion's one clear advantage is at event level.** It raises a false event on 34% of events, the cascade on 64%: where VIIRS and GFM saw nothing, the cascade falls back to the microwave vote, which flags the same cells for weeks. Caught without a false event: fusion 36%, the cascade 24%.
- **The order does not matter much:** GFM first scores 0.44 pooled.
- **What it means.** On cell-level scores, fusion as built does not beat the simplest way of combining the products. The clearest gap is on the cells VIIRS sees, where VIIRS alone beats fusion (Observed cells). A fusion model should at least learn what the cascade does by rule; that is the next step to try (Next steps, 9).

### Can fusion beat the cascade? Three switches

Fusion differs from the cascade in three ways. Each was switched on and off separately, which gives 2 × 2 × 2 versions of fusion (`scripts/fusion/cv_pattern_model.py`; the rules and the test for success were committed before it ran):

- **How evidence is combined.** As built, each product's report adds its own evidence, as if the products were independent. The alternative learns, on the training folds, how strongly each combination of reports signals flood: 27 combinations of flood, dry or couldn't see, per land type. "Couldn't see" itself carries no evidence, because cloud and flood come together on map days, and that would not carry over to ordinary days.
- **Memory.** As built, yesterday's belief carries over (decay 0.9), so evidence builds up over days. The alternative judges each day on its own, like the cascade.
- **Threshold.** As built, a cell is called flooded at P ≥ 0.5. The alternative uses the threshold with the best F1 on the training folds, chosen inside each fold.

The test for success was fixed beforehand: above the cascade on both pooled and basin-weighted F1, with fewer false events.

In the first run the threshold could go down to 0.02. Every fold chose 0.02, and a built-in check failed: the learned version without memory should at least match the cascade on its own training folds, and it did not. As planned for that case, the threshold was allowed down to 0.001; the check then passed. The table shows that follow-up. Rows at threshold 0.5 are the same in both runs.

| version | threshold | pooled F1 | vs cascade | basin-weighted F1 | vs cascade | false events | caught, no false event | reliability slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fusion as built: each product separately, memory, 0.5 | 0.5 | 0.44 | −0.015 (−0.092 to +0.045) | 0.31 | −0.083 (−0.124 to −0.043) | 34% | 36% | 0.43 |
| combinations learned, memory, 0.5 | 0.5 | 0.43 | −0.029 (−0.108 to +0.030) | 0.30 | −0.095 (−0.140 to −0.054) | 31% | 36% | 0.43 |
| each product separately, memory, threshold in fold | 0.001 | 0.49 | +0.036 (−0.033 to +0.085) | 0.37 | −0.027 (−0.063 to +0.012) | 90% | 3% | 0.43 |
| combinations learned, memory, threshold in fold | 0.001–0.002 | 0.49 | +0.032 (−0.035 to +0.081) | 0.36 | −0.035 (−0.073 to +0.005) | 88% | 5% | 0.43 |
| each product separately, no memory, threshold in fold | 0.002–0.02 | 0.45 | −0.011 (−0.024 to +0.005) | 0.38 | −0.012 (−0.051 to +0.015) | 65% | 22% | 1.25 |
| combinations learned, no memory, threshold in fold | 0.002–0.01 | 0.49 | +0.030 (−0.004 to +0.060) | 0.39 | −0.010 (−0.047 to +0.018) | 64% | 24% | 4.29 |
| cascade: VIIRS, else GFM, else microwave | – | 0.46 | – | 0.40 | – | 64% | 24% | – |

The two versions with no memory and threshold 0.5 are left out of the table: they almost never call flood (pooled F1 0.10 and 0). With a base flood rate of 0.2%, one day's evidence never reaches P 0.5 on its own.

**Reading.**

- **No version passes.** The closest is the learned combinations with no memory and a threshold chosen in each fold: pooled F1 0.49 against the cascade's 0.46 (interval −0.004 to +0.060), level basin-weighted (0.39 against 0.40) and level on false events (64%). It behaves like a learned cascade: slightly better on the reference score, no better elsewhere.
- **Learning the combinations helps only without memory.** With memory it changes nothing (0.43 against 0.44 at threshold 0.5). Without memory it adds 0.04 pooled over treating the products separately (0.49 against 0.45).
- **Memory sets fusion apart, both ways.** It gives fusion its low false-event rate (34%): a cell has to build up evidence over several days before it is called flooded. With a low threshold, the same memory makes false events explode (88–90%), because evidence that builds up over a wet season crosses a low threshold almost everywhere. Those versions chose the lowest threshold allowed even after the follow-up; going lower would only add false events.
- **The probabilities are not yet usable as probabilities.** With memory the slope stays at 0.43. Without memory and with the learned combinations they are far too low (slope 4.3), because the base rate is set from ordinary days and applied on flood days; only the threshold was moved to compensate.
- **What it means.** With these three products, fusion can catch about as many floods as the cascade, or raise far fewer false events, but not both. Which matters more is a decision about how the product will be used (Next steps, 10).

**The trade-off, threshold by threshold.** Every version scored at every threshold, the same threshold in every fold (`cv_pattern_model.py --sweep`):

&#91;image: F1 against the share of events with a false event, for every version of fusion at every threshold, with the cascade as a point\]

*How to read it.* Each point is one version at one threshold, the P at or above which a cell is called flooded. Left: pooled F1; right: basin-weighted F1. Up is better; left is better (fewer false events). The ringed point is fusion as built; the black diamond is the cascade.

| version | threshold | recall | precision | pooled F1 | basin-weighted F1 | events caught | false events | caught, no false event |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with memory, each product separately | 0.9 | 0.37 | 0.47 | 0.41 | 0.29 | 59% | 25% | 36% |
| with memory, each product separately (as built) | 0.5 | 0.43 | 0.46 | 0.44 | 0.31 | 68% | 34% | 36% |
| with memory, each product separately | 0.2 | 0.46 | 0.45 | 0.46 | 0.33 | 71% | 43% | 30% |
| with memory, each product separately | 0.05 | 0.50 | 0.44 | 0.47 | 0.34 | 75% | 48% | 29% |
| with memory, each product separately | 0.01 | 0.55 | 0.43 | 0.48 | 0.35 | 78% | 55% | 25% |
| with memory, each product separately | 0.002 | 0.59 | 0.42 | 0.49 | 0.36 | 81% | 63% | 20% |
| no memory, combinations learned | 0.01 | 0.45 | 0.55 | 0.49 | 0.41 | 83% | 62% | 24% |
| no memory, each product separately | 0.02 | 0.37 | 0.57 | 0.45 | 0.40 | 82% | 61% | 24% |
| cascade: VIIRS, else GFM, else microwave | – | 0.41 | 0.52 | 0.46 | 0.40 | 86% | 64% | 24% |

- **With memory, the threshold is a dial.** Each step down catches more floods and raises more false events: from 0.9 (false events on 25% of events, F1 0.41), through fusion as built at 0.5 (34%, 0.44), to 0.002 (63%, 0.49). Basin-weighted, it stays below the cascade at every threshold (at most 0.37 against 0.40).
- **Without memory there is no dial.** Above a threshold of about 0.02, fusion flags almost nothing; at or below it, it lands on the cascade's point, with about 62% false events.
- **Caught without a false event is highest with memory and a high threshold:** 36–37% at 0.5–0.8, against the cascade's 24%.
- **One point looks better than the cascade on everything:** combinations learned, no memory, threshold 0.01: pooled 0.49 against 0.46, basin-weighted 0.41 against 0.40, false events 62% against 64%. But that threshold was found by looking at these test scores. Chosen fairly, inside each fold, the same version only ties the cascade (table above), and the margins are within the uncertainty.
- **How to choose.** If the product feeds alerts, where a false alarm costs trust, fusion as built or a higher threshold. If it feeds maps, where missed water costs more, a low threshold with memory, or no memory; and then the simple cascade does about as well.

### Per continent

&#91;image: F1 per continent, five-fold cross-validation: fusion, VIIRS, microwave vote and GFM, with 95% intervals\]

*How to read it.* Each dot is one product's F1 on the gold-map days of that continent's events; the bar is its 95% interval from resampling events. FloodScan and GFDS, the two halves of the microwave vote, are in the tables.

| continent (events; gold flooded cell-days) | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM | best single |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Africa (80 events; 109,025 flooded cell-days) | 0.31 (0.21–0.41) | 0.26 (0.15–0.37) | 0.34 (0.23–0.42) | 0.34 (0.24–0.44) | 0.25 (0.16–0.33) | 0.11 (0.05–0.17) | FloodScan |
| Asia (61 events; 270,263 flooded cell-days) | 0.49 (0.36–0.54) | 0.17 (0.08–0.26) | 0.40 (0.33–0.46) | 0.39 (0.29–0.44) | 0.21 (0.14–0.29) | 0.30 (0.23–0.42) | microwave vote |
| Oceania (13 events; 19,553 flooded cell-days) | 0.45 (0.12–0.54) | 0.53 (0.00–0.68) | 0.43 (0.30–0.50) | 0.43 (0.23–0.51) | 0.31 (0.19–0.36) | 0.17 (0.06–0.38) | VIIRS |
| South America (16 events; 5,173 flooded cell-days) | 0.34 (0.29–0.39) | 0.17 (0.08–0.28) | 0.22 (0.16–0.26) | 0.20 (0.11–0.27) | 0.14 (0.04–0.21) | 0.31 (0.06–0.56) | GFM |

Recall · precision:

| continent | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM |
| --- | --- | --- | --- | --- | --- | --- |
| Africa | 0.28 · 0.37 | 0.18 · 0.48 | 0.36 · 0.31 | 0.31 · 0.39 | 0.20 · 0.32 | 0.06 · 0.80 |
| Asia | 0.49 · 0.49 | 0.10 · 0.56 | 0.41 · 0.40 | 0.34 · 0.44 | 0.14 · 0.39 | 0.18 · 0.87 |
| Oceania | 0.41 · 0.50 | 0.51 · 0.54 | 0.47 · 0.40 | 0.39 · 0.48 | 0.27 · 0.37 | 0.09 · 0.81 |
| South America | 0.45 · 0.28 | 0.10 · 0.45 | 0.29 · 0.18 | 0.20 · 0.21 | 0.12 · 0.15 | 0.19 · 0.83 |

- **Asia** holds two thirds of the gold flood, including the Bangladesh basin. Fusion leads there: 0.49 against 0.40 for the microwave vote.
- **Africa:** FloodScan and the microwave vote (0.34) are slightly ahead of fusion (0.31).
- **Oceania** (13 events): VIIRS 0.53, fusion 0.45, with wide intervals.
- **South America** (16 events, 1% of the gold flood): fusion 0.34. GFM's 0.31 has an interval of 0.06–0.56.

### Per land type

&#91;image: F1 per land type, five-fold cross-validation: fusion, VIIRS, microwave vote and GFM, with 95% intervals\]

| land type (events; gold flooded cell-days) | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM | best single |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open (167 events; 355,927 flooded cell-days) | 0.47 (0.36–0.54) | 0.23 (0.16–0.32) | 0.41 (0.34–0.47) | 0.40 (0.32–0.45) | 0.24 (0.18–0.31) | 0.24 (0.18–0.33) | microwave vote |
| urban (161 events; 2,209 flooded cell-days) | 0.16 (0.10–0.23) | 0.12 (0.08–0.17) | 0.11 (0.08–0.14) | 0.12 (0.09–0.15) | 0.08 (0.04–0.11) | 0.24 (0.15–0.33) | GFM |
| vegetated (159 events; 45,878 flooded cell-days) | 0.30 (0.23–0.34) | 0.17 (0.10–0.24) | 0.26 (0.20–0.30) | 0.27 (0.22–0.33) | 0.14 (0.10–0.19) | 0.25 (0.17–0.30) | FloodScan |

Recall · precision:

| land type | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM |
| --- | --- | --- | --- | --- | --- | --- |
| open | 0.44 · 0.51 | 0.15 · 0.54 | 0.40 · 0.43 | 0.33 · 0.49 | 0.17 · 0.41 | 0.14 · 0.87 |
| urban | 0.14 · 0.20 | 0.07 · 0.32 | 0.23 · 0.07 | 0.18 · 0.09 | 0.11 · 0.06 | 0.14 · 0.68 |
| vegetated | 0.37 · 0.25 | 0.11 · 0.41 | 0.40 · 0.19 | 0.36 · 0.22 | 0.12 · 0.17 | 0.15 · 0.77 |

Urban land is small (2,209 flooded cell-days) and hard for every product; GFM is best there.

### Observed cells

Each product only where it saw; fusion on the same cells. The last column is the share of all gold flooded cell-days the product saw.

| cells | product: recall · precision (F1) | fusion: recall · precision (F1) | share of gold flood seen |
| --- | --- | --- | --- |
| where VIIRS observed | 0.51 · 0.52 (0.52) | 0.44 · 0.45 (0.44) | 28% |
| where microwave vote observed | 0.40 · 0.37 (0.38) | 0.43 · 0.46 (0.44) | 100% |
| where FloodScan observed | 0.34 · 0.42 (0.37) | 0.43 · 0.46 (0.44) | 100% |
| where GFDS observed | 0.23 · 0.36 (0.28) | 0.43 · 0.46 (0.45) | 71% |
| where GFM observed | 0.28 · 0.86 (0.42) | 0.45 · 0.47 (0.46) | 51% |

- Fusion beats GFM on GFM's own cells (0.46 against 0.42), and the microwave products on theirs. VIIRS still beats fusion on the cells VIIRS saw (0.52 against 0.44).
- GFM had an image for 7% of floodable cell-days overall, but for 32% of the labelled cells on map days and 51% of the gold flooded ones. Map days fall on Sentinel-1 passes, most likely because many maps are drawn from Sentinel-1 (next section).

### Maps not drawn from Sentinel-1

Map days whose gold maps all came from other satellites (132 events, 158,796 gold flooded cell-days):

| product | recall · precision (F1) | F1 interval |
| --- | --- | --- |
| fusion | 0.49 · 0.43 (0.46) | 0.38–0.53 |
| fusion, old dry-day rule | 0.43 · 0.44 (0.43) | 0.36–0.51 |
| microwave vote | 0.47 · 0.37 (0.42) | 0.35–0.47 |
| FloodScan | 0.38 · 0.44 (0.41) | 0.33–0.48 |
| VIIRS | 0.23 · 0.52 (0.32) | 0.20–0.45 |
| GFDS | 0.26 · 0.36 (0.30) | 0.24–0.35 |
| GFM | 0.07 · 0.73 (0.12) | 0.08–0.16 |

GFM falls to F1 0.12 on these days, while fusion keeps 0.46. Scores on Sentinel-1-derived maps favour GFM.

### VIIRS: what it adds, and its archive gap

VIIRS saw 28% of the gold flooded cell-days. 55 events (114,306 flooded cell-days) fall in 2021-01 to 2023-07, when the VIIRS archive has no files, so its recall there is 0.

**What VIIRS adds to fusion.** Fusion run again without VIIRS, with the same folds, fitted rates and base rate (F1, all cells). The interval compares the same resampled events with and without VIIRS:

| events | fusion | fusion without VIIRS | added by VIIRS (95% interval) |
| --- | --- | --- | --- |
| 115 with VIIRS files | 0.44 | 0.36 | 0.08 (0.06–0.10) |
| 55 in the gap | 0.45 | 0.45 | none: no VIIRS files |
| all 170 | 0.44 | 0.39 | 0.06 (0.04–0.08) |

- Without VIIRS, fusion is level with the microwave vote: 0.39 against 0.38 overall, and 0.36 against 0.37 where VIIRS has files. Fusion's lead over every single product comes from VIIRS.
- Event by event, where VIIRS has files, it raises fusion's F1 on 68 of 109 events and lowers it on 4.

**By period**, every product (F1, all cells, 95% interval):

| product | VIIRS archive available (115 events) | VIIRS gap, 2021-01 to 2023-07 (55 events) |
| --- | --- | --- |
| fusion | 0.44 (0.32–0.51) | 0.45 (0.22–0.54) |
| VIIRS | 0.29 (0.21–0.38) | 0.00 (0.00–0.00) |
| microwave vote | 0.37 (0.29–0.41) | 0.42 (0.28–0.50) |
| FloodScan | 0.35 (0.27–0.40) | 0.43 (0.27–0.50) |
| GFDS | 0.20 (0.14–0.26) | 0.30 (0.19–0.36) |
| GFM | 0.23 (0.16–0.27) | 0.29 (0.15–0.45) |

The two columns are different events, and every product scores higher on the gap events, so this table does not measure what the gap costs. Within each column: where VIIRS has files, fusion leads the microwave vote by 0.07; in the gap, where fusion runs on GFM and microwave only, it leads FloodScan by 0.02, within the uncertainty.

Outside the gap VIIRS scores F1 0.29, against 0.49 on the first run's 29 test events in these continents. The likely reason is cloud over the large monsoon floods added in Asia; this has not been measured.

**What the gap costs.** Probably an optical input's worth on 28% of the gold flood: where VIIRS has files, it adds 0.08 F1. MODIS could give fusion an optical input there (Next steps, 5). How much it would add is not known: MODIS has never been run through fusion, and its accuracy is not VIIRS's.

### Which microwave product does the work

The microwave vote says flood if FloodScan or GFDS says flood (Inputs). To see which of the two does the work, fusion was run again with the microwave slot filled by FloodScan alone, then GFDS alone, then nothing. Everything else is unchanged: the same folds, the same GFM and VIIRS rates, and the same base rate. Each microwave choice gets its own hit and false-alarm rates, fitted on the other four folds (`scripts/fusion/cv_microwave_components.py`; the rules were fixed before it ran).

| microwave slot | recall | precision | F1 | F1 change from as built | events caught / false / caught, no false | reliability slope |
| --- | --- | --- | --- | --- | --- | --- |
| microwave vote (as built) | 0.43 | 0.46 | 0.44 (0.35–0.50) | – | 68% / 34% / 36% | 0.43 |
| FloodScan only | 0.40 | 0.47 | 0.43 (0.33–0.48) | −0.011 (−0.019 to −0.002) | 61% / 32% / 32% | 0.43 |
| GFDS only | 0.30 | 0.54 | 0.39 (0.31–0.45) | −0.056 (−0.097 to −0.012) | 61% / 28% / 35% | 0.52 |
| no microwave | 0.23 | 0.64 | 0.34 (0.24–0.41) | −0.106 (−0.172 to −0.043) | 50% / 24% / 28% | 0.65 |

The change column compares the same resampled events with and without the swap. The reliability slope here pools all land types on map days (target 0.8–1.2).

F1 by continent and VIIRS period (events in brackets):

| events | microwave vote (as built) | FloodScan only | GFDS only | no microwave | FloodScan only, change | GFDS only, change |
| --- | --- | --- | --- | --- | --- | --- |
| Africa (80) | 0.31 | 0.31 | 0.29 | 0.25 | 0.000 (−0.008 to +0.010) | −0.021 (−0.069 to +0.024) |
| Asia (61) | 0.49 | 0.47 | 0.42 | 0.36 | −0.016 (−0.025 to −0.003) | −0.069 (−0.132 to −0.014) |
| Oceania (13) | 0.45 | 0.46 | 0.45 | 0.44 | +0.005 (−0.015 to +0.024) | −0.008 (−0.079 to 0.000) |
| South America (16) | 0.34 | 0.34 | 0.32 | 0.36 | −0.002 (−0.044 to +0.047) | −0.023 (−0.107 to +0.060) |
| VIIRS available (115) | 0.44 | 0.42 | 0.41 | 0.39 | −0.015 (−0.022 to −0.004) | −0.033 (−0.068 to +0.017) |
| VIIRS gap (55) | 0.45 | 0.45 | 0.33 | 0.17 | 0.000 (−0.015 to +0.014) | −0.125 (−0.205 to −0.028) |

**Reading.**

- **FloodScan does almost all of the microwave work.** With FloodScan alone, fusion loses 0.011 F1 overall, and nothing in Africa or in the VIIRS gap. GFDS adds a little on top of it, mostly in Asia (0.016).
- **GFDS alone is clearly weaker:** 0.056 lower overall, mostly in recall (0.30 against 0.43), and 0.125 lower in the VIIRS gap, where fusion leans on microwave most.
- **The microwave slot matters.** Without any microwave input, fusion loses 0.11 F1 overall, and 0.28 in the VIIRS gap, where only GFM is left.
- **The trade-off.** Microwave input brings both recall and false alarms. GFDS alone raises fewer false events (28% against 34%) and gives better-calibrated probabilities (slope 0.52 against 0.43). With no microwave the slope reaches 0.65, at a large cost in recall. Part of the over-confidence comes in with the microwave evidence.

### Criteria

| land type | fusion CSI | best single (CSI) | (b) passes | slope, map days | slope, map and dry days | (c) passes |
| --- | --- | --- | --- | --- | --- | --- |
| open | 0.31 | microwave vote (0.26) | yes | 0.46 | 0.03 | no |
| vegetated | 0.18 | FloodScan (0.16) | yes | 0.24 | 0.01 | no |
| urban | 0.09 | GFM (0.13) | no | 0.22 | 0.01 | no |

- **(a) Coverage.** Fusion has a value on 99.99% of floodable cell-days, the microwave vote on 99.98%: equal, not better. GFM has one on 7%, VIIRS on 38%.
- **(b) Mapping skill** passes on the point estimate in open land (CSI 0.31 against 0.26) and vegetated land (0.18 against 0.16), with overlapping F1 intervals; it fails in urban land (0.09 against GFM's 0.13).
- **(c) Reliability** fails in every land type (next section).

### Reliability

&#91;image: Fusion reliability in open land on gold-map days: observed flood share against predicted probability, slope 0.46\]

*How to read it.* Open land, map days only. Each dot is one band of fusion's P (0–0.1, 0.1–0.2 and so on): across is the band's mean P, up is the share of its cells that gold calls flooded. On the dashed line, probabilities mean what they say.

- Below P 0.1, 7% of cells are flooded. From P 0.1 up to 1, 42–51% are flooded, whatever P says. Fusion's P mostly says whether some source reported flood, not how likely the flood is. Most cells sit at the two ends: 2.5 million cell-days below 0.1, 265,000 above 0.9.
- Vegetated and urban land look the same, lower: 16–26% flooded above P 0.1 in vegetated land, 8–26% in urban land.
- Over map days and the days before and after together, fusion's mean P in open land is 0.039 while 0.3% of labelled cells are flooded: about 13 times too high. Almost all of that is on the days before and after, where unmapped water also counts as dry.

### Was the event caught?

Rules in "Was the event caught?" above. 148 of the 170 events have at least 5 floodable cells flooded on a map day; the other 22 (13 with none) are left out here.

| product | events | caught | false event | caught, no false event | median days to detection |
| --- | --- | --- | --- | --- | --- |
| microwave vote | 148 | 74% (67%–80%) | 53% (45%–60%) | 24% (18%–31%) | −1 |
| fusion | 148 | 68% (59%–75%) | 34% (26%–41%) | 36% (29%–44%) | −2 |
| GFDS | 148 | 60% (52%–68%) | 39% (31%–47%) | 24% (18%–31%) | 0 |
| fusion, old dry-day rule | 148 | 59% (51%–66%) | 26% (19%–33%) | 36% (28%–43%) | −1 |
| FloodScan | 148 | 55% (48%–64%) | 32% (26%–41%) | 28% (21%–36%) | 0 |
| VIIRS | 148 | 53% (45%–61%) | 39% (31%–47%) | 16% (10%–23%) | −2 |
| GFM | 148 | 49% (41%–57%) | 13% (8%–19%) | 39% (32%–47%) | 0 |

Caught / false event, per continent (events):

| continent | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM |
| --- | --- | --- | --- | --- | --- | --- |
| Africa (72) | 60% / 31% | 54% / 38% | 67% / 51% | 49% / 31% | 57% / 38% | 29% / 10% |
| Asia (53) | 72% / 36% | 55% / 45% | 79% / 58% | 57% / 38% | 60% / 42% | 68% / 11% |
| Oceania (11) | 82% / 18% | 18% / 9% | 82% / 27% | 82% / 9% | 82% / 27% | 73% / 18% |
| South America (12) | 83% / 58% | 67% / 50% | 83% / 58% | 67% / 42% | 58% / 50% | 67% / 33% |

Caught / false event, per land type (events with at least 5 flooded cells of that type):

| land type | fusion | VIIRS | microwave vote | FloodScan | GFDS | GFM |
| --- | --- | --- | --- | --- | --- | --- |
| open (143) | 70% / 37% | 52% / 43% | 75% / 56% | 58% / 34% | 62% / 40% | 54% / 13% |
| urban (56) | 25% / 5% | 30% / 7% | 59% / 25% | 43% / 12% | 39% / 12% | 20% / 2% |
| vegetated (99) | 53% / 16% | 45% / 24% | 67% / 39% | 58% / 25% | 49% / 26% | 41% / 3% |

- Fusion notices more floods than any single product except the microwave vote. It raises far more false events than GFM (34% against 13%) and about as many as FloodScan (32%). Caught without a false event, fusion (36%) and GFM (39%) are about equal.
- The microwave vote catches 74% but raises a false event on 53%: it often flags the same cells for weeks before the flood. This is what the false-event check was added for.
- Median days to detection are −2 to 0 for every product: detection on, or just before, the first map. Part of that is the persistence above.

### Water versus flood: where the problems sit

This repeats the main results on two kinds of cell: those normally wet in that calendar month, and the rest (definitions and a worked map in Water versus flood, above). The question is whether the problems above come from the water-versus-flood mismatch. Same models, gold-map days, nothing refitted (`scripts/fusion/cv_normally_wet.py`; the rules were fixed before it ran).

F1, all cells:

| cells | share of labelled cells | share of gold flood | fusion | microwave vote | FloodScan | VIIRS | GFM | GFDS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normally wet | 7% | 38% | 0.60 (0.46–0.66) | 0.49 | 0.45 | 0.21 | 0.20 | 0.24 |
| other | 93% | 62% | 0.36 (0.29–0.42) | 0.34 | 0.34 | 0.23 | 0.27 | 0.22 |

Flood calls on gold-dry cells, which the scores count as false alarms. On normally-wet cells many of them are water the analyst treated as normal (the orange cells in the Bangladesh map). Each rate is the share of gold-dry cells a product saw where it called flood. Lift is how many times more often two products make that call on the same cell than they would if they were independent (1 = independent):

| pair | normally wet: share of dry cells both saw | normally wet: share of joint false alarms | false-alarm rates, normally wet | false-alarm rates, other | lift, normally wet | lift, other |
| --- | --- | --- | --- | --- | --- | --- |
| GFM + microwave vote | 5% | 41% | 5.6% / 26.4% | 0.5% / 9.5% | 1.6 | 3.7 |
| GFM + VIIRS | 4% | 35% | 5.6% / 31.9% | 0.5% / 7.1% | 2.4 | 10.3 |
| microwave vote + VIIRS | 3% | 18% | 22.5% / 23.7% | 5.9% / 3.4% | 2.2 | 8.1 |

Reliability, all land types:

| cells | flooded share at P below 0.1 | flooded share at P 0.1 to 1 | slope (target 0.8–1.2) |
| --- | --- | --- | --- |
| normally wet | 39% | 65%–71% | 0.29 |
| other | 4% | 27%–36% | 0.34 |

Event level, with each event's footprint split by whether a cell is normally wet in the month of the first map:

| product | normally-wet footprint (111 events): caught / false event | other footprint (144 events): caught / false event |
| --- | --- | --- |
| fusion | 64% / 44% | 65% / 30% |
| microwave vote | 75% / 60% | 72% / 53% |
| FloodScan | 55% / 38% | 57% / 30% |
| GFDS | 59% / 43% | 58% / 40% |
| VIIRS | 56% / 50% | 51% / 37% |
| GFM | 43% / 23% | 53% / 12% |

**Reading.**

- **Where the flood is.** Normally-wet cells are only 7% of the labelled cells, but they hold 38% of the gold flood.
- **Every product calls flood on gold-dry cells far more often when the cell is normally wet:** 3 to 11 times as often (GFM 5.6% against 0.5%). Much of that is water the analyst counted as normal, not a mistake about water. Fusion uses one false-alarm rate per land type, which is too low for normally-wet cells and too high for the rest, so fitting the two separately is justified (Next steps, 8).
- **Fusion's lead is larger on normally-wet cells:** 0.60 (0.46–0.66) against the microwave vote's 0.49 (0.42–0.54), a gap of 0.11; on other cells 0.36 against 0.34, a gap of 0.02. The intervals overlap in both. Much of fusion's advantage is VIIRS seeing water on cells that are often wet, which gold counts as flood about half the time.
- **It does not explain the over-confident probabilities.** Within each kind of cell, the share actually flooded barely changes from P 0.1 to P 1 (slopes 0.29 and 0.34). That points to the engine, which counts correlated evidence as if it were independent and carries it over from day to day, rather than to the definition of flood.
- **Nor most of the shared mistakes.** Two products calling flood on the same gold-dry cell is more common than chance predicts by 3.7 to 10 times on ordinary cells, but only 1.6 to 2.4 times on normally-wet cells. On normally-wet cells shared calls are common simply because the water is there for every product to see.
- **False events** are more frequent on normally-wet footprints (fusion 44% against 30%), but common elsewhere too (microwave vote 53%).

*Caveats.* The event-level split uses the month of the first map for the whole footprint, while the false-event days (30 to 10 days before) may fall in the month before. On normally-wet cells, 39% of cells at P below 0.1 are flooded: these are map days inside mapped floods, where flood is common anyway, so this is not only the base rate being set from GFM's stricter definition.

**What follows.** Scored against these gold maps, fusion estimates water outside permanent water on event days, with seasonal water counted as flood about half the time. Two ways to make that explicit: separate rates for normally-wet cells (Next steps, 8), and later the water-first design (Water versus flood, above).

### Ordinary days: how often each product cries flood when there is none

Every precision above is from map days inside mapped floods, where 9% of labelled cells are flooded. On an ordinary day flood is about 60 times rarer, so the same false-alarm rate gives far lower precision. Two measurements, rules fixed before running (`scripts/fusion/cv_ordinary_days.py`):

- **The days before and after each event:** the 30 days before the first map and 60 after the last, cells labelled dry under the adopted rule, cross-validated. Still near floods, in the same places, and unmapped water counts as dry, so this is an upper bound on false alarms.
- **The random sample:** 30 places in the four continents with no mapped event, every day of 2023–2025 (32,880 block-days). There is no gold there, so GFM is the referee where it looked: its precision against gold is 0.86, but it sees only 7% of cell-days and misses 86% of flood, so agreement with GFM is a floor, not the truth. The numbers that need no referee are each product's flood-call rate against the base flood rate, and the block-level false events. Fusion here uses rates fitted on all 170 events; the sample was never fitted on.

| product | days before and after events: flood calls per 100 dry cells | before / after +1–30 / after +31–60 | ordinary days (random sample): flood calls per 100 floodable cells | of its calls where GFM also looked, share GFM agreed | block-days with a false "event" (≥10% of cells flagged) |
| --- | --- | --- | --- | --- | --- |
| fusion | 3.00 | 1.91 / 4.01 / 3.08 | 1.06 | 1.7% | 2.7% |
| cascade | 2.65 | 2.35 / 3.15 / 2.45 | 1.33 | 14.1% | 2.7% |
| microwave vote | 3.34 | 2.85 / 4.01 / 3.17 | 2.44 | 0.5% | 5.3% |
| FloodScan | 2.01 | 1.75 / 2.43 / 1.85 | 0.10 | 3.8% | 0.1% |
| GFDS | 1.70 | 1.39 / 2.03 / 1.69 | 2.38 | 0.2% | 5.2% |
| VIIRS | 0.73 | 0.61 / 0.86 / 0.71 | 0.39 | 2.9% | 0.2% |
| GFM | 0.03 | 0.03 / 0.03 / 0.03 | 0.00 | 100% (the referee) | 0.0% |

**Implied ordinary-day precision, open land** (a formula, not a measurement: map-day recall × base rate ÷ (that + false-alarm rate × (1 − base rate)), with the dry-day false-alarm rate and the base rate of 0.23%):

| product | map-day recall | map-day precision | dry-day false-alarm rate | implied ordinary-day precision |
| --- | --- | --- | --- | --- |
| fusion | 0.44 | 0.51 | 3.71% | 0.03 |
| microwave vote | 0.40 | 0.43 | 3.76% | 0.02 |
| FloodScan | 0.33 | 0.49 | 2.30% | 0.03 |
| GFDS | 0.17 | 0.41 | 1.99% | 0.02 |
| VIIRS | 0.15 | 0.54 | 1.04% | 0.03 |
| GFM | 0.14 | 0.87 | 0.04% | 0.46 |

**Far from the flood.** The expert maps examine areas much larger than the flood and label the rest dry. Taking only days 31–60 after the last map and only cells at least 5 cells (about 5 km) from anything flooded on a map day gives the quietest labelled cells we have (`scripts/fusion/cv_far_from_flood.py`, rules fixed first). False alarms there are about half those near the flood, but still well above GFM's:

| product | far from the flood: flood calls per 100 dry cells (open land) | within 5 km of the flood | implied ordinary-day precision, far cells |
| --- | --- | --- | --- |
| fusion | 2.24 | 6.51 | 0.04 |
| fusion, FloodScan alone | 1.76 | 5.31 | 0.05 |
| cascade | 1.85 | 4.63 | – |
| microwave vote | 2.25 | 5.70 | 0.04 |
| FloodScan | 1.29 | 3.44 | 0.06 |
| GFDS | 1.26 | 3.17 | 0.03 |
| VIIRS | 0.42 | 2.05 | 0.07 |
| GFM | 0.01 | 0.07 | 0.69 |

These cells are still inside mapped events, in the same places, and water the analysts did not map counts as dry, so the rates are an upper bound on false alarms and the implied precisions a lower bound. Together with the bounds above, fusion's ordinary-day precision is most likely in the range 0.04–0.2. The Sentinel-2 referee below narrows it to 0.04–0.09 outside normally-wet land.

**The Sentinel-2 referee.** Sentinel-2's scene classification marks water at 20 m from an optical sensor none of the four inputs use (`scripts/fusion/experiment_s2_referee.py`, rules fixed first, 8d3b9af and b9d1bcc). On the 30 sample blocks, every cloud-free day of 2023–2025 (3.0 million observed cell-days on 1,073 scene-days), a cell is "water" by the same 50% observed / 30% rules as every other layer, and "unusual water" if it is water on a cell that is not normally wet in that month. Sentinel-2 sees water on 2.7 cells in 100 and unusual water on 0.94, about five times the 0.2 flood base rate, so unusual water is not all flood either.

Fusion's ordinary-day calls split three ways: 69% on cells where Sentinel-2 saw no water that day, 22% on water in normally-wet cells, 9% on unusual water. So outside normally-wet land fusion's ordinary-day precision is 0.04–0.09, and a further fifth of its calls land on water that is normally there, which is not strictly wrong (see Water versus flood). Dropping GFDS halves the calls and roughly doubles precision against both labels while keeping most of the recall:

| product | flood calls per 100 observed cell-days | precision vs water | precision vs unusual water | recall vs unusual water |
| --- | --- | --- | --- | --- |
| fusion (as built) | 1.38 | 0.31 | 0.09 | 0.13 |
| fusion, FloodScan alone | 0.53 | 0.48 | 0.18 | 0.10 |
| cascade | 1.47 | 0.26 | 0.09 | 0.14 |
| VIIRS | 0.81 | 0.38 | 0.14 | 0.12 |
| microwave vote | 2.97 | 0.15 | 0.04 | 0.11 |
| GFDS | 2.89 | 0.14 | 0.03 | 0.11 |
| FloodScan | 0.13 | 0.48 | 0.15 | 0.02 |
| GFM | 0.00 | too few calls to score |  |  |

Caveats. Sentinel-2 labels only cloud-free days, which are drier than average, so this is ordinary-day behaviour, not flood-day skill. Water under trees is invisible to it, so vegetated land reads worse than it is. The "no water" calls include water that drained after fusion's memory picked it up and shallow or turbid water the classifier misses, so they are not all false. The result is concentrated: 16 of 30 blocks have any Sentinel-2 water, Asia holds 96% of it, and three blocks hold 90% of fusion's calls (one Asian city-reserve block 43%); without that block fusion's precision is 0.30 against water and 0.06 against unusual water (a sensitivity, not pre-registered). GFM almost never flagged on cloud-free sample days, so it has no usable referee number. Ten of 9,629 scenes exist only on a pay-to-read bucket and were left out (listed in `results/experiment_s2_referee_excluded.csv`).

**Reading.**

- **On ordinary days, fusion says flood on about 1 in 100 floodable cells**, and where GFM also looked, GFM agreed with only 1.7% of those calls. But GFM is a poor referee: against the expert maps it misses 86% of flood, so some of fusion's "false" calls are water GFM did not see. The bounds that do not depend on a referee are these: real flood covers about 0.2 cells in 100 on an ordinary day, and fusion calls flood on 1.06 in 100, so even if every real flood were among its calls, precision could not exceed about 0.2. So fusion's ordinary-day precision lies between 0.03 (if GFM were right) and about 0.2 (if fusion were), against 0.46 on map days. Every product except GFM sits in the same range. GFM's calls are rare and mostly right.
- **The microwave vote's ordinary-day noise is GFDS, not FloodScan.** On the random sample FloodScan calls flood on 0.10% of cell-days, GFDS on 2.4%; the vote (either says flood) inherits GFDS's rate. Near events the two are similar (2.0% and 1.7%), which is why the map-day scores never showed it. Combined with the earlier finding that FloodScan alone costs fusion only 0.01 F1, dropping GFDS from the vote looks right; a direct test is listed below.
- **False alarms are highest in the month after an event** (fusion 4.0 per 100 cells, against 1.9 before): some of that is water still draining, counted as dry because it lies outside the cells the maps flooded.
- **Block-level false events away from floods** (the check deferred in ADR 0007): fusion and the cascade flag at least 10% of a block's floodable cells on 2.7% of block-days, about 10 days a year per place; the microwave vote and GFDS on 5%; FloodScan, VIIRS and GFM almost never.
- **Caveats.** The sample is 30 places, 8 of them cities in Asia. GFM as referee sees only 7% of cell-days and misses most flood (its recall against gold is 0.14), so "share GFM agreed" understates true precision for every product; the ranking between products is the usable part.

### Dropping GFDS: three times fewer false alarms on ordinary days, for 0.01 F1

Step 12 re-ran the ordinary-day measurements with FloodScan alone in fusion's microwave slot (and in the cascade), rates refitted the same way (`cv_ordinary_days.py --microwave floodscan`, added and committed after the first run and before this one):

|  | microwave vote (as built) | FloodScan alone |
| --- | --- | --- |
| map-day F1 (from "Which microwave product does the work") | 0.44 | 0.43 (−0.011, −0.019 to −0.002) |
| ordinary days: fusion's flood calls per 100 floodable cells | 1.06 | 0.34 |
| of those calls where GFM also looked, share GFM agreed | 1.7% | 3.8% |
| block-days with a false "event" (≥10% of cells flagged) | 2.7% (about 10 days a year per place) | 0.03% (about 1 day in 10 years) |
| days before and after events: flood calls per 100 dry cells | 3.00 | 2.58 |
| cascade, ordinary-day flood calls per 100 cells | 1.33 | 0.42 |

**Reading.** GFDS supplies most of fusion's ordinary-day false alarms and almost none of its skill on map days. With FloodScan alone, fusion says flood three times less often on days with no flood, and false block-level events all but disappear, at a cost of 0.01 F1 on map days. GFDS's feed also stopped on 14 September 2026, so a live product could not use it anyway. **Recommendation:** make the microwave input FloodScan alone (`sources.gfds.enabled: false`). This changes fixed decision 2 of the original brief (the microwave vote), so it is yours to confirm. The remaining ordinary-day precision, about 0.03 implied, is still far below GFM's; that is the next problem, not this one.

### Separate rates for normally-wet cells: it scores worse

Step 8 tested fitting each product's hit and false-alarm rates per land type **and** whether the cell is normally wet in that month (six strata instead of three), everything else as built, same folds (`scripts/fusion/cv_normally_wet_strata.py`; rules and the expected result committed before running). The expectation was better precision on ordinary cells and little change elsewhere.

| view | fusion as built | with normally-wet strata | change (95% interval) |
| --- | --- | --- | --- |
| pooled F1 | 0.44 | 0.37 | −0.069 (−0.091 to −0.029) |
| basin-weighted F1 | 0.31 | 0.29 | −0.025 (−0.036 to −0.017) |
| normally-wet cells | 0.60 | 0.40 | −0.202 (−0.252 to −0.133) |
| other cells | 0.359 | 0.364 | +0.006 (+0.004 to +0.008) |
| events caught / false events | 68% / 34% | 61% / 27% |  |
| reliability slope (map days) | 0.43 | 0.39 |  |

The fitted rates say why. On normally-wet cells the false-alarm rates are 4–9 times higher than elsewhere (VIIRS 19.5% against 2.2%, microwave 13.7% against 3.6%, GFM 3.3% against 0.4%), so the new model rightly discounts a flood report there. But gold counts water on those cells as flood about half the time (Water versus flood), so discounting them loses recall exactly where the gold flood sits: recall on normally-wet cells falls from 0.53 to 0.28. On other cells the gain is real but tiny.

**Reading.** The better-specified model scores worse because the answer key is inconsistent on normally-wet cells. As built, fusion's single rate per land type happens to match gold's habit of calling that water flood. This is the water-versus-flood ambiguity again: until the target is defined (water, or water that is not normally there), fitting the rates more carefully cannot help. Not adopted.

### Fitted rates

Hit rate and false-alarm rate per source and land type, as the range over the five folds (adopted rule):

| source | land type | hit rate | false-alarm rate |
| --- | --- | --- | --- |
| GFM | open | 0.25–0.37 | 0.5%–0.6% |
| GFM | urban | 0.38–0.47 | 0.1%–0.2% |
| GFM | vegetated | 0.26–0.40 | 0.3%–0.3% |
| microwave vote | open | 0.37–0.42 | 3.0%–4.5% |
| microwave vote | urban | 0.20–0.27 | 4.2%–5.1% |
| microwave vote | vegetated | 0.31–0.42 | 1.8%–3.5% |
| VIIRS | open | 0.43–0.63 | 2.0%–3.4% |
| VIIRS | urban | 0.28–0.32 | 0.6%–0.8% |
| VIIRS | vegetated | 0.57–0.63 | 0.5%–1.1% |

The fold that holds the Bangladesh basin, and so is fitted without it, has the highest GFM hit rate in open land (0.37, against 0.25–0.28 in the others) and the lowest microwave hit rate (0.37). VIIRS's open-land hit rate varies across all folds (0.43–0.63).

### How this compares with the first run

The first run (one split, 49 test events on all continents, six strata, every examined cell dry before and after each event) gave fusion F1 0.44 and VIIRS 0.48. This run gives fusion 0.44 and VIIRS 0.22, on a different and larger set of events: the 51 added events include the three largest Bangladesh floods and many events in the VIIRS gap.

### What these scores cannot show

- Precision is measured inside mapped floods, where 12% of labelled open-land cells on map days are flooded. On ordinary days far fewer cells flood, so precision there will be far lower (Not yet validated).
- Pooled scores follow the largest floods; the Bangladesh basin alone is 40% of the gold flood.

## Other approaches tried

Four alternatives were tested on the first run's events. One, MODIS, is still worth adding to the optical vote, though its first-run numbers do not carry over (see its row); the other three are not used.

| approach | idea | result | verdict |
| --- | --- | --- | --- |
| **MODIS flood product** (NASA MCDWD) | fill the 2021-01 to 2023-07 period, when the NOAA VIIRS archive has no files | first run (52 events on all continents, the old dry-day rule): MODIS alone reaches recall 0.19 · precision 0.73 (F1 0.30), and MODIS-or-GFM 0.27 · 0.76 (F1 0.40); fusion without optical input reached 0.13 · 0.45 (F1 0.20). These do not carry over: in this analysis, fusion without optical input scores 0.40 · 0.52 (F1 0.45) on the 55 gap events, above MODIS-or-GFM's first-run score | **add** as part of the optical vote (VIIRS or MODIS), to give fusion an optical input in the gap, where VIIRS adds 0.08 F1 elsewhere; MODIS's value inside fusion is unmeasured. Needs MODIS over whole event windows, a Databricks job |
| **Agreement between sources** (no gold) | estimate each source's accuracy from how often the sources agree, without gold maps | the method assumes the sources err independently, but on gold-dry cells their false alarms coincide 4–18 times more often than that allows. Its "flood" class looks like seasonal water (2.4% of cell-days, 15 times the base flood rate) | not used; accuracy comes from gold |
| **Seasonal mask** | ignore flood calls on normally-wet cells | every source got worse, because gold counts seasonal water as flood (see Water versus flood) | not used |
| **Learning GFM from daily inputs** ("super-GFM") | train a model on GFM's own calls to estimate a GFM-like map on days without a Sentinel-1 pass | 0.25 · 0.77 (F1 0.38), no better than simply carrying GFM's last value forward (0.25 · 0.81, F1 0.38) | not used |

### CYGNSS: a first look at a fourth source

CYGNSS is a NASA constellation that measures GPS signals reflected off the ground; calm water reflects strongly. The UC Berkeley daily watermask built from it gives water, land or no data on a 0.01° grid, from August 2018, between latitudes 37.4° S and 37.4° N, a few days behind real time. Like the microwave products it sees through cloud, but at 1 km instead of 9 km. 148 of the 170 events, holding 82% of the gold flood, fall inside its dates and latitudes. It is read with your Earthdata token, fetching only the rows each window needs: about 11 MB a day for the largest window, instead of a 257 MB file.

**Checked before scoring:**

- **It is not a daily snapshot.** Over a Bangladesh window in July–August 2020, 6% of cells change from one day to the next, 13% over 8 days and 24% over 30 days, and the files' own metadata gives a 31-day coverage. It behaves like a rolling composite of several weeks. So it cannot enter fusion as fresh daily evidence: fusion's memory would count the same water many times.
- **Its cells are offset by half a cell from ours,** so each of our cells takes a quarter of four CYGNSS cells. A cell is observed if at least half of it has data, and water if at least 30% of the observed part is water, the rules every source uses.

**Scored on a few events.** The events were fixed by rule before running: the two with the most gold flood in each continent inside its coverage. That gives 8 events, 51 window map days and 161,934 gold flooded cell-days, 70% of them in Bangladesh (`scripts/fusion/experiment_cygnss.py`). Recall · precision (F1), gold-map days:

| product | all cells | normally-wet cells | other cells | share of gold flood seen |
| --- | --- | --- | --- | --- |
| CYGNSS | 0.67 · 0.37 (0.47) | 0.81 · 0.74 (0.77) | 0.58 · 0.24 (0.34) | 100% |
| fusion (as built) | 0.47 · 0.53 (0.50) | 0.59 · 0.83 (0.69) | 0.39 · 0.38 (0.38) | 100% |
| cascade | 0.35 · 0.65 (0.46) | 0.29 · 0.89 (0.44) | 0.39 · 0.56 (0.46) | 100% |
| microwave vote | 0.40 · 0.45 (0.42) | 0.39 · 0.82 (0.53) | 0.40 · 0.34 (0.37) | 100% |
| FloodScan | 0.36 · 0.45 (0.40) | 0.36 · 0.81 (0.49) | 0.36 · 0.34 (0.35) | 100% |
| GFM | 0.11 · 0.88 (0.19) | 0.07 · 0.93 (0.12) | 0.14 · 0.87 (0.23) | 54% |
| VIIRS | 0.11 · 0.61 (0.18) | 0.10 · 0.86 (0.17) | 0.12 · 0.52 (0.19) | 28% |

F1 by event:

| event | gold flooded cell-days | CYGNSS | fusion (as built) | cascade | microwave vote | FloodScan | GFM | VIIRS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EMSR342 (Australia 2019) | 9,503 | 0.57 | 0.57 | 0.73 | 0.52 | 0.54 | 0.00 | 0.74 |
| EMSR637 (Australia 2022) | 2,236 | 0.40 | 0.16 | 0.40 | 0.41 | 0.33 | 0.22 | 0.00 |
| EMSR865 (Colombia 2026) | 791 | 0.22 | 0.32 | 0.44 | 0.28 | 0.28 | 0.29 | 0.25 |
| FL20200713BGD (Bangladesh 2020) | 80,274 | 0.48 | 0.56 | 0.42 | 0.41 | 0.39 | 0.25 | 0.17 |
| FL20220525BGD (Bangladesh 2022) | 33,435 | 0.66 | 0.63 | 0.62 | 0.59 | 0.54 | 0.19 | 0.00 |
| FL20230321PER (Peru 2023) | 1,060 | 0.37 | 0.35 | 0.27 | 0.27 | 0.13 | 0.00 | 0.00 |
| TC20190312MOZ (Idai, Mozambique) | 16,581 | 0.34 | 0.13 | 0.39 | 0.37 | 0.35 | 0.19 | 0.11 |
| TC20190324ZWE (Idai, Zimbabwe) | 18,054 | 0.09 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 |

**Reading.**

- **CYGNSS has the highest F1 of any single product on these events** (0.47; next is the microwave vote at 0.42) and the highest recall of anything (0.67), and it sees almost all of the gold flood.
- **Its precision on ordinary cells is low (0.24).** It maps water broadly, and a composite of several weeks keeps water that has already drained. Its strength is on normally-wet cells (F1 0.77), where gold counts water as flood about half the time (Water versus flood).
- **Event by event** it is the best on three events and level with the best on one, and weaker where VIIRS saw clearly (Australia 2019) and on Colombia 2026.
- **A first look only:** eight events, 70% of the flood in Bangladesh, no intervals.

**What it would take.** Score it on all 148 events in its coverage, about 590 map days to read (Next steps, 11). Then it could enter as a fourth input treated as a slowly updating layer, not as daily evidence. Or it could be the water source in a water-first design (Water versus flood), where a water mask is exactly what is wanted.

**On all 148 events (26 October).** The same scoring on every event inside CYGNSS's dates and latitudes, with intervals from resampling events (`experiment_cygnss.py --all`): 330,980 gold flooded cell-days. F1, all cells, recall · precision:

| product | all cells | normally-wet cells | other cells |
| --- | --- | --- | --- |
| CYGNSS | 0.73 · 0.23 (0.35, 0.27–0.40) | 0.82 · 0.62 (0.70) | 0.69 · 0.16 (0.26) |
| fusion (as built) | 0.41 · 0.49 (0.45, 0.33–0.51) | 0.52 · 0.78 (0.62) | 0.35 · 0.38 (0.37) |
| cascade | 0.42 · 0.55 (0.47, 0.41–0.54) | 0.32 · 0.81 (0.46) | 0.47 · 0.49 (0.48) |
| microwave vote | 0.39 · 0.38 (0.38) | 0.38 · 0.75 (0.50) | 0.40 · 0.30 (0.34) |
| FloodScan | 0.33 · 0.43 (0.37) | 0.33 · 0.77 (0.46) | 0.33 · 0.35 (0.34) |
| GFM | 0.14 · 0.89 (0.25) | 0.11 · 0.91 (0.20) | 0.16 · 0.88 (0.27) |
| VIIRS | 0.14 · 0.55 (0.23) | 0.11 · 0.71 (0.19) | 0.16 · 0.51 (0.25) |

The first look's 0.47 does not hold: over all 148 events CYGNSS scores 0.35, below FloodScan (0.37), with the highest recall of anything (0.73) and the lowest precision (0.23). It is the best product by far on normally-wet cells (0.70) and the worst on other cells (0.26). That is what a water mask looks like scored against flood maps: it finds the water, including all the water that is normally there. As a flood input in the current design it would add false alarms; as the water layer in a water-first design, or as a background "how wet is this cell these weeks" level, it is the strongest candidate we have.

## Not yet validated

| gap | why it matters | what would close it |
| --- | --- | --- |
| precision on ordinary days | all precision above is measured inside mapped floods, where about a tenth of labelled cells on map days are flooded (12% in open land); elsewhere it will be far lower | done 26 October (Results, Ordinary days): on the 30 in-scope sample blocks, fusion's implied precision is about 0.03. Sentinel-2 water as an input-independent referee done 27 October: 0.04–0.09 outside normally-wet land, 0.18 with FloodScan alone in the microwave slot |
| one dominant basin | four Bangladesh floods in one basin hold 40% of the gold flood, so pooled scores follow them | Results shows every score with and without that basin; mapped events from other large floodplains would reduce the weight |
| independence from GFM | many Copernicus maps are drawn from Sentinel-1, GFM's own satellite | partly done: Results scores every product on map days without Sentinel-1 maps (GFM falls to F1 0.12, fusion keeps 0.46); a gold source fully independent of Sentinel-1 is still missing |
| river gauges | an independent check on timing and on floods the analysts never mapped | not started; the best-documented gauge network with flood thresholds (USGS/NWS) is in North America, now out of scope. Candidates are in `docs/gauge_sources.md` |
| small floods | a flood covering under 30% of a 1 km cell is dry for gold and products alike | a finer grid, at much higher cost |
| GFM on seasonal floodplains | GFM's stricter water definition probably costs recall there | score GFM on normally-wet versus other cells |
| uncertainty | per-continent counts are small | intervals by resampling whole events are beside every score; Oceania (13 events) and South America (16) stay thin, and more mapped events would help |
| FloodScan outside Africa | the licence for the Asia-Australia and South America files is unconfirmed | check with whoever holds the AER agreement |
| seasonal water in VIIRS and FloodScan | whether they count it as flood decides how they compare with GFM on floodplains | compare each source's flood-call rate on normally-wet and other cells on ordinary days, month by month, using the built events and the global sample |

## Next steps

Steps 1–4 are done (15 October). Steps 5–7 extend the analysis, and each waits on something outside this work.

- [x] **1. Settle the two proposals:** the rule for days before and after each event, and the event-level rules (the 10% share, the ±3-day window, the ordinary-day period). Decided for you on 14 October, before any result existed (ADRs 0006 and 0007).
- [x] **2. Code changes**, each with its test and a short decision record (ADR), and `docs/methods.md` updated in the same commit:
  - [x] scope and event selection in `configs/fusion.yaml` (continents listed, no quotas or size limits)
  - [x] three land types instead of six strata
  - [x] the days-before-and-after rule
  - [x] five folds by basin, balanced on flood area
- [x] **3. Build the 51 new events.** All built (15 October).
- [x] **4. Re-run fitting and scoring** over the five folds on all 170 events: cell-level (all cells and observed cells, per continent and land type) and event-level. Done (15 October): results/cv/ in the repo, Results above, and `docs/final_report.md`.
- [ ] **5. MODIS in the optical vote.** The stake: where VIIRS has files it adds 0.08 F1 to fusion, and the VIIRS gap holds 28% of the gold flood. Needs MODIS over every event window, a Databricks job. It is blocked on an Earthdata login the job can use: either your personal token stored as a Databricks secret, or the team account fixed.
- [x] **6. Precision on ordinary days** from the global random sample, with the deferred check for false events away from floods (ADR 0007). Done 26 October (Results, Ordinary days).
- [ ] **7. Gauge test:** choose a gauge source inside the four continents.
- [x] **8. Normally-wet stratum.** Fit each source's rates separately for cells normally wet in that calendar month (JRC monthly recurrence), within each land type. False-alarm rates differ 3 to 11 times between the two (Results, Water versus flood). Cheap: the masks are cached for every map-day month; the other months of the 51 new events need about 195 more window-month downloads (minutes). Done 26 October: it scores worse (pooled F1 0.37 against 0.44), because gold counts water on normally-wet cells as flood about half the time; not adopted (Results, Separate rates for normally-wet cells).
- [x] **9. Beat the cascade.** Fusion as built only matches "VIIRS, else GFM, else microwave" on the reference score and loses to it basin-weighted. The direct test is to learn the evidence for each combination of reports (each of the 27 patterns of flood, dry or couldn't see from the three sources, per land type) on the training folds, instead of multiplying each source's evidence as if they were independent. This changes fixed decision 7 of the original brief, so it needs your yes. Success: above the cascade on the pooled and the basin-weighted scores, while keeping fusion's lower false-event rate. Done 16 October: no version passes (Results, Can fusion beat the cascade?).
- [ ] **10. Choose the operating point: catch floods or avoid false alarms.** Step 9 found (16 October) that no version beats the cascade on both F1 and false events. Fusion as built sits at one end (F1 0.44, false events on 34% of events); the cascade and the no-memory versions at the other (F1 0.46–0.49, false events 64%). Which end, or a point between, depends on how the product will be used: an alert that must not cry wolf, or a map that must not miss water. The points in between are now shown (Results, The trade-off, threshold by threshold); the choice is yours.
- [x] **11. CYGNSS on all events in its coverage.** On 8 events it has the best F1 of any single product (0.47) and the highest recall (0.67), but low precision on ordinary cells (Other approaches tried, CYGNSS). Done 26 October: over all 148 events F1 0.35 (0.27–0.40), recall 0.73, precision 0.23; best on normally-wet cells (0.70), worst elsewhere (0.26). Not a flood input for the current design; the candidate water layer for a water-first design (Other approaches tried, CYGNSS).
- [x] **12. Drop GFDS from the microwave vote?** GFDS flags 2.4% of ordinary cell-days against FloodScan's 0.1%, and FloodScan alone costs fusion only 0.01 F1 on map days. Test: fusion with FloodScan alone in the microwave slot on the random sample and on the days before and after events, beside the vote. GFDS's feed also ended on 14 September 2026. Done 26 October: fusion's ordinary-day flood calls fall from 1.06 to 0.34 per 100 cells for 0.01 F1 (Results, Dropping GFDS). Recommended; yours to confirm, as it changes fixed decision 2.
