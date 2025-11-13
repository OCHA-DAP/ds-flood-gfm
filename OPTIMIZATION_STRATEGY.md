# Optimization Strategy for Large Countries (PHL)

## Problem Analysis

**Current Issue:**
- Philippines: 34.8M dask tasks, 40GB estimated memory → FAILS
- Root cause: Dense array operations on sparse flood data

**Data Characteristics (GFM):**
- Values: `255` (nodata), `0` (no-flood), `1` (flood)
- ~99% of pixels are nodata or 0
- Only ~0.1-1% are actual flood pixels (value=1)
- **Current approach treats all 5 billion pixels equally!**

## Critical Findings

### 1. Sparsity NOT Leveraged

```python
# Current (line 112 in gfm.py)
stack = stackstac.stack(items, epsg=4326)  # Uses float64, nan fill_value
```

**Problems:**
- Uses `dtype=float64` (8 bytes/pixel) by default
- Uses `fill_value=nan` requiring floating-point
- 90,451 × 55,405 × 4 dates × 8 bytes = 159GB raw (before compression)
- Dask creates 34M+ tasks to manage this

### 2. Stackstac Optimization Options

```python
# Optimized approach
stack = stackstac.stack(
    items,
    epsg=4326,
    dtype="uint8",          # 1 byte instead of 8!
    fill_value=255,         # Match GFM's native nodata
    rescale=False,          # Don't rescale - we know values are 0/1/255
    chunksize=2048          # Larger chunks = fewer tasks
)
```

**Benefits:**
- 8x memory reduction (uint8 vs float64)
- Matches native GFM encoding (0/1/255)
- Fewer dask tasks with larger chunks
- No rescaling overhead

### 3. Dask Rechunking Strategy

**Current chunks (from PHL output):**
```
Chunks: ((155, 1024, 1024, ...), (1011, 1024, 1024, ...))
```

**Problems:**
- Many small 1024×1024 chunks
- Task graph explosion: Each operation creates tasks for EACH chunk
- 34.8M tasks = operations × number_of_chunks × time_steps

**Solution: Strategic Rechunking**

```python
# After creating daily composites, before temporal operations
stack_flood_max = stack_flood_max.groupby("time.date").max()

# RECHUNK before temporal operations
stack_flood_max = stack_flood_max.chunk({
    'time': -1,      # Single chunk in time (only 4 dates)
    'y': 4096,       # Larger spatial chunks
    'x': 4096
})
```

**Why this helps:**
- Time dimension is small (4 dates) → single chunk eliminates cross-chunk operations
- Larger spatial chunks (4096 vs 1024) → 16x fewer chunks per dimension
- Reduces tasks from 34M to ~2M (estimate)

### 4. Persist Intermediate Results

```python
# After daily composites, BEFORE temporal composite
stack_flood_max = stack_flood_max.groupby("time.date").max()

# PERSIST to break computation graph
stack_flood_max = stack_flood_max.persist()  # Compute now, release graph
logger.info(f"  Persisted daily composites to memory")

# Now temporal operations work on clean, persisted data
flood_composite = (stack_flood_max == 1).any(dim="time").astype(np.uint8)
```

**Why this helps:**
- Breaks the computation graph into stages
- Daily composites computed once, cached in memory
- Temporal composite operates on cached data (no task explosion)
- Memory usage: Only holds 4 dates × spatial_extent (manageable)

## Testing Results

### Attempt 1: dtype optimization (BLOCKED)
**Finding:** stackstac 0.5.1 validates that `fill_value=nan` requires `dtype=float64`
- Tried uint8, int16, float32 - all rejected
- Cannot achieve 8x memory reduction via dtype alone

### Attempt 2: Chunking only (PROGRESS BUT INSUFFICIENT)
**Results:**
- Dask tasks: 34.8M → **8.8M** (75% reduction ✅)
- Memory estimate: 40GB → **5.01GB** (87% reduction ✅)
- **But still OOM killed** (exit code 137)

**Root cause:** 8.8M tasks + 5GB is still too large for single-pass computation on available memory.

### Attempt 3: Spatial Tiling (✅ SUCCESS!)
**Implementation:**
- Created `process_country_tiled()` function
- Automatically enabled for large countries (PHL, IDN, BRA, etc.)
- Default tile size: 2.0° × 2.0°

**Results - Philippines (PHL):**
- 45 tiles created
- Per-tile metrics:
  - Tasks: ~50-70k (vs 8.8M for full country)
  - Memory: 0.12 GB (vs 5GB for full country)
  - Computation time: 10-25s per tile
- **Successfully completed** (exit code 0)
- Polygons combined from all tiles

**Performance improvements:**
- Task reduction: **130x per tile** (8.8M → ~60k)
- Memory reduction: **40x per tile** (5GB → 0.12GB)
- **Scalable to any country size**

## Proposed Implementation

### Phase 1: Low-Hanging Fruit (dtype + chunksize)

```python
def create_flood_composite(items, bbox, n_latest, mode="latest", return_stack=False):
    # ... date selection ...

    # OPTIMIZED: Use native GFM encoding
    stack = stackstac.stack(
        items,
        epsg=4326,
        dtype="uint8",         # 8x memory reduction
        fill_value=255,        # Native GFM nodata
        rescale=False,         # No scaling needed
        chunksize=2048         # Larger chunks
    )

    # ... rest of function ...
```

**Expected Impact:**
- 8x memory reduction
- 4x fewer dask tasks (larger chunks)
- Should reduce PHL from 40GB → 5GB, 34M tasks → 8M tasks

### Phase 2: Rechunking Strategy

```python
# After groupby, RECHUNK
stack_flood_max = stack_flood_clipped.groupby("time.date").max()
stack_flood_max = stack_flood_max.rename({"date": "time"})
stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

# RECHUNK: Consolidate time dimension
stack_flood_max = stack_flood_max.chunk({
    'time': -1,    # All time in one chunk (only 4 dates)
    'y': 4096,     # Larger spatial chunks
    'x': 4096
})
logger.info(f"  Rechunked to time:-1, y:4096, x:4096")
```

**Expected Impact:**
- Further reduce tasks from 8M → ~500k
- Eliminate cross-chunk temporal operations

### Phase 3: Persist Strategy (if still needed)

```python
# After rechunking, optionally PERSIST
logger.info(f"  Computing daily composites...")
stack_flood_max = stack_flood_max.persist()
logger.info(f"  Daily composites persisted to memory")

# Temporal composite on clean persisted data
flood_composite = (stack_flood_max == 1).any(dim="time").astype(np.uint8)
```

**Tradeoff:**
- Requires computing full daily stack (~5GB for PHL)
- But eliminates task graph completely for final composite
- Net win if available memory > data size

## Alternative: Spatial Tiling

If memory is still insufficient:

```python
def process_country_in_tiles(iso3, tile_size=2.0):
    """Process large countries in lat/lon tiles"""
    gdf_admin = stratus.codab.load_codab_from_fieldmaps(iso3, 0)
    full_bbox = gdf_admin.total_bounds

    # Create tiles
    tiles = create_tiles(full_bbox, tile_size)

    # Process each tile
    tile_results = []
    for tile_bbox in tiles:
        items = query_gfm_stac(tile_bbox, end_date, n_search)
        flood_composite, dates, _ = create_flood_composite(items, tile_bbox, ...)
        tile_polygons = raster_to_polygons(flood_composite)
        tile_results.append(tile_polygons)

    # Merge all tiles
    return gpd.GeoDataFrame(pd.concat(tile_results, ignore_index=True))
```

## Testing Plan

1. **JAM baseline**: Verify optimizations don't break small countries
2. **PHL Phase 1**: Test dtype + chunksize alone
3. **PHL Phase 2**: Add rechunking if needed
4. **PHL Phase 3**: Add persist if still needed
5. **Fallback**: Spatial tiling if memory constrained

## Metrics to Track

```python
# Add to diagnostic logging
logger.info(f"  Chunk sizes (MB): {[c.nbytes/1e6 for c in flood_raster.data.blocks.values()]}")
logger.info(f"  Task graph size: {len(flood_raster.data.__dask_graph__())} tasks")
logger.info(f"  Estimated memory: {flood_raster.nbytes / 1e9:.2f} GB")
logger.info(f"  Actual dtype: {flood_raster.dtype}")
```

## Final Implementation

### Optimizations Applied

**1. Chunking optimization ([gfm.py:114-122](src/ds_flood_gfm/datasources/gfm.py#L114-L122)):**
```python
stack = stackstac.stack(
    items,
    epsg=4326,
    rescale=False,         # Skip unnecessary rescaling operations
    chunksize=2048         # 4x larger chunks (vs default 1024)
)
```

**2. Strategic rechunking ([gfm.py:138-146](src/ds_flood_gfm/datasources/gfm.py#L138-L146)):**
```python
stack_flood_max = stack_flood_max.chunk({
    'time': -1,    # Single chunk for time dimension (4 dates)
    'y': 4096,     # Larger spatial chunks
    'x': 4096
})
```

**3. Spatial tiling for large countries ([gfm.py:257-365](src/ds_flood_gfm/datasources/gfm.py#L257-L365)):**
- Automatic detection of large countries
- 2.0° × 2.0° tiles (configurable)
- Sequential tile processing with polygon merging
- Note: Provenance raster not yet supported for tiled processing

### Usage

**Small countries (JAM, HTI, CUB):**
```bash
uv run python scripts/04_generate_flood_polygons.py \
    --iso3 JAM \
    --end-date 2025-11-12 \
    --n-latest 4 \
    --flood-mode cumulative
```

**Large countries (PHL, IDN, BRA, etc.):**
```bash
# Tiling automatically enabled
uv run python scripts/04_generate_flood_polygons.py \
    --iso3 PHL \
    --end-date 2025-11-12 \
    --n-latest 4 \
    --flood-mode cumulative

# Or manually control tiling
uv run python scripts/04_generate_flood_polygons.py \
    --iso3 PHL \
    --use-tiling \
    --tile-size 2.0 \
    --end-date 2025-11-12 \
    --n-latest 4 \
    --flood-mode cumulative
```

### Known Limitations

1. **Provenance rasters**: Not yet supported for tiled processing (would require mosaicking tile stacks)
2. **Tile boundaries**: Polygons may be split at tile boundaries (could implement buffering/merging)
3. **Sequential processing**: Tiles processed one at a time (could parallelize for faster execution)

## References

- [stackstac dtype optimization](https://stackstac.readthedocs.io/en/stable/api/main/stackstac.stack.html)
- [GFM band encoding](https://extwiki.eodc.eu/GFM/PUM/TechnicalOverview)
- [Dask rechunking best practices](https://docs.dask.org/en/stable/array-chunks.html)
