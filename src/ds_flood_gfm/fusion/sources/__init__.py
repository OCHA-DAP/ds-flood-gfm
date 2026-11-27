"""Source adapters. Each turns one product into daily rasters on the 0.01-degree grid.

Value rule for every adapter (fixed decision 1):
    1 = observed flood, 0 = observed dry, 255 = could not observe.
NoData comes only from the product's own masks (nodata, swath, exclusion,
sentinel values) or from the product not covering the place/day.

Three states are never conflated:
    upstream absence (no scene / no file for the day)  -> 255 for that day, recorded
    fetch failure (network / server error)            -> raises with the resource named
    real dry (observed, below threshold)              -> 0
"""

import numpy as np

FLOOD, DRY, NODATA = 1, 0, 255


def merge_microwave(floodscan: np.ndarray, gfds: np.ndarray) -> np.ndarray:
    """Fixed decision 2: flood if either says flood; dry if at least one observed and
    neither says flood; else NoData."""
    if floodscan.shape != gfds.shape:
        raise ValueError(f"shape mismatch {floodscan.shape} vs {gfds.shape}")
    out = np.full(floodscan.shape, NODATA, "uint8")
    observed = (floodscan != NODATA) | (gfds != NODATA)
    out[observed] = DRY
    out[(floodscan == FLOOD) | (gfds == FLOOD)] = FLOOD
    return out
