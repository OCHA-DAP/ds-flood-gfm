"""The global 0.01-degree target grid (EPSG:4326).

Cell (row, col) covers lon [-180 + col*res, -180 + (col+1)*res) and
lat (90 - (row+1)*res, 90 - row*res]. Rows grow southward. All index maths is
integer so that windows computed for different inputs line up exactly.

GFDS's 0.09-degree pixels nest exactly 9x9 in this grid: GFDS pixel k's left
edge is -180.05 + 0.09k = -180 + 0.01*(9k - 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine

RES = 0.01
NCOLS = 36000
NROWS = 18000
_EPS = 1e-9


@dataclass(frozen=True)
class Window:
    """Half-open index window [row0, row1) x [col0, col1) on the global grid."""

    row0: int
    row1: int
    col0: int
    col1: int

    def __post_init__(self) -> None:
        if not (0 <= self.row0 < self.row1 <= NROWS and 0 <= self.col0 < self.col1 <= NCOLS):
            raise ValueError(f"window out of grid bounds: {self}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row1 - self.row0, self.col1 - self.col0)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in degrees."""
        return (
            -180.0 + self.col0 * RES,
            90.0 - self.row1 * RES,
            -180.0 + self.col1 * RES,
            90.0 - self.row0 * RES,
        )

    @property
    def transform(self) -> Affine:
        return Affine(RES, 0.0, -180.0 + self.col0 * RES, 0.0, -RES, 90.0 - self.row0 * RES)

    def lon(self) -> np.ndarray:
        """Cell-centre longitudes, length = ncols."""
        return -180.0 + (np.arange(self.col0, self.col1) + 0.5) * RES

    def lat(self) -> np.ndarray:
        """Cell-centre latitudes, length = nrows (north to south)."""
        return 90.0 - (np.arange(self.row0, self.row1) + 0.5) * RES

    def contains(self, other: Window) -> bool:
        return (
            self.row0 <= other.row0
            and other.row1 <= self.row1
            and self.col0 <= other.col0
            and other.col1 <= self.col1
        )

    def slices_of(self, inner: Window) -> tuple[slice, slice]:
        """Array slices of `inner` inside an array laid out on `self`."""
        if not self.contains(inner):
            raise ValueError(f"{inner} is not inside {self}")
        return (
            slice(inner.row0 - self.row0, inner.row1 - self.row0),
            slice(inner.col0 - self.col0, inner.col1 - self.col0),
        )

    def key(self) -> str:
        return f"r{self.row0}-{self.row1}_c{self.col0}-{self.col1}"


def col_of(lon: np.ndarray | float) -> np.ndarray:
    """Column index of the cell containing longitude `lon` (floor)."""
    return np.floor((np.asarray(lon, dtype="float64") + 180.0) / RES + _EPS).astype("int64")


def row_of(lat: np.ndarray | float) -> np.ndarray:
    """Row index of the cell containing latitude `lat` (floor from the north)."""
    return np.floor((90.0 - np.asarray(lat, dtype="float64")) / RES + _EPS).astype("int64")


def window_from_bounds(minx: float, miny: float, maxx: float, maxy: float) -> Window:
    """Smallest window whose cells cover the bounds (snapped outward), clipped to the globe."""
    if not (minx < maxx and miny < maxy):
        raise ValueError(f"degenerate bounds: {(minx, miny, maxx, maxy)}")
    col0 = max(int(math.floor((minx + 180.0) / RES + _EPS)), 0)
    col1 = min(int(math.ceil((maxx + 180.0) / RES - _EPS)), NCOLS)
    row0 = max(int(math.floor((90.0 - maxy) / RES + _EPS)), 0)
    row1 = min(int(math.ceil((90.0 - miny) / RES - _EPS)), NROWS)
    return Window(row0, row1, col0, col1)


def tiles_for_window(win: Window, tile_deg: int) -> list[Window]:
    """Processing tiles (tile_deg x tile_deg) that intersect `win`, each clipped to `win`."""
    n = round(tile_deg / RES)
    out = []
    for tr in range(win.row0 // n, (win.row1 - 1) // n + 1):
        for tc in range(win.col0 // n, (win.col1 - 1) // n + 1):
            r0, r1 = max(tr * n, win.row0), min((tr + 1) * n, win.row1)
            c0, c1 = max(tc * n, win.col0), min((tc + 1) * n, win.col1)
            out.append(Window(r0, r1, c0, c1))
    return out


def tile_window(tile_row: int, tile_col: int, tile_deg: int) -> Window:
    n = round(tile_deg / RES)
    return Window(tile_row * n, (tile_row + 1) * n, tile_col * n, (tile_col + 1) * n)


def tile_name(tile_row: int, tile_col: int, tile_deg: int) -> str:
    """Name a processing tile by its top-left corner, e.g. N10E030."""
    top = 90 - tile_row * tile_deg
    left = -180 + tile_col * tile_deg
    return f"{'N' if top >= 0 else 'S'}{abs(top):02d}{'E' if left >= 0 else 'W'}{abs(left):03d}"


def cell_index(lon: np.ndarray, lat: np.ndarray, win: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local (row, col) of points inside `win`, plus a boolean mask of which points fall inside."""
    r = row_of(lat) - win.row0
    c = col_of(lon) - win.col0
    inside = (r >= 0) & (r < win.shape[0]) & (c >= 0) & (c < win.shape[1])
    return r, c, inside


def bincount_2d(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], weights: np.ndarray | None = None) -> np.ndarray:
    """Sum `weights` (or count points) per cell. Inputs must already be inside `shape`."""
    flat = rows.astype("int64") * shape[1] + cols.astype("int64")
    return np.bincount(flat, weights=weights, minlength=shape[0] * shape[1]).reshape(shape)
