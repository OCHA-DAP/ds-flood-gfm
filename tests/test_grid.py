"""Grid maths. Every adapter relies on these indices lining up exactly.

Why these cases:
- window_from_bounds snaps outward and round-trips through bounds;
- cell_index maps cell centres back to their own (row, col);
- GFDS 0.09-degree pixels nest exactly 9x9 (the adapter relies on (i + 5) // 9);
- tiles_for_window partitions a window without overlap or gaps;
- gold burn_fraction returns exact area fractions for axis-aligned shapes.
"""

import numpy as np
import shapely

from ds_flood_gfm.fusion import gold, grid
from ds_flood_gfm.fusion.sources.gfds import gfds_cols, gfds_rows


def test_window_round_trip_and_snap():
    w = grid.window_from_bounds(6.505, 7.6, 6.9, 8.0)
    minx, miny, maxx, maxy = w.bounds
    assert np.isclose(minx, 6.50) and np.isclose(maxx, 6.90) and np.isclose(miny, 7.60) and np.isclose(maxy, 8.00)
    assert w.shape == (40, 40)


def test_cell_index_of_centres():
    w = grid.Window(100, 110, 200, 215)
    lon2, lat2 = np.meshgrid(w.lon(), w.lat())
    r, c, inside = grid.cell_index(lon2, lat2, w)
    assert inside.all()
    assert (r == np.arange(10)[:, None]).all() and (c == np.arange(15)[None, :]).all()


def test_gfds_nesting_9x9():
    # GFDS pixel k spans lon [-180.05 + 0.09k, -180.05 + 0.09(k+1)); target cell j centre at -180 + 0.01(j + 0.5)
    w = grid.Window(0, 18000, 0, 36000)
    cols = gfds_cols(w)
    lon_c = w.lon()
    k = np.floor((lon_c + 180.05) / 0.09).astype(int) % 4000
    assert (cols == k).all()
    rows = gfds_rows(w)
    lat_c = w.lat()
    kr = np.floor((90.05 - lat_c) / 0.09).astype(int)
    assert (rows == kr).all()
    # each GFDS column holds exactly 9 target columns (except the split one at the antimeridian)
    counts = np.bincount(cols)
    assert (counts[1:-1] == 9).all()


def test_tiles_partition_window():
    w = grid.window_from_bounds(-3.3, 41.2, 2.7, 45.9)
    tiles = grid.tiles_for_window(w, 1)
    cover = np.zeros(w.shape, int)
    for t in tiles:
        rs, cs = w.slices_of(t)
        cover[rs, cs] += 1
    assert (cover == 1).all()


def test_burn_fraction_exact_for_aligned_box():
    w = grid.Window(5000, 5010, 18000, 18010)
    minx, miny, maxx, maxy = w.bounds
    # covers exactly the left half of every cell in the first row
    box = shapely.box(minx, maxy - 0.01, minx + 0.005, maxy)
    f = gold.burn_fraction(box, w, 0.001)
    assert np.isclose(f[0, 0], 0.5) and np.isclose(f[1:, :].sum(), 0.0) and np.isclose(f[0, 1:].sum(), 0.0)


def test_polygonal_drops_lines_and_z():
    geom = shapely.GeometryCollection([shapely.LineString([(0, 0, 5), (1, 1, 5)]), shapely.box(0, 0, 1, 1)])
    p = gold._polygonal(shapely.to_wkb(geom))
    assert p.geom_type == "Polygon" and not shapely.has_z(p)
