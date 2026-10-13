"""Tests for the seasonal baseline (GSW monthly recurrence), synthetic, with no network.

Why these cases:
- The tile name decides which file is read. The Inner Niger Delta box must map to the tile that
  was read by hand on 2026-10-13 (0000240000-0000680000), and a box on a tile corner must read all four.
- The recurrence grid stops at 80 N. A window beyond it must raise, not read an empty array.
- The 40 x 40 block share is the whole baseline. Full, half, below-threshold and just-below-50%
  blocks must give 1.0, 0.5, 0.2 and 0.0.
"""

import numpy as np
import pytest

from ds_flood_gfm.fusion import baseline


def test_tile_lookup_matches_the_file_read_by_hand():
    tiles = baseline.tiles_for_bounds(-4.5, 14.5, -4.3, 14.7)
    assert tiles == [(-10.0, 20.0)]
    assert baseline.tile_name(*tiles[0]) == "0000240000-0000680000.tif"
    assert baseline.tiles_for_bounds(9.9, 9.9, 10.1, 10.1) == [(0.0, 20.0), (10.0, 20.0), (0.0, 10.0), (10.0, 10.0)]


def test_beyond_80_north_raises():
    with pytest.raises(ValueError):
        baseline.tiles_for_bounds(0, 79.5, 1, 80.5)


def test_wet_share_per_40x40_block():
    rec = np.zeros((80, 80), "int16")
    rec[:40, :40] = 80                      # cell (0,0): every pixel wet
    rec[:40, 40:60] = 60                    # cell (0,1): half the pixels wet
    rec[40:48, :40] = 50                    # cell (1,0): 8 of 40 rows wet (20%), 50 counts as wet
    rec[40:, 40:] = 49                      # cell (1,1): just below 50, never wet
    share = baseline.wet_share(rec, 50)
    np.testing.assert_allclose(share, [[1.0, 0.5], [0.2, 0.0]])
    assert ((share >= 0.30) == np.array([[True, True], [False, False]])).all()
