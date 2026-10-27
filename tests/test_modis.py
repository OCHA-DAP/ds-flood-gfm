"""Tests for the MODIS (MCDWD) adapter on synthetic tiles, with no network.

Why these cases:
- Tile lookup decides which files are downloaded. A window on a tile edge must fetch both tiles,
  and a point must fetch exactly one (h18v08 holds Lokoja).
- The flag mapping is the value rule. "Couldn't see" (255) must never become dry, and normal water
  (1) must never become flood. The recurring (seasonal) class must follow `recurring_as`.
- A 0.01-degree cell holds 4.8 x 4.8 MODIS pixels, so the pixel-to-cell assignment is not
  one-to-one. A synthetic tile with one class per cell column checks that every column lands
  on the right value, including below-threshold mixed cells and an unobserved column.
- A window crossing a tile edge must sum both tiles without double-counting pixels.
- Fail loudly: a missing token is a misconfiguration, and an HTML login page instead of HDF is a
  rejected token. Neither may pass silently as "no data".
"""

import numpy as np
import pytest

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import load_config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA
from ds_flood_gfm.fusion.sources import modis

PIX = modis.PIX


def test_tiles_for_bounds():
    assert modis.tiles_for_bounds(6.5, 7.5, 7.0, 8.0) == [(18, 8)]
    assert modis.tiles_for_bounds(9.5, 9.5, 10.5, 10.5) == [(18, 7), (19, 7), (18, 8), (19, 8)]


def test_flood_codes_mapping():
    codes = np.array([0, 1, 2, 3, 255], "uint8")
    obs, fl = modis.flood_codes(codes, "flood")
    assert obs.tolist() == [True, True, True, True, False]
    assert fl.tolist() == [False, False, True, True, False]
    _, fl_dry = modis.flood_codes(codes, "dry")
    assert fl_dry.tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError):
        modis.flood_codes(codes, "maybe")


def _tile_by_centre(fill):
    """4800 x 4800 tile (h18v08: lon 0..10, lat 10..0) with codes set from each pixel's centre."""
    lon = (np.arange(modis.TILE_PX) + 0.5) * PIX
    lat = 10 - (np.arange(modis.TILE_PX) + 0.5) * PIX
    return fill(lon[None, :], lat[:, None], np.arange(modis.TILE_PX)[:, None])


def test_cell_values_from_one_tile():
    def fill(lon, lat, i):
        c = np.zeros((modis.TILE_PX, modis.TILE_PX), "uint8")
        c[:, ((lon >= 5.00) & (lon < 5.01))[0]] = 3                  # col 0: unusual flood
        c[:, ((lon >= 5.01) & (lon < 5.02))[0]] = 255                # col 1: could not observe
        c[:, ((lon >= 5.02) & (lon < 5.03))[0]] = 2                  # col 2: recurring flood
        c[:, ((lon >= 5.03) & (lon < 5.04))[0]] = 1                  # col 3: normal water
        mixed = ((lon >= 5.04) & (lon < 5.05))[0]                    # col 4: 1 row in 5 flooded (20% < 30%)
        c[np.ix_((i[:, 0] % 5) == 0, mixed)] = 3
        return c

    win = grid.window_from_bounds(5.0, 5.0, 5.05, 5.03)
    counts = modis.tile_counts(_tile_by_centre(fill), 18, 8, win)
    v_fl, f_fl = modis.values_from_counts(counts, "flood", 0.5, 0.30)
    v_dry, _ = modis.values_from_counts(counts, "dry", 0.5, 0.30)
    assert (v_fl[:, 0] == FLOOD).all() and (v_fl[:, 1] == NODATA).all()
    assert (v_fl[:, 2] == FLOOD).all() and (v_dry[:, 2] == DRY).all()
    assert (v_fl[:, 3] == DRY).all() and (v_fl[:, 4] == DRY).all()
    assert np.isnan(f_fl[:, 1]).all() and (f_fl[:, 4] < 0.30).all()


def test_window_across_a_tile_edge_sums_both_tiles_once():
    win = grid.window_from_bounds(9.98, 5.0, 10.02, 5.01)
    left = modis.tile_counts(np.full((modis.TILE_PX, modis.TILE_PX), 3, "uint8"), 18, 8, win)   # lon 0..10: flood
    right = modis.tile_counts(np.zeros((modis.TILE_PX, modis.TILE_PX), "uint8"), 19, 8, win)    # lon 10..20: dry
    total = {k: left[k] + right[k] for k in left}
    v, _ = modis.values_from_counts(total, "flood", 0.5, 0.30)
    assert v.tolist() == [[FLOOD, FLOOD, DRY, DRY]]
    assert ((left["n_all"] > 0) & (right["n_all"] > 0)).sum() == 0   # no cell takes pixels from both
    expected = round(0.04 / PIX) * round(0.01 / PIX)
    assert abs(total["n_all"].sum() - expected) <= 5                  # every pixel counted once


def test_payload_check_separates_rejected_token_from_corrupt_file():
    modis.check_hdf_payload(modis.HDF4_MAGIC + b"rest", "u")
    with pytest.raises(PermissionError):
        modis.check_hdf_payload(b"<!DOCTYPE html><html>", "u")
    with pytest.raises(ValueError):
        modis.check_hdf_payload(b"\x00\x01\x02\x03garbage", "u")


def test_missing_token_is_a_misconfiguration(monkeypatch):
    cfg = load_config()
    for k in ("token_env", "login_user_env", "login_password_env"):
        monkeypatch.delenv(cfg["sources"]["modis"][k], raising=False)
    with pytest.raises(RuntimeError, match="misconfigured"):
        modis.MODISFloodSource(cfg)
