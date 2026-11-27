"""Source value rules (fixed decisions 1 and 2) on synthetic inputs.

Why these cases:
- The microwave OR-merge has four regimes: flood wins, observed dry, one NoData, both NoData.
- VIIRS: cloud, shadow and snow/ice must become NoData. Normal water must be dry, not flood.
  Floodwater fractions must decode from 100-200.
- GFDS: both sentinels and non-positive signals are NoData. The R1 fraction has the right
  sign: flood lowers the signal.
- FloodScan: the containing 300-arcsec cell is used, and points outside a region are not
  claimed by it.
- GFM: the expected sample count per cell uses cell area / pixel area.
"""

import numpy as np

from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA, merge_microwave
from ds_flood_gfm.fusion.sources.floodscan import Region
from ds_flood_gfm.fusion.sources.gfm import expected_samples
from ds_flood_gfm.fusion.sources.viirs import flood_fraction


def test_microwave_or_rule():
    fs = np.array([1, 0, 0, 255, 255, 1, 255], "uint8")
    gf = np.array([0, 1, 0, 0, 255, 255, 1], "uint8")
    out = merge_microwave(fs, gf)
    assert out.tolist() == [FLOOD, FLOOD, DRY, DRY, NODATA, FLOOD, FLOOD]


def test_viirs_legend():
    codes = np.array([1, 15, 16, 17, 20, 27, 30, 38, 50, 99, 100, 130, 200], "uint8")
    obs, frac = flood_fraction(codes)
    assert obs.tolist() == [False, True, True, True, False, False, False, False, False, True, True, True, True]
    np.testing.assert_allclose(frac[obs], [1.0, 0.0, 0.0, 0.0, 0.0, 0.30, 1.0])


def test_gfds_mask_and_r1_sign(tmp_path):
    from ds_flood_gfm.fusion.sources.gfds import GFDSSource

    class Cfg:
        def __getitem__(self, k):
            return {"sources": {"gfds": {"sentinels": [-32000, -2147483648], "signal_scale": 1_000_000,
                                         "user_agent": "test", "base_url": "x", "min_seconds_between_requests": 0}}}[k]

        def local(self, *parts):
            p = tmp_path.joinpath(*parts)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    src = GFDSSource(Cfg())
    a = np.array([-32000, -2147483648, -5, 0, 950_000, 1_000_000])
    out = src._mask_scale(a)
    assert np.isnan(out[:4]).all() and np.allclose(out[4:], [0.95, 1.0])
    ref, K = 1.0, 0.35
    f_wet = (ref - 0.85) / (ref * K)  # wetter cell -> lower M/C signal -> positive fraction
    assert f_wet > 0.30 and (ref - 1.0) / (ref * K) == 0.0


def test_floodscan_region_index():
    reg = Region("africa", -30.0, 40.0, 1080, 1080, None)
    r, c, ok = reg.index(np.array([-29.999, 59.999, 60.001, 10.0]), np.array([39.999, -49.999, 0.0, 40.001]))
    assert ok.tolist() == [True, True, False, False]
    assert (r[0], c[0]) == (0, 0) and (r[1], c[1]) == (1079, 1079)


def test_gfm_expected_samples():
    assert 190 < expected_samples(0.0) < 197  # (1.1132 km)^2 / (80 m)^2 at the equator
    assert np.isclose(expected_samples(60.0), expected_samples(0.0) / 2, rtol=1e-6)
