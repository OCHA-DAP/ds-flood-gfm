"""Tests for the daily job wrapper (run_window) on synthetic observations.

Why these cases:
- The job must reproduce engine.run_sequence exactly. Validation and the operational
  run share this code path, so any drift between them would invalidate Phase 5.
- Re-running is idempotent and resumable: completed days are skipped, not recomputed.
- A missing previous-day state must refuse to start unless cold_start is explicit
  (no silent re-initialisation of the carry-over).
- Non-consecutive days are rejected (the carry-over needs every day).
- Written COGs carry the four documented bands and NaN outside the domain.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml

from ds_flood_gfm.fusion import engine, grid, job
from ds_flood_gfm.fusion.config import DEFAULT_CONFIG, load_config

SOURCES = ["gfm", "microwave", "viirs"]


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"]["local_root"] = str(tmp_path / "local")
    p = tmp_path / "fusion.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


@pytest.fixture
def model():
    acc = pd.DataFrame([{"source": s, "stratum_code": k, "sens": 0.7, "spec": 0.95} for s in SOURCES for k in range(6)])
    prior = pd.DataFrame({"stratum_code": range(6), "base_rate": [0.02] * 6})
    return engine.Model.from_tables(acc, prior, SOURCES, 0.9)


class FakeProvider:
    def __init__(self, obs_seq, days):
        self.obs_seq, self.idx = obs_seq, {d: i for i, d in enumerate(days)}
        self.calls = 0

    def get(self, day):
        self.calls += 1
        i = self.idx[day]
        return {s: self.obs_seq[s][i] for s in SOURCES}, SOURCES


def _setup(rng, n_days=6):
    win = grid.Window(1000, 1010, 2000, 2012)
    days = [dt.date(2024, 9, 1) + dt.timedelta(days=i) for i in range(n_days)]
    obs = {s: rng.choice([0, 1, 255], size=(n_days, *win.shape), p=[0.5, 0.2, 0.3]).astype("uint8") for s in SOURCES}
    dom = np.ones(win.shape, bool)
    dom[0, 0] = False
    st = job.StaticForWindow(dom, {m: np.full(win.shape, 2, "uint8") for m in range(1, 13)})
    return win, days, obs, st


def test_job_matches_engine_and_is_idempotent(cfg, model):
    rng = np.random.default_rng(5)
    win, days, obs, st = _setup(rng)
    prov = FakeProvider(obs, days)
    s1 = job.run_window(cfg, "t", win, days, prov, model, st, cold_start=True)
    assert s1["written"] == len(days) and s1["start"] == "cold"
    ref = engine.run_sequence(obs, np.stack([st.strata_by_month[9]] * len(days)), st.domain, model)
    with rasterio.open(job.io.daily_cog_path(cfg, "t", days[-1])) as ds:
        assert [ds.descriptions[i] for i in range(4)] == ["p_flood", "abs_logodds", "n_sources_observed", "days_since_obs"]
        p = ds.read(1)
    np.testing.assert_allclose(p[st.domain], ref["p_flood"][-1][st.domain], rtol=1e-5)
    assert np.isnan(p[0, 0])
    calls = prov.calls
    s2 = job.run_window(cfg, "t", win, days, prov, model, st)
    assert s2["skipped"] == len(days) and prov.calls == calls  # nothing re-fetched


def test_resume_from_state_equals_full_run(cfg, model):
    rng = np.random.default_rng(6)
    win, days, obs, st = _setup(rng, n_days=8)
    job.run_window(cfg, "a", win, days[:4], FakeProvider(obs, days), model, st, cold_start=True)
    job.run_window(cfg, "a", win, days[4:], FakeProvider(obs, days), model, st)  # warm start from day 4 state
    job.run_window(cfg, "b", win, days, FakeProvider(obs, days), model, st, cold_start=True)
    with rasterio.open(job.io.daily_cog_path(cfg, "a", days[-1])) as a, rasterio.open(job.io.daily_cog_path(cfg, "b", days[-1])) as b:
        np.testing.assert_allclose(a.read(1), b.read(1), rtol=1e-5, equal_nan=True)


def test_missing_state_refuses_without_cold_start(cfg, model):
    rng = np.random.default_rng(7)
    win, days, obs, st = _setup(rng)
    with pytest.raises(FileNotFoundError, match="cold_start"):
        job.run_window(cfg, "c", win, days, FakeProvider(obs, days), model, st)


def test_non_consecutive_days_rejected(cfg, model):
    rng = np.random.default_rng(8)
    win, days, obs, st = _setup(rng)
    with pytest.raises(ValueError, match="consecutive"):
        job.run_window(cfg, "d", win, [days[0], days[2]], FakeProvider(obs, days), model, st, cold_start=True)
