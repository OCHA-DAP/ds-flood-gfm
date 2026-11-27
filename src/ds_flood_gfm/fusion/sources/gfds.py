"""JRC/GDACS Global Flood Detection System (GFDS) on the 0.01-degree grid.

Daily `signal` GeoTIFF (0.09 degrees, int32 x 1e6, sentinels -32000 and
int32-min) is turned into a flooded fraction with the R1 physics route from
experiments/gfds_downscaling_poc/06_independent_fractions.py, using GFDS's
own static baseline mean as the dry reference (no FloodScan involved):

    f = (s_ref - s) / (s_ref * K),   K = 1 - eps_water / eps_land ~ 0.35
    flood iff f >= flood_threshold (0.30, same rule as FloodScan)

NoData = sentinel / non-positive signal or baseline (the product's own
masks). GFDS pixels nest exactly 9x9 in the target grid:
0.01-degree row i lies in GFDS row (i + 5) // 9, likewise for columns.

The server answers HTTP 403 (not 404) for days it does not have. A 403 is
recorded as upstream absence only after a known-good file is confirmed
readable in the same minute; otherwise it raises (we may be blocked).
Requests are spaced >= min_seconds_between_requests apart.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window as RWindow

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA

NCOLS_G, NROWS_G = 4000, 2000
KNOWN_GOOD_DAY = dt.date(2024, 9, 15)
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="10",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_TIMEOUT="180",
)


def gfds_rows(win: grid.Window) -> np.ndarray:
    return (np.arange(win.row0, win.row1) + 5) // 9


def gfds_cols(win: grid.Window) -> np.ndarray:
    return ((np.arange(win.col0, win.col1) + 5) // 9) % NCOLS_G


class GFDSAbsent(Exception):
    pass


def wait_for_store(store_dir, windows: list[grid.Window], days: list[dt.date], timeout_s: float, poll_s: float = 60.0) -> None:
    """Block until every day file exists and covers the windows' GFDS rows (or is an absent marker).

    Used when a builder runs alongside the (slow, rate-limited) prefetch. Raises TimeoutError.
    """
    need = set()
    for w in windows:
        need.update(gfds_rows(w).tolist())
    covered: set = set()
    t0 = time.time()
    while True:
        for d in days:
            if d in covered:
                continue
            p = store_dir / f"{d:%Y%m%d}.npz"
            if p.exists():
                with np.load(p) as z:
                    if "absent" in z.files or need <= set(z["rows"].tolist()):
                        covered.add(d)
        missing = [d for d in days if d not in covered]
        if not missing:
            return
        if time.time() - t0 >= timeout_s:
            raise TimeoutError(f"GFDS store lacks {len(missing)} days (first {missing[0]}) after {timeout_s / 3600:.1f} h")
        time.sleep(poll_s)


class GFDSSource:
    name = "gfds"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.s = cfg["sources"]["gfds"]
        self._lock = threading.Lock()
        self._local = threading.local()  # per-connection request spacing
        self._last_known_good_check = 0.0
        self._ref: np.ndarray | None = None
        self.store_dir = cfg.local("gfds", "days", "x").parent
        self.gdal_env = dict(GDAL_ENV, GDAL_HTTP_USERAGENT=self.s["user_agent"])

    # ------------------------------------------------------------------ urls / http
    def url(self, day: dt.date) -> str:
        return f"{self.s['base_url']}/ALL/SignalTiffs/{day:%Y}/{day:%m}/signal_{day:%Y%m%d}_ALL.tif"

    def _throttle(self) -> None:
        """Space requests >= min_seconds_between_requests apart on each connection (thread);
        at most max_concurrent_requests connections run, so the server sees <= that many req/s."""
        last = getattr(self._local, "last", 0.0)
        wait = self.s["min_seconds_between_requests"] - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._local.last = time.monotonic()

    def _mask_scale(self, a: np.ndarray) -> np.ndarray:
        out = a.astype("float64")
        for s in self.s["sentinels"]:
            out[a == s] = np.nan
        out /= self.s["signal_scale"]
        out[out <= 0] = np.nan
        return out

    def _confirm_server_alive(self) -> None:
        with self._lock:
            self._confirm_server_alive_locked()

    def _confirm_server_alive_locked(self) -> None:
        if time.monotonic() - self._last_known_good_check < 60:
            return
        self._throttle()
        req = urllib.request.Request(self.url(KNOWN_GOOD_DAY), method="HEAD", headers={"User-Agent": self.s["user_agent"]})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 200:
                    raise RuntimeError(f"GFDS known-good file returned {r.status}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GFDS known-good file {self.url(KNOWN_GOOD_DAY)} returned HTTP {e.code}: server refusing us, not an absent day") from e
        self._last_known_good_check = time.monotonic()

    def reference(self) -> np.ndarray:
        """GFDS static baseline mean signal (global, float64, NaN where invalid)."""
        if self._ref is None:
            local = self.cfg.local("raw", "gfds", "bt_signal_avg.tif")
            if not local.exists():
                self._throttle()
                url = f"{self.s['base_url']}/{self.s['baseline_avg']}"
                req = urllib.request.Request(url, headers={"User-Agent": self.s["user_agent"]})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                tmp = local.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.rename(local)
            with rasterio.open(local) as ds:
                a = ds.read(1)
                if a.shape != (NROWS_G, NCOLS_G):
                    raise ValueError(f"unexpected GFDS baseline shape {a.shape}")
            self._ref = self._mask_scale(a)
        return self._ref

    # ------------------------------------------------------------------ day store
    def _day_path(self, day: dt.date):
        return self.store_dir / f"{day:%Y%m%d}.npz"

    def _read_bands(self, day: dt.date, bands: list[tuple[int, int]]) -> dict[str, np.ndarray]:
        out = {}
        try:
            with rasterio.Env(**self.gdal_env):
                self._throttle()
                with rasterio.open("/vsicurl/" + self.url(day)) as ds:
                    if ds.shape != (NROWS_G, NCOLS_G):
                        raise ValueError(f"unexpected GFDS grid {ds.shape} for {day}")
                    for k0, k1 in bands:
                        self._throttle()
                        out[f"{k0}_{k1}"] = ds.read(1, window=RWindow(0, k0, NCOLS_G, k1 - k0))
        except RasterioIOError as e:
            msg = str(e)
            if "403" in msg or "404" in msg or "not recognized" in msg or "does not exist" in msg:
                self._confirm_server_alive()
                raise GFDSAbsent(f"GFDS has no signal file for {day}") from e
            raise
        return out

    def _prefetch_day(self, day: dt.date, windows: list[grid.Window]) -> str:
        need = set()
        for w in windows:
            rows = gfds_rows(w)
            need.update(range(int(rows.min()), int(rows.max()) + 1))
        path = self._day_path(day)
        if path.exists():
            with np.load(path) as z:
                if "absent" in z.files:
                    return "skipped"
                have = set(z["rows"].tolist())
                if need <= have:
                    return "skipped"
                need |= have
        rows = sorted(need)
        gap = self.s["band_merge_gap_rows"]
        bands, start = [], rows[0]
        for a, b in zip(rows, rows[1:] + [None]):
            if b is None or b > a + 1 + gap:
                bands.append((start, a + 1))
                start = b
        tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.part.npz")
        from ds_flood_gfm.fusion.net import retry

        try:
            got = retry(lambda: self._read_bands(day, bands), f"GFDS {day}")
        except GFDSAbsent:
            np.savez_compressed(tmp, absent=np.array(True))
            tmp.rename(path)
            return "absent"
        full_rows = np.concatenate([np.arange(k0, k1) for k0, k1 in bands])
        data = np.concatenate([got[f"{k0}_{k1}"] for k0, k1 in bands], axis=0)
        np.savez_compressed(tmp, rows=full_rows, data=data)
        tmp.rename(path)
        return "fetched"

    def prefetch(self, day_windows: dict[dt.date, list[grid.Window]], order: list[dt.date] | None = None) -> dict[str, int]:
        """Fetch, for each day, the GFDS row bands covering every window active that day.

        Idempotent per day (a day already covering the rows is skipped; absent days are
        stored as an explicit marker). Days run on up to max_concurrent_requests connections.
        """
        from concurrent.futures import ThreadPoolExecutor

        days = order if order is not None else sorted(day_windows)
        stats = {"fetched": 0, "skipped": 0, "absent": 0}
        with ThreadPoolExecutor(self.s["max_concurrent_requests"]) as ex:
            for r in ex.map(lambda d: self._prefetch_day(d, day_windows[d]), days):
                stats[r] += 1
        return stats

    def observe_days(self, win: grid.Window, days: list[dt.date]) -> tuple[np.ndarray, np.ndarray, list[dt.date]]:
        """(values 1/0/255, R1 fraction float32, list of absent days) from the prefetched store.

        A day missing from the store raises: prefetch must run first.
        """
        ref = self.reference()
        gr, gc = gfds_rows(win), gfds_cols(win)
        refw = ref[np.ix_(gr, gc)]
        K, thr = self.s["k_emissivity"], self.s["flood_threshold"]
        out = np.full((len(days), *win.shape), NODATA, "uint8")
        frac = np.full((len(days), *win.shape), np.nan, "float32")
        absent = []
        for i, d in enumerate(days):
            path = self._day_path(d)
            if not path.exists():
                raise FileNotFoundError(f"GFDS day {d} not prefetched ({path})")
            with np.load(path) as z:
                if "absent" in z.files:
                    absent.append(d)
                    continue
                rows, data = z["rows"], z["data"]
            pos = np.searchsorted(rows, gr)
            if (pos >= len(rows)).any() or (rows[np.minimum(pos, len(rows) - 1)] != gr).any():
                raise ValueError(f"GFDS store for {d} lacks rows for window {win.key()}")
            sig = self._mask_scale(data[np.ix_(pos, gc)])
            with np.errstate(invalid="ignore", divide="ignore"):
                f = (refw - sig) / (refw * K)
            ok = np.isfinite(f)
            frac[i][ok] = f[ok]
            out[i][ok] = DRY
            out[i][ok & (f >= thr)] = FLOOD
        return out, frac, absent
