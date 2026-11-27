"""NASA LANCE MODIS flood product (MCDWD, Terra + Aqua, 250 m) on the 0.01-degree grid.

Archive (LAADS DAAC; scripted downloads need an Earthdata token, read from env `token_env`):
  {archive}/allData/61/{product}/YYYY/DOY/{product}.AYYYYDOY.hHHvVV.061.<production time>.hdf
  MCDWD_L3      reprocessed, 2000-2025 (complete through 2025-12-31, checked 2026-10-13)
  MCDWD_L3_NRT  near-real-time files archived from 2026-01-01 (not reprocessed), same layout
HDF4, read with pyhdf (the rasterio wheel's GDAL has no HDF4 driver). 10 x 10 degree tiles on a
geographic grid, 4800 x 4800 pixels of 0.0020833 deg; tile h,v has its upper-left corner at
(-180 + 10h, 90 - 10v) (user guide Rev D section 1; each file's StructMetadata: GCTP_GEO,
corner registration).

One file holds every composite: Flood_1Day_250m, FloodCS_1Day_250m (cloud shadows screened),
Flood_2Day_250m, Flood_3Day_250m. Flag values, from each layer's description attribute:
  255 insufficient data | 0 no water | 1 water matching the reference water |
  2 recurring flood (expected seasonal flood) | 3 flood (unusual)

Mapping to the value rule (fixed decision 1):
  NoData  : 255
  observed: 0, 1, 2, 3
  flood   : 3; and 2 when recurring_as == "flood" (else it counts as observed, not flooded).
            1 is normal water, not flood (as GFM's reference water).
Cell rule as for GFM and VIIRS: observed iff >= observed_min_frac of the cell's pixels are
observed; flood iff flood pixels / observed pixels >= flood_threshold. A 0.01-degree cell holds
4.8 x 4.8 pixels, so each pixel goes to the cell that contains its centre.

Three states are kept apart:
  - no file for a tile and day in the listing  -> upstream absence: NoData, recorded in `absent`
  - no token in the environment                 -> misconfigured: raises at construction
  - an HTML page or 401 instead of an HDF file  -> token rejected: raises
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ds_flood_gfm.fusion import grid
from ds_flood_gfm.fusion.config import Config
from ds_flood_gfm.fusion.sources import DRY, FLOOD, NODATA

TILE_DEG = 10
TILE_PX = 4800
PIX = TILE_DEG / TILE_PX
HDF4_MAGIC = b"\x0e\x03\x13\x01"
LAYERS = ("Flood_1Day_250m", "FloodCS_1Day_250m", "Flood_2Day_250m", "Flood_3Day_250m")
_HDF_LOCK = threading.Lock()  # the HDF4 library is not thread-safe; downloads run in threads, reads one at a time


def tiles_for_bounds(minx: float, miny: float, maxx: float, maxy: float) -> list[tuple[int, int]]:
    """(h, v) of every 10-degree tile that intersects the bounds."""
    h0, h1 = math.floor((minx + 180) / TILE_DEG), math.ceil((maxx + 180) / TILE_DEG)
    v0, v1 = math.floor((90 - maxy) / TILE_DEG), math.ceil((90 - miny) / TILE_DEG)
    return [(h, v) for v in range(max(v0, 0), min(v1, 18)) for h in range(max(h0, 0), min(h1, 36))]


def flood_codes(codes: np.ndarray, recurring_as: str) -> tuple[np.ndarray, np.ndarray]:
    """(observed, flood) pixel masks from MCDWD flag values."""
    if recurring_as not in ("flood", "dry"):
        raise ValueError(f"recurring_as must be 'flood' or 'dry', not {recurring_as!r}")
    observed = codes != 255
    flood = (codes == 3) | ((codes == 2) if recurring_as == "flood" else False)
    return observed, flood


def tile_counts(codes: np.ndarray, h: int, v: int, win: grid.Window) -> dict[str, np.ndarray] | None:
    """Per window cell: all pixels, observed pixels, and pixels of each flood class, from one tile.

    Counting codes 3 and 2 separately lets both recurring-flood mappings be scored from one read.
    Returns None when the tile does not reach the window.
    """
    lon0, lat0 = -180 + TILE_DEG * h, 90 - TILE_DEG * v
    minx, miny, maxx, maxy = win.bounds
    c0, c1 = max(math.floor((minx - lon0) / PIX), 0), min(math.ceil((maxx - lon0) / PIX), TILE_PX)
    r0, r1 = max(math.floor((lat0 - maxy) / PIX), 0), min(math.ceil((lat0 - miny) / PIX), TILE_PX)
    if c0 >= c1 or r0 >= r1:
        return None
    a = codes[r0:r1, c0:c1]
    lon = lon0 + (np.arange(c0, c1) + 0.5) * PIX
    lat = lat0 - (np.arange(r0, r1) + 0.5) * PIX
    rows, cols = grid.row_of(lat) - win.row0, grid.col_of(lon) - win.col0
    rin = (rows >= 0) & (rows < win.shape[0])
    cin = (cols >= 0) & (cols < win.shape[1])
    a = a[np.ix_(rin, cin)]
    rr, cc = np.meshgrid(rows[rin], cols[cin], indexing="ij")
    out = {"n_all": grid.bincount_2d(rr.ravel(), cc.ravel(), win.shape)}
    for key, m in (("n_obs", a != 255), ("n_flood", a == 3), ("n_recurring", a == 2)):
        out[key] = grid.bincount_2d(rr[m], cc[m], win.shape)
    return out


def values_from_counts(counts: dict[str, np.ndarray], recurring_as: str, observed_min_frac: float,
                       flood_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Value raster (1 / 0 / NODATA) and flood fraction (NaN where unobserved) from summed counts."""
    if recurring_as not in ("flood", "dry"):
        raise ValueError(f"recurring_as must be 'flood' or 'dry', not {recurring_as!r}")
    n_all, n_obs = counts["n_all"], counts["n_obs"]
    n_fl = counts["n_flood"] + (counts["n_recurring"] if recurring_as == "flood" else 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = (n_all > 0) & (n_obs / np.where(n_all > 0, n_all, 1) >= observed_min_frac)
        frac = np.where(observed, n_fl / np.where(n_obs > 0, n_obs, 1), np.nan).astype("float32")
    out = np.full(n_all.shape, NODATA, "uint8")
    out[observed] = DRY
    out[observed & (frac >= flood_threshold)] = FLOOD
    return out, frac


def earthdata_token(username: str, password: str) -> str:
    """An Earthdata user token for a login: the existing one, or a new one if none exists.

    POST https://urs.earthdata.nasa.gov/api/users/find_or_create_token with HTTP Basic auth (Earthdata
    Login user-token documentation). Tokens last 60 days, and a user may hold at most two; find-or-create
    never adds a third.
    """
    import base64

    # secrets stored via the CLI often carry a trailing newline: harmless in a .netrc (whitespace-split),
    # fatal in HTTP Basic auth
    username, password = username.strip(), password.strip()
    req = urllib.request.Request("https://urs.earthdata.nasa.gov/api/users/find_or_create_token", method="POST",
                                 headers={"Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            tok = json.loads(r.read()).get("access_token")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError("Earthdata login rejected (HTTP 401) while requesting a token") from e
        raise
    if not tok:
        raise RuntimeError("Earthdata token API answered without an access_token")
    return tok


def check_hdf_payload(head: bytes, url: str) -> None:
    """Raise if a download is not an HDF4 file (LAADS answers a rejected token with an HTML login page)."""
    if head[:4] == HDF4_MAGIC:
        return
    if head.lstrip()[:1] == b"<":
        raise PermissionError(f"Earthdata token rejected or expired: got an HTML page instead of HDF for {url}")
    raise ValueError(f"not an HDF4 file (first bytes {head[:8]!r}): {url}")


class MODISFloodSource:
    name = "modis"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.s = cfg["sources"]["modis"]
        self.token = os.environ.get(self.s["token_env"])
        user, pw = os.environ.get(self.s["login_user_env"]), os.environ.get(self.s["login_password_env"])
        if not self.token and user and pw:
            # scheduled jobs: the team's shared Earthdata login from the dsci scope (KB pipelines/imerg.md)
            self.token = earthdata_token(user, pw)
        if not self.token:
            raise RuntimeError(f"misconfigured: set {self.s['token_env']} (an Earthdata token), or "
                               f"{self.s['login_user_env']} and {self.s['login_password_env']}; see docs/decisions.md D-37")
        self.absent: list[tuple[str, str]] = []
        self.until = dt.date.fromisoformat(self.s["reprocessed_until"])

    def product(self, day: dt.date) -> str:
        return self.s["product_reprocessed"] if day <= self.until else self.s["product_nrt_archive"]

    def _get(self, url: str) -> bytes:
        from ds_flood_gfm.fusion.net import retry

        def _do():
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()

        import time

        for attempt in range(5):
            try:
                return retry(_do, f"MODIS {url}")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise PermissionError(f"Earthdata token rejected (HTTP {e.code}) for {url}") from e
                # LAADS answers load with 429/5xx: a busy server is transient, any other status is not
                if e.code in (429, 502, 503, 504) and attempt < 4:
                    time.sleep(30 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"unreachable: {url}")

    def day_listing(self, day: dt.date) -> dict[str, str]:
        """'hHHvVV' -> file name for one day (cached). Empty dict: the day folder does not exist upstream."""
        prod = self.product(day)
        path = self.cfg.local("raw", "modis_listing", f"{prod}_{day:%Y%j}.json")
        if path.exists():
            return json.loads(path.read_text())
        url = f"{self.s['archive']}/allData/{self.s['collection']}/{prod}/{day:%Y/%j}.json"
        try:
            d = json.loads(self._get(url))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            d = {"content": []}
        items = d.get("content", d) if isinstance(d, dict) else d
        out = {n.split(".")[2]: n for n in (x["name"] for x in items) if n.endswith(".hdf")}
        path.write_text(json.dumps(out))
        return out

    def read_tile(self, day: dt.date, h: int, v: int, layers: tuple[str, ...]) -> dict[str, np.ndarray] | None:
        """Requested layers of one tile-day, or None if the file does not exist upstream (recorded)."""
        key = f"h{h:02d}v{v:02d}"
        name = self.day_listing(day).get(key)
        if name is None:
            self.absent.append((day.isoformat(), key))
            return None
        url = f"{self.s['archive']}/allData/{self.s['collection']}/{self.product(day)}/{day:%Y/%j}/{name}"
        data = self._get(url)
        check_hdf_payload(data[:16], url)
        from pyhdf.SD import SD, SDC

        tmp_dir = self.cfg.local("raw", "modis_tmp")
        tmp_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".hdf", delete=False) as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            with _HDF_LOCK:
                sd = SD(str(tmp), SDC.READ)
                out = {layer: sd.select(layer).get() for layer in layers}
                sd.end()
        finally:
            tmp.unlink()
        return out

    def counts_day(self, win: grid.Window, day: dt.date, layers: tuple[str, ...] = LAYERS) -> dict[str, dict[str, np.ndarray]]:
        """Per layer, the window's pixel counts summed over every tile it touches."""
        acc = {layer: None for layer in layers}
        for h, v in tiles_for_bounds(*win.bounds):
            arrays = self.read_tile(day, h, v, layers)
            if arrays is None:
                continue
            for layer, codes in arrays.items():
                c = tile_counts(codes, h, v, win)
                if c is None:
                    continue
                acc[layer] = c if acc[layer] is None else {k: acc[layer][k] + c[k] for k in c}
        empty = {k: np.zeros(win.shape) for k in ("n_all", "n_obs", "n_flood", "n_recurring")}
        return {layer: (c if c is not None else empty) for layer, c in acc.items()}

    def observe_day(self, win: grid.Window, day: dt.date) -> tuple[np.ndarray, np.ndarray]:
        layer = self.s["layer"]
        c = self.counts_day(win, day, (layer,))[layer]
        return values_from_counts(c, self.s["recurring_as"], self.s["observed_min_frac"], self.s["flood_threshold"])
