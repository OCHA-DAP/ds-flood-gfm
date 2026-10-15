"""Fetch JRC's own GFDS per-sensor AMSR2 signal for a date range: the reference our rebuild is scored against.

Source: https://www.gdacs.org/flooddetection/DATA/AMSR2/SignalTiffs/YYYY/MM/signal_YYYYMMDD_HD5.tif
(0.09 deg, 4000x2000, int32 x 1e6). Whole files, up to 3 connections, each >= 1.2 s between requests.
A day missing upstream (HTTP 403/404) is recorded in missing.txt after a known-good file is confirmed
readable, so a block cannot pass as a gap. Any other error raises.

    uv run python experiments/gfds_rebuild/03_fetch_jrc_reference.py 2022-10-01 2022-10-31
"""

import datetime as dt
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.gdacs.org/flooddetection/DATA/AMSR2/SignalTiffs"
UA = "ds-flood-gfm research (OCHA CHD)"
KNOWN_GOOD = f"{BASE}/2024/09/signal_20240915_HD5.tif"
CACHE = Path("data/gdacs_gfds/rebuild/jrc_amsr2_signal")
_local = threading.local()


def get(url, method="GET"):
    wait = 1.2 - (time.monotonic() - getattr(_local, "last", 0.0))
    if wait > 0:
        time.sleep(wait)
    _local.last = time.monotonic()
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError) as e:  # network timeout: retry, then raise
            if attempt == 2:
                raise
            print(f"  retry {url}: {e}", flush=True)
            time.sleep(30)


def fetch(day):
    name = f"signal_{day:%Y%m%d}_HD5.tif"
    out = CACHE / name
    if out.exists():
        return "skipped"
    try:
        data = get(f"{BASE}/{day:%Y}/{day:%m}/{name}")
    except urllib.error.HTTPError as e:
        if e.code not in (403, 404):
            raise
        get(KNOWN_GOOD, method="HEAD")  # raises if the server is refusing us
        with open(CACHE / "missing.txt", "a") as fh:
            fh.write(f"{day} HTTP {e.code}\n")
        return "absent"
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(out)
    print(f"{name}  {len(data) / 1e6:.1f} MB", flush=True)
    return "fetched"


start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
CACHE.mkdir(parents=True, exist_ok=True)
days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
with ThreadPoolExecutor(3) as ex:
    results = list(ex.map(fetch, days))
print({k: results.count(k) for k in ("fetched", "skipped", "absent")})
