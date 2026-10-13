"""Narrow retry for transient network failures.

Only connection resets, remote disconnects and timeouts are retried, a few times
with backoff, then the error is raised with the resource named. HTTP errors
(404, 403, ...) are never retried here: they carry meaning (absence) that the
caller must decide on.
"""

from __future__ import annotations

import http.client
import socket
import time
import urllib.error
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

TRANSIENT = (ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead, socket.timeout, TimeoutError)


def _is_transient(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, TRANSIENT):
        return True
    if isinstance(e, urllib.error.URLError) and isinstance(e.reason, (*TRANSIENT, OSError)):
        return True
    # GDAL /vsicurl network failures surface as rasterio errors whose text starts "CURL error"
    # (DNS, resets, timeouts); HTTP status errors read "HTTP response code: NNN" and are not retried
    if type(e).__name__ in {"RasterioIOError", "CPLE_HttpResponseError", "CPLE_OpenFailedError"} and "CURL error" in str(e):
        return True
    # a remote COG that GDAL momentarily cannot parse (truncated/garbled response); seen on EODC
    # for a file that reads fine seconds later. Bounded by `attempts`, then raised.
    if type(e).__name__ == "RasterioIOError" and "/vsicurl/" in str(e) and (
        "not recognized as being in a supported file format" in str(e) or "does not exist in the file system" in str(e)
    ):
        # GDAL also reports a failed open during a network blip as "does not exist"; files seen this
        # way read fine minutes later. A true 404 still raises after `attempts`.
        return True
    # requests/pystac_client wrap the same failures
    name = type(e).__name__
    return name in {"ConnectionError", "ChunkedEncodingError", "ReadTimeout", "ConnectTimeout", "APIError"} and "Connection" in str(e)


def retry(fn: Callable[[], T], what: str, attempts: int = 4, wait_s: float = 10.0) -> T:
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_transient(e) or i == attempts:
                if _is_transient(e):
                    raise RuntimeError(f"{what}: still failing after {attempts} attempts: {e}") from e
                raise
            time.sleep(wait_s * i)
    raise AssertionError("unreachable")
