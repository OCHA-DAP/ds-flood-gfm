"""Transient-retry rules. Why: a dropped connection must not fail a multi-hour run,
but an HTTP 403/404 means "absent" for GDACS/GFM and must reach the caller unchanged,
and a persistent failure must raise, naming the resource, rather than loop forever."""

import http.client
import urllib.error

import pytest

from ds_flood_gfm.fusion import net


def test_transient_then_success():
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return "ok"

    assert net.retry(f, "x", attempts=4, wait_s=0) == "ok" and calls["n"] == 3


def test_http_error_not_retried():
    calls = {"n": 0}

    def f():
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        net.retry(f, "x", attempts=4, wait_s=0)
    assert calls["n"] == 1


def test_persistent_failure_raises_with_resource():
    def f():
        raise ConnectionResetError("reset")

    with pytest.raises(RuntimeError, match="VIIRS listing"):
        net.retry(f, "VIIRS listing 2024/09/01", attempts=2, wait_s=0)


def test_gdal_curl_error_is_transient_but_http_status_is_not():
    from rasterio.errors import RasterioIOError

    assert net._is_transient(RasterioIOError("CURL error: Could not resolve host: noaa-jpss.s3.amazonaws.com"))
    assert not net._is_transient(RasterioIOError("HTTP response code: 404"))


def test_unparseable_remote_cog_is_retried():
    from rasterio.errors import RasterioIOError

    assert net._is_transient(RasterioIOError("'/vsicurl/https://x/y.tif' not recognized as being in a supported file format."))
    assert not net._is_transient(RasterioIOError("'/local/y.tif' not recognized as being in a supported file format."))
