"""Is JRC's GFDS archive still served, and which answer-key and baseline files exist?"""
import re
import time
import urllib.request

UA = {"User-Agent": "ds-flood-gfm research (OCHA CHD)"}
B = "https://www.gdacs.org/flooddetection/DATA"


def listing(path):
    req = urllib.request.Request(f"{B}/{path}/", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode(errors="replace")
        names = re.findall(r'href="[^"]*/([^"/]+\.(?:tif|txt))"', html, flags=re.I)
        return r.status, names
    except urllib.error.HTTPError as e:
        return e.code, []


for path in ("AMSR2/SignalTiffs/2026/09", "AMSR2/SignalTiffs/2018/08", "AMSR2/SourceTiffs/2018/08",
             "AMSR2/CalibrationTiffs/2018/08", "AMSR2/PositionTiffs/2018/08", "AMSR2/MagTiffs/2018/08",
             "ALL/AveragesAndSd", "AMSR2/AveragesAndSd", "ALL/SignalTiffs/2026/09"):
    status, names = listing(path)
    print(f"{path}: HTTP {status}, {len(names)} files; e.g. {names[:3]}")
    time.sleep(1.2)
