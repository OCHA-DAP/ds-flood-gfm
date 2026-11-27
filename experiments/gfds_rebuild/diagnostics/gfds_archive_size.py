"""Size the GFDS archive from the server's directory listings (no data files downloaded).

For each folder: list years, list months per year, list ONE month per year (the middle
one present) to get mean file size and files/month; estimate the year as
mean_size * files_in_sampled_month * months_present. Exact for the sampled month,
extrapolated for the rest. Requests >= 1.2 s apart. Any HTTP error other than 404 raises.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.gdacs.org/flooddetection/DATA"
UA = "ds-flood-gfm research (OCHA CHD)"
ROW = re.compile(r"\d+/\d+/\d{4}\s+\d+:\d+ [AP]M\s+(\d+|&lt;dir&gt;) <A HREF=\"[^\"]+\">([^<]+)</A>")
FOLDERS = sys.argv[2:] or [
    "ALL/SignalTiffs",
    "AMSR2/SignalTiffs", "AMSR2/SourceTiffs", "AMSR2/CalibrationTiffs", "AMSR2/PositionTiffs",
    "GPM/SignalTiffs", "GPM/SourceTiffs", "GPM/CalibrationTiffs", "GPM/PositionTiffs",
    "SINGLE/SignalTiffs",
]
_last = [0.0]


def listing(path):
    wait = 1.2 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    req = urllib.request.Request(f"{BASE}/{path}/", headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                html = r.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, TimeoutError) as e:  # network timeout/handshake: retry, then raise
            if attempt == 2:
                raise
            print(f"  retry {path}: {e}", flush=True)
            time.sleep(30)
    return [(name, None if size.startswith("&") else int(size)) for size, name in ROW.findall(html)]


out = {}
for f in FOLDERS:
    years = [n for n, s in listing(f) if s is None and n.isdigit()]
    per_year = {}
    for y in years:
        months = [n for n, s in (listing(f"{f}/{y}") or []) if s is None]
        if not months:
            per_year[y] = {"months": 0, "est_bytes": 0}
            continue
        mid = months[len(months) // 2]
        files = [s for n, s in (listing(f"{f}/{y}/{mid}") or []) if s is not None and n.lower().endswith(".tif")]
        mean = sum(files) / len(files) if files else 0
        per_year[y] = {"months": len(months), "sampled": mid, "files_in_sample": len(files),
                       "mean_mb": round(mean / 1e6, 2), "est_bytes": int(mean * len(files) * len(months))}
    total = sum(v["est_bytes"] for v in per_year.values())
    out[f] = {"years": f"{years[0]}-{years[-1]}" if years else "none", "est_gb": round(total / 1e9, 1), "per_year": per_year}
    print(f"{f:28s} {out[f]['years']:10s} ~{out[f]['est_gb']:7.1f} GB", flush=True)

with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print(f"TOTAL ~{sum(v['est_gb'] for v in out.values()):.1f} GB")
