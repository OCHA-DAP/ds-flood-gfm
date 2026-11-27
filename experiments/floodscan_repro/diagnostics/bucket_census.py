"""Count gridded files per sensor per year in the CSU CDR bucket (no truncation: paginated)."""
import re
import sys
import time
import urllib.request

B = "https://s3.us-east-1.amazonaws.com/noaa-cdr-microwave-imager-brit-temp-csu-pds"


def keys(prefix):
    out, token = [], None
    while True:
        url = f"{B}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    xml = r.read().decode()
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        out += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


for kind in ("FCDR-GRID", "ICDR-GRID"):
    for year in range(2012, 2027):
        ks = keys(f"data/{kind}/{year}/")
        if not ks:
            continue
        sensors = {}
        for k in ks:
            m = re.search(r"/CSU_([A-Z0-9]+)_[FI]CDR-GRID_V02R00_([A-Z0-9]+)_D", k)
            sensors[m.group(1) + ":" + m.group(2)] = sensors.get(m.group(1) + ":" + m.group(2), 0) + 1
        print(kind, year, sensors, flush=True)
print("swath FCDR 2024 sensors:", sorted({re.search(r"/CSU_([A-Z0-9]+)_", k).group(1) for k in keys("data/FCDR/2024/")}))
