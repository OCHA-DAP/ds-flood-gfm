"""Download JRC's five AMSR2 answer-key files for 2016-08-20 (names from the folder listings; the 2016 files
carry a _GZ suffix)."""
import re
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "ds-flood-gfm research (OCHA CHD)"}
B = "https://www.gdacs.org/flooddetection/DATA/AMSR2"
DAY = "20160820"
OUT = Path(f"data/gdacs_gfds/rebuild/answer_key_{DAY}")
OUT.mkdir(parents=True, exist_ok=True)
for folder in ("SignalTiffs", "SourceTiffs", "CalibrationTiffs", "PositionTiffs", "MagTiffs"):
    html = urllib.request.urlopen(urllib.request.Request(f"{B}/{folder}/2016/08/", headers=UA), timeout=60).read().decode(errors="replace")
    names = [n for n in re.findall(r'href="[^"]*/([^"/]+\.tif)"', html, flags=re.I) if DAY in n]
    if len(names) != 1:
        raise FileNotFoundError(f"{folder}: expected one file for {DAY}, got {names}")
    time.sleep(1.3)
    dst = OUT / names[0]
    if dst.exists():
        continue
    data = urllib.request.urlopen(urllib.request.Request(f"{B}/{folder}/2016/08/{names[0]}", headers=UA), timeout=600).read()
    dst.write_bytes(data)
    print(f"{folder}: {names[0]} {len(data) / 1e6:.1f} MB", flush=True)
    time.sleep(1.3)
