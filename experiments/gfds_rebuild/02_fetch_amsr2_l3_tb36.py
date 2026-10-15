"""Fetch AMSR2 L3 36 GHz brightness temperature (0.1 deg, daily, one file per pass) from JAXA G-Portal.

Source: G-Portal SFTP (ftp.gportal.jaxa.jp:2051), /standard/GCOM-W/GCOM-W.AMSR2/L3.TB36GHz_10/2/YYYY/MM/
Files: GW1AM2_YYYYMMDD_01D_EQM{A,D}_L3SGT36HA*.h5 (global equirectangular; A = ascending, D = descending).
The polar-stereographic variants (PNM*, PSM*) are skipped.

Credentials: GPORTAL_USER / GPORTAL_PASSWORD in the repo .env (never printed).
Resumable: files already present with the remote size are skipped. Files are fetched one at a
time, >= 1 s apart (G-Portal terms warn against bulk scraping). Run ONE instance at a time: with
two concurrent sessions on one account, G-Portal dropped a connection (2026-10-15).
A dropped connection is retried by reconnecting (30 s pause, at most 3 tries per file); any other
error, or a third drop, raises.

    uv run --no-project --with paramiko --with python-dotenv \
        python experiments/gfds_rebuild/02_fetch_amsr2_l3_tb36.py 2022-10-01 2022-10-31
"""

import datetime as dt
import stat
import sys
import time
from pathlib import Path

import paramiko
from dotenv import dotenv_values

ENV = Path(__file__).resolve().parents[2] / ".env"
ROOT = "/standard/GCOM-W/GCOM-W.AMSR2/L3.TB36GHz_10/2"
CACHE = Path("data/gdacs_gfds/rebuild/amsr2_l3_tb36")

start, end = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
cfg = dotenv_values(ENV)
if not cfg.get("GPORTAL_USER") or not cfg.get("GPORTAL_PASSWORD"):
    sys.exit(f"GPORTAL_USER / GPORTAL_PASSWORD missing from {ENV}")

def connect():
    t = paramiko.Transport(("ftp.gportal.jaxa.jp", 2051))
    t.banner_timeout = 60
    t.connect(username=cfg["GPORTAL_USER"], password=cfg["GPORTAL_PASSWORD"])
    s = paramiko.SFTPClient.from_transport(t)
    # Without a read timeout a stalled transfer hangs forever (seen 2026-10-15: connection left
    # ESTABLISHED, no data for 50+ min). 120 s with no data -> socket timeout -> reconnect below.
    s.get_channel().settimeout(120)
    return t, s


DROPPED = (paramiko.SSHException, EOFError, ConnectionError, TimeoutError)
t, sftp = connect()


def get_with_reconnect(remote, local):
    global t, sftp
    for attempt in range(3):
        try:
            sftp.get(remote, local)
            return
        except DROPPED as e:
            if attempt == 2:
                raise
            print(f"  connection dropped on {remote} ({e!r}); reconnecting in 30 s", flush=True)
            sftp.close()
            t.close()
            time.sleep(30)
            t, sftp = connect()


months = sorted({(d.year, d.month) for d in (start + dt.timedelta(n) for n in range((end - start).days + 1))})
stats = {"fetched": 0, "skipped": 0}
try:
    for y, m in months:
        remote_dir = f"{ROOT}/{y}/{m:02d}"
        entries = [e for e in sftp.listdir_attr(remote_dir) if not stat.S_ISDIR(e.st_mode)]
        for e in sorted(entries, key=lambda e: e.filename):
            name = e.filename
            if "_01D_EQM" not in name or not name.endswith(".h5"):
                continue
            day = dt.datetime.strptime(name.split("_")[1], "%Y%m%d").date()
            if not start <= day <= end:
                continue
            out = CACHE / f"{y}" / f"{m:02d}" / name
            if out.exists() and out.stat().st_size == e.st_size:
                stats["skipped"] += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            get_with_reconnect(f"{remote_dir}/{name}", str(tmp))
            if tmp.stat().st_size != e.st_size:
                raise IOError(f"size mismatch for {name}: {tmp.stat().st_size} != {e.st_size}")
            tmp.rename(out)
            stats["fetched"] += 1
            print(f"{name}  {e.st_size / 1e6:.1f} MB", flush=True)
            time.sleep(1.0)
finally:
    sftp.close()
    t.close()

expected = 2 * ((end - start).days + 1)
have = stats["fetched"] + stats["skipped"]
print(f"done: {stats}; {have} of {expected} expected pass files")
if have != expected:
    raise SystemExit(f"MISSING {expected - have} pass files in {start}..{end} (listed upstream as absent)")
