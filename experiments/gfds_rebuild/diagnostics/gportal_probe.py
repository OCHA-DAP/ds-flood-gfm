"""List-only probe of JAXA G-Portal SFTP: where AMSR2 / AMSR3 swath products live and how big a day is.

Reads GPORTAL_USER / GPORTAL_PASSWORD from the repo .env (never printed). One login, directory
listings only (no downloads), >= 0.5 s between listings, capped at MAX_LISTINGS.
"""

import stat
from pathlib import Path
import sys
import time

import paramiko
from dotenv import dotenv_values

ENV = str(Path(__file__).resolve().parents[3] / ".env")
HOST, PORT = "ftp.gportal.jaxa.jp", 2051
MAX_LISTINGS = 80
KEYWORDS = ("standard", "GCOM-W", "GOSAT-GW", "AMSR2", "AMSR3", "L1B", "L1R", "L3")

cfg = dotenv_values(ENV)
user, pw = cfg.get("GPORTAL_USER"), cfg.get("GPORTAL_PASSWORD")
if not user or not pw:
    sys.exit("GPORTAL_USER / GPORTAL_PASSWORD missing from .env")

t = paramiko.Transport((HOST, PORT))
t.banner_timeout = 60
t.connect(username=user, password=pw)
sftp = paramiko.SFTPClient.from_transport(t)
n = [0]


def ls(path):
    if n[0] >= MAX_LISTINGS:
        raise RuntimeError("listing cap reached")
    n[0] += 1
    time.sleep(0.5)
    return sorted(sftp.listdir_attr(path), key=lambda a: a.filename)


def show(path, entries, limit=40):
    dirs = [e.filename for e in entries if stat.S_ISDIR(e.st_mode)]
    files = [(e.filename, e.st_size) for e in entries if not stat.S_ISDIR(e.st_mode)]
    print(f"\n{path}  ({len(dirs)} dirs, {len(files)} files)")
    if dirs:
        print("  dirs:", dirs[:limit], "..." if len(dirs) > limit else "")
    for f, s in files[:8]:
        print(f"  {s / 1e6:9.1f} MB  {f}")
    if files:
        tot = sum(s for _, s in files)
        print(f"  ... {len(files)} files, {tot / 1e9:.2f} GB total")
    return dirs, files


# Walk only along directories whose names match the keywords, then descend into the
# last-listed (most recent) child for date levels, until files appear.
def walk(path, depth=0):
    entries = ls(path)
    dirs, files = show(path, entries)
    if files or depth > 9:
        return
    hits = [d for d in dirs if any(k.lower() in d.lower() for k in KEYWORDS)]
    if hits:
        for d in hits:
            walk(f"{path.rstrip('/')}/{d}", depth + 1)
    elif dirs:  # version / year / month / day levels: follow the latest only
        walk(f"{path.rstrip('/')}/{dirs[-1]}", depth + 1)


try:
    walk(sys.argv[1] if len(sys.argv) > 1 else "/")
finally:
    print(f"\nlistings used: {n[0]}")
    sftp.close()
    t.close()
