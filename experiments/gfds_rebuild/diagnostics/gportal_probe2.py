"""Focused list-only G-Portal probe. For each start path: list it; if it has no files, descend into
the LAST child (latest version/year/month) until files appear. Optionally list extra named dirs.
Credentials from the repo .env, never printed. >= 0.5 s between listings, capped.
"""

import stat
from pathlib import Path
import sys
import time

import paramiko
from dotenv import dotenv_values

cfg = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
t = paramiko.Transport(("ftp.gportal.jaxa.jp", 2051))
t.banner_timeout = 60
t.connect(username=cfg["GPORTAL_USER"], password=cfg["GPORTAL_PASSWORD"])
sftp = paramiko.SFTPClient.from_transport(t)
n = [0]


def ls(path):
    if n[0] >= 60:
        raise RuntimeError("listing cap reached")
    n[0] += 1
    time.sleep(0.5)
    return sorted(sftp.listdir_attr(path), key=lambda a: a.filename)


def descend(path, depth=0):
    e = ls(path)
    dirs = [x.filename for x in e if stat.S_ISDIR(x.st_mode)]
    files = [(x.filename, x.st_size) for x in e if not stat.S_ISDIR(x.st_mode)]
    print(f"\n{path}  ({len(dirs)} dirs, {len(files)} files)")
    if dirs:
        print("  dirs:", dirs[:45], "..." if len(dirs) > 45 else "")
    if files:
        for f, s in files[:6]:
            print(f"  {s / 1e6:9.2f} MB  {f}")
        if len(files) > 6:
            print(f"  ...  last: {files[-1][0]}")
        print(f"  {len(files)} files, {sum(s for _, s in files) / 1e9:.3f} GB")
        return
    if dirs and depth < 8:
        descend(f"{path.rstrip('/')}/{dirs[-1]}", depth + 1)


try:
    for p in sys.argv[1:]:
        try:
            descend(p)
        except FileNotFoundError:
            print(f"\n{p}: not found")
        except PermissionError:
            print(f"\n{p} (or a child): PERMISSION DENIED for this account")
        except PermissionError:
            print(f"\n{p} (or a child): PERMISSION DENIED for this account")
finally:
    print(f"\nlistings used: {n[0]}")
    sftp.close()
    t.close()
