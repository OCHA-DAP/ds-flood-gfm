"""Probe the new G-Portal SFTP host (AWS Transfer) with the .env credentials; never prints them."""
import stat
import sys
from pathlib import Path

import paramiko
from dotenv import dotenv_values

cfg = dotenv_values(Path(sys.argv[1]))
host, port = sys.argv[2], int(sys.argv[3])
t = paramiko.Transport((host, port))
t.banner_timeout = 60
try:
    t.connect(username=cfg["GPORTAL_USER"], password=cfg["GPORTAL_PASSWORD"])
except paramiko.AuthenticationException as e:
    print(f"connected to {host}:{port} but authentication failed: {e}")
    sys.exit(1)
print(f"connected and authenticated on {host}:{port}; server version {t.remote_version}")
s = paramiko.SFTPClient.from_transport(t)
for p in ("/", "/standard", "/standard/GCOM-W", "/standard/GCOM-W/GCOM-W.AMSR2", "/standard/GCOM-W/GCOM-W.AMSR2/L1B",
          "/standard/GCOM-W/GCOM-W.AMSR2/L1B/2/2016/08/20"):
    try:
        ents = s.listdir_attr(p)
    except OSError as e:
        print(f"{p}: {type(e).__name__}: {e}")
        continue
    print(f"{p}: {len(ents)} entries")
    for e in ents[:8]:
        print(f"   {'d' if stat.S_ISDIR(e.st_mode) else '-'} {e.st_size:>12} {e.filename}")
s.close(); t.close()
