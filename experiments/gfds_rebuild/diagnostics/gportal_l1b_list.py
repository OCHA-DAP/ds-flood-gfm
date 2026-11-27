"""List JAXA AMSR2 L1B granules on G-Portal for 2016-08-19/20 (names carry start time and A/D)."""
import stat
import sys
from pathlib import Path

import paramiko
from dotenv import dotenv_values

cfg = dotenv_values(Path(sys.argv[1]))
t = paramiko.Transport(("ftp.gportal.jaxa.jp", 2051))
t.banner_timeout = 60
t.connect(username=cfg["GPORTAL_USER"], password=cfg["GPORTAL_PASSWORD"])
s = paramiko.SFTPClient.from_transport(t)
s.get_channel().settimeout(120)
for prod in ("L1B", "L1R"):
    for day in ("2016/08/19", "2016/08/20"):
        d = f"/standard/GCOM-W/GCOM-W.AMSR2/{prod}/2/{day}"
        try:
            ents = [e for e in s.listdir_attr(d) if not stat.S_ISDIR(e.st_mode)]
        except IOError as e:
            print(prod, day, "ERROR", e)
            continue
        print(prod, day, len(ents), "files; total MB", round(sum(e.st_size for e in ents) / 1e6))
        for e in ents[:4]:
            print("   ", e.filename, round(e.st_size / 1e6, 1), "MB")
s.close(); t.close()
