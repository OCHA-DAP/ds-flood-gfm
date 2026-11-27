"""Where is our FloodScan-core FF 'mass' that SFED does not have? 0.5-deg blocks, by latitude band
and by whether SFED ever floods the block, plus a monthly split (wet-season soil moisture vs recession)."""
import datetime as dt
import numpy as np

W = "."  # run from the repo root
NGA = "/Users/zackarno/Documents/CHD/repos/ds-flood-gfm/data/gdacs_gfds/nga2022"
z = np.load(f"{W}/data/floodscan_repro/nga2022_core_l3.npz")
ff, lat, lon = z["ff"].astype("float64"), z["lat"], z["lon"]
days = [dt.date.fromisoformat(s) for s in z["days"]]
sc = np.load(f"{NGA}/sfed_coords.npz"); sx, sy = sc["x"], sc["y"]; res = 1 / 12
SF = np.stack([np.load(f"{NGA}/sfed_{d:%Y%m%d}.npy").astype("float64") for d in days])
oj0 = int(round((3.0 - (lon[0] - 0.05)) / 0.1)); oi0 = int(round(((lat[0] + 0.05) - 14.0) / 0.1))
sj0 = int(round((3.0 - (sx[0] - res / 2)) / res)); si0 = int(round(((sy[0] + res / 2) - 14.0) / res))


def blocks(x, i0, j0, k):
    sub = x[:, i0: i0 + 20 * k, j0: j0 + 23 * k].reshape(x.shape[0], 20, k, 23, k)
    n = np.isfinite(sub).sum((2, 4)); s = np.nansum(sub, (2, 4))
    return np.where(n >= 0.5 * k * k, s / np.maximum(n, 1), np.nan)


ob, sb = blocks(ff, oi0, oj0, 5), blocks(SF, si0, sj0, 6)
blat = 14.0 - 0.5 * (np.arange(20) + 0.5)
both = np.isfinite(ob) & np.isfinite(sb)
tot_o, tot_s = np.nansum(np.where(both, ob, 0)), np.nansum(np.where(both, sb, 0))
smax = np.nanmax(sb, 0)
print(f"total FF mass ours / SFED = {tot_o / tot_s:.2f}")
for name, m in {"north of 11N": blat > 11, "8-11N": (blat > 8) & (blat <= 11), "south of 8N": blat <= 8}.items():
    mm = both & m[None, :, None]
    print(f"  {name:12s}: share of our mass {np.nansum(np.where(mm, ob, 0)) / tot_o:.2f}, "
          f"share of SFED mass {np.nansum(np.where(mm, sb, 0)) / tot_s:.2f}")
never = (smax < 0.02)[None]
print(f"  blocks SFED never floods (max < 0.02): {never.mean():.2f} of blocks hold {np.nansum(np.where(both & never, ob, 0)) / tot_o:.2f} of our mass")
for mo in (8, 9, 10, 11):
    t = np.array([d.month == mo for d in days])
    print(f"  month {mo:2d}: mean ours {np.nanmean(np.where(both[t], ob[t], np.nan)):.4f}  mean SFED {np.nanmean(np.where(both[t], sb[t], np.nan)):.4f}")
