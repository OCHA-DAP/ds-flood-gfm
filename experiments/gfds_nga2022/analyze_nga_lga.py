"""Stage 2: LGA-level (admin2) comparison — the common-data-model test.

Aggregates GFDS 4-day anomaly and FloodScan SFED to Nigeria LGAs per day,
scores per-LGA correlation and peak-timing lag, and checks the known
reference LGAs (Kogi confluence + Bayelsa delta).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import xarray as xr
from rasterio.transform import from_bounds as transform_from_bounds
from rasterstats import zonal_stats

OUT = Path("outputs/gfds_validation")

anom4 = xr.open_dataarray(OUT / "nga_anom_static_4day.nc")
sfed = xr.open_dataarray(OUT / "nga_sfed_on_gfds.nc")
sfed, anom4 = xr.align(sfed, anom4)

adm2 = stratus.codab.load_codab_from_blob("nga", admin_level=2)

ny, nx = anom4.sizes["y"], anom4.sizes["x"]
res = 0.09
west, east = float(anom4.x.min()) - res / 2, float(anom4.x.max()) + res / 2
north, south = float(anom4.y.max()) + res / 2, float(anom4.y.min()) - res / 2
affine = transform_from_bounds(west, south, east, north, nx, ny)

# y must be descending for this affine; verify and flip if needed
if anom4.y.values[0] < anom4.y.values[-1]:
    anom4 = anom4.isel(y=slice(None, None, -1))
    sfed = sfed.isel(y=slice(None, None, -1))

rows = []
for ts in pd.to_datetime(anom4.time.values):
    a = anom4.sel(time=ts).values.astype("float64")
    s = sfed.sel(time=ts).values.astype("float64")
    za = zonal_stats(adm2, a, affine=affine, nodata=np.nan,
                     all_touched=True, stats="mean")
    zs = zonal_stats(adm2, s, affine=affine, nodata=np.nan,
                     all_touched=True, stats="mean")
    for pc, name, adm1, ga, gs in zip(
            adm2["ADM2_PCODE"], adm2["ADM2_EN"], adm2["ADM1_EN"],
            (z["mean"] for z in za), (z["mean"] for z in zs)):
        rows.append({"date": ts, "pcode": pc, "lga": name, "state": adm1,
                     "gfds": ga, "sfed": gs})

df = pd.DataFrame(rows)
df.to_parquet(OUT / "nga_lga_daily.parquet")

per = []
for (pc, lga, state), sub in df.groupby(["pcode", "lga", "state"]):
    sub = sub.dropna(subset=["gfds", "sfed"]).sort_values("date")
    if len(sub) < 60:
        continue
    r = sub["gfds"].corr(sub["sfed"])
    lag = (sub.loc[sub["gfds"].idxmax(), "date"]
           - sub.loc[sub["sfed"].idxmax(), "date"]).days
    per.append({"pcode": pc, "lga": lga, "state": state, "r": r,
                "peak_lag_days": lag, "sfed_max": sub["sfed"].max(),
                "gfds_max": sub["gfds"].max(), "n_days": len(sub)})
res_df = pd.DataFrame(per)
res_df.to_csv(OUT / "nga_lga_comparison.csv", index=False)

fl = res_df[res_df.sfed_max > 0.05]
print(f"LGAs scored: {len(res_df)} | 'flooded' (SFED mean-max>0.05): {len(fl)}")
print(f"flooded LGAs: median r = {fl.r.median():.2f} | "
      f"r>=0.6: {(fl.r >= 0.6).mean():.0%} | "
      f"median |peak lag| = {fl.peak_lag_days.abs().median():.0f} d")
print(f"non-flooded LGAs: median r = {res_df[res_df.sfed_max <= 0.05].r.median():.2f}")

REF = ["Lokoja", "Ajaokuta", "Ibaji", "Bassa", "Kogi", "Yenagoa",
       "Sagbama", "Ekeremor"]
print("\n--- reference LGAs (known flooded, Kogi + Bayelsa) ---")
cols = ["lga", "state", "r", "peak_lag_days", "sfed_max", "gfds_max"]
ref_rows = res_df[res_df.lga.isin(REF)][cols].sort_values("r", ascending=False)
print(ref_rows.to_string(index=False))

# small multiple: reference LGA time series
fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
for ax, lga in zip(axes.ravel(), ["Lokoja", "Ajaokuta", "Ibaji",
                                  "Yenagoa", "Sagbama", "Ekeremor"]):
    sub = df[df.lga == lga].sort_values("date")
    ax.plot(sub.date, sub.gfds, color="tab:red", lw=1)
    ax.set_ylabel("GFDS anomaly (sd)", color="tab:red", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(sub.date, sub.sfed, color="tab:blue", lw=1)
    ax2.set_ylabel("SFED", color="tab:blue", fontsize=8)
    row = res_df[res_df.lga == lga]
    r = row.r.iloc[0] if len(row) else np.nan
    ax.set_title(f"{lga} (r={r:.2f})", fontsize=10)
    ax.axvline(pd.Timestamp("2022-09-13"), ls="--", c="k", alpha=0.4)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "nga_reference_lgas.png", dpi=150, bbox_inches="tight")
print("\nwrote nga_lga_comparison.csv, nga_lga_daily.parquet, nga_reference_lgas.png")
