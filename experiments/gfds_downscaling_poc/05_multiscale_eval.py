"""POC step 5: at what scale does the 10 km microwave signal add skill?

Aggregates the three downscaled maps and the GFM reference to a ladder of
evaluation scales and correlates wet fractions. Separates what the static
GSW prior contributes (all of the fine-scale skill) from what the GFDS/SFED
10 km fields contribute (skill above the prior, growing with scale).
Run: uv run python experiments/gfds_downscaling_poc/05_multiscale_eval.py
(Reuses step 4's construction by executing its module body up to plotting.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path("experiments/gfds_downscaling_poc/04_downscale_validate.py").read_text()
exec(_src.split("fig, axes")[0])  # build ds_gfds, ds_sfed, ctrl, gfm_wet, dom


def block_reduce(a, k, m):
    ny, nx = a.shape
    ny2, nx2 = ny // k * k, nx // k * k
    aa = np.where(m, a, np.nan)[:ny2, :nx2].reshape(ny2 // k, k, nx2 // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(aa, axis=3), axis=1)


rows = []
for k, label in [(1, "83 m"), (6, "0.5 km"), (12, "1 km"), (36, "3 km"), (72, "6 km")]:
    g = block_reduce(gfm_wet.astype(float), k, dom)
    valid_share = block_reduce(dom.astype(float), k, np.ones_like(dom, bool))
    mm = np.isfinite(g) & (valid_share > 0.5)
    for name, pred in [("GFDS", ds_gfds), ("SFED", ds_sfed), ("prior-only", ctrl)]:
        p = block_reduce(pred.astype(float), k, dom)
        ok = mm & np.isfinite(p)
        rows.append({
            "scale": label, "map": name,
            "r_vs_GFM": round(float(np.corrcoef(p[ok], g[ok])[0, 1]), 3),
            "RMSE": round(float(np.sqrt(np.mean((p[ok] - g[ok]) ** 2))), 4),
        })

df = pd.DataFrame(rows).pivot(index="scale", columns="map", values="r_vs_GFM")
df = df.loc[["83 m", "0.5 km", "1 km", "3 km", "6 km"]]
print("correlation of wet fraction vs GFM, by evaluation scale:")
print(df.to_string())
df.to_csv(OUT / "multiscale_corr.csv")
print("wrote", OUT / "multiscale_corr.csv")
