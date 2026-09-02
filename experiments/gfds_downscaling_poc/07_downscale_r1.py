"""POC step 7: run the allocation stage on the R1 physics fraction.

Closes the loop the earlier steps left open: the 83 m radar comparison had
only been run with the R0 (SFED-trained) fraction as the GFDS-side input.
Here the fully independent R1 fraction (and its MDFF-thresholded variant)
goes through the identical flood-fill and is scored against the same radar
reference. Saves r1_downscaled.npz for the book chapter.
Run: uv run python experiments/gfds_downscaling_poc/07_downscale_r1.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path("experiments/gfds_downscaling_poc/04_downscale_validate.py").read_text()
# build downscale(), w_gfds, gfm_wet, dom, skill() etc.; stop before plotting
exec(_src.split("fig, axes")[0])

z2 = np.load(OUT / "independent_fractions.npz")
tt = pd.to_datetime(z2["time"])
window = pd.to_datetime(json.loads((OUT / "gfm_extent_dates.json").read_text()))
mask = np.asarray(tt.isin(window))
if int(mask.sum()) < 3:
    raise RuntimeError(f"only {mask.sum()} of the GFM window dates found in the "
                       "R1 stack — check independent_fractions.npz")

w_cells = np.nanmax(z2["R1"][mask], axis=0)
w_r1 = xr.full_like(w_gfds, np.nan)
for k, (ciy, cix) in enumerate(zip(z2["iy"], z2["ix"])):
    w_r1.values[ciy, cix] = w_cells[k]
w_r1m = xr.where(w_r1 >= 0.10, w_r1, 0.0)

ds_r1 = downscale(w_r1)
ds_r1m = downscale(w_r1m)

res = pd.DataFrame({
    "downscaled R1 physics": skill(ds_r1),
    "downscaled R1 + MDFF 0.10": skill(ds_r1m),
    "downscaled R0 (SFED-trained)": skill(ds_gfds),
    "downscaled SFED (licensed)": skill(ds_sfed),
    "history-only control": skill(ctrl),
}).T
print(res.to_string())
np.savez_compressed(OUT / "r1_downscaled.npz", ds_r1=ds_r1, ds_r1m=ds_r1m)
print("wrote", OUT / "r1_downscaled.npz")
