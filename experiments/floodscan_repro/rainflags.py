"""AER's precipitation flags (ADD s. 3.7): Ferraro et al. (1996) land decision tree with AER's modifications.

Inputs are top-of-atmosphere brightness temperatures (K) on the same grid: 19 GHz V/H (AMSR2 18.7),
22 GHz V (AMSR2 23.8) and 85 GHz V (AMSR2 89 A-scan). AER runs this on its 27 km composite footprints;
here it runs on the CSU 0.25-deg grid cells.

Scattering index over land (Grody 1991; Ferraro 1996 eq. 6):
    SI = 451.9 - 0.44 T19V - 1.775 T22V + 0.00575 T22V^2 - T85V
Primary rain flag (Ferraro 1996 fig. 9 as modified by AER, s. 3.7 step 1):
    scattering:  SI > SI0(dc, x)                 (AER: local climatology instead of the constant 10)
    not snow:    not (T22V < 264 and T22V < 175 + 0.49 T85V)
    not desert:  not (T19V - T19H > 22)
    not arid:    not (SI < SI0 + 5 and T85V > 253 and T19V - T19H > 7)
                 and not (SI >= SI0 + 5 and T85V > 255 and T19V - T19H > 15)
Final flag (step 2): primary flags dilated by a radius and re-tested at SI > SI0 - 1; desert and arid
tests are not re-applied. AER dilates by 7 cells of 5 arcmin (35', 0.58 deg); here 2 cells of 0.25 deg.
AER separates land and water decisions where footprints are > 95% water; cells that wet are excluded
from the flooded-fraction chain anyway (f_pow > 0.5), so everything here uses the land tree.
Frozen-surface flags (Ferraro fig. 1) are not built: the box is 0-20 N and never freezes.

SI0 climatology (s. 3.7 step 1.1), built in 04 from the training years:
    SI61     61-day running median of SI per year and pass, 15th percentile over years per day-of-year
    dSI61    descending minus ascending SI61 (arid-conditions indicator)
    SI0      10 + 2 max(dSI61, 0) + max(SI61 - 7, -1)
"""

import numpy as np
from scipy.ndimage import binary_dilation

DILATE_CELLS = 2  # AER: 7 cells at 5' for 27/36 km data


def scattering_index(t19v, t22v, t85v):
    return 451.9 - 0.44 * t19v - 1.775 * t22v + 0.00575 * t22v ** 2 - t85v


def primary_rain(si, si0, t19v, t19h, t22v, t85v):
    with np.errstate(invalid="ignore"):
        scattering = si > si0
        snow = (t22v < 264) & (t22v < 175 + 0.49 * t85v)
        desert = (t19v - t19h) > 22
        arid = ((si < si0 + 5) & (t85v > 253) & ((t19v - t19h) > 7)) | ((si >= si0 + 5) & (t85v > 255) & ((t19v - t19h) > 15))
    return scattering & ~snow & ~desert & ~arid


def final_rain(si, si0, t19v, t19h, t22v, t85v):
    """Primary flags dilated and re-tested (s. 3.7 step 2). NaN inputs never flag."""
    primary = primary_rain(si, si0, t19v, t19h, t22v, t85v)
    yy, xx = np.mgrid[-DILATE_CELLS : DILATE_CELLS + 1, -DILATE_CELLS : DILATE_CELLS + 1]
    disk = (yy ** 2 + xx ** 2) <= DILATE_CELLS ** 2
    disk = disk.reshape((1,) * (primary.ndim - 2) + disk.shape)  # dilate within each 2-D field of a (time, y, x) stack
    grown = binary_dilation(primary, structure=disk)
    with np.errstate(invalid="ignore"):
        retest = si > si0 - 1
    return primary | (grown & retest)


def si0_from_training(si61_years_by_pass):
    """si61_years_by_pass: {'asc': array (years, 365, ny, nx), 'dsc': same}; the 61-day running median of
    SI per year on the climatological day-of-year axis. Returns {'asc': SI0, 'dsc': SI0} (365, ny, nx)."""
    with np.errstate(all="ignore"):
        p15 = {p: np.nanpercentile(a, 15, axis=0) for p, a in si61_years_by_pass.items()}
    dsi = p15["dsc"] - p15["asc"]
    return {p: 10 + 2 * np.maximum(dsi, 0) + np.maximum(p15[p] - 7, -1) for p in p15}
