"""Earth-leaving brightness temperature (ADD s. 3.6, eq. 8) with the RSS absorption model.

AER: TB_EL = (TB - A) / B with A the upwelling emission and B the transmittance, from a radiative-transfer
model driven by NWP profiles; reflected downwelling and surface temperature are deliberately not corrected.
Here the same two terms come from Wentz & Meissner (2000), AMSR Ocean Algorithm ATBD v2, eqs. 22, 24,
26-29 and Table 4, which need only columnar water vapour V (mm) and give, per frequency:
    A_O = aO1 + aO2 (T_D - 270)            vertically integrated oxygen absorption
    A_V = aV1 V + aV2 V^2                  water vapour absorption
    tau = exp(-sec(theta) (A_O + A_V))     clear-sky transmittance (cloud liquid water ignored)
    T_D = b0 + b1 V + ... + b4 V^4 + b5 zeta(T_S - T_V);   T_U = T_D + b6 + b7 V
    TB_up = (1 - tau) T_U
    TB_EL = (TB - TB_up) / tau
The b5 term uses sea-surface temperature; over land the 2 m air temperature stands in for T_S (OUR CHOICE;
the term is at most +-14 K x b5 = 8 K in T_D, i.e. ~0.4 K in TB_up at 36.5 GHz).
AMSR2 earth incidence angle 55 deg. Coefficients for 18.7, 23.8, 36.5 and 89.0 GHz.
"""

import numpy as np

SEC_THETA = 1 / np.cos(np.deg2rad(55.0))
COEF = {  # Table 4: b0..b7, aO1, aO2, aV1, aV2
    "18": (240.24, 298.88e-2, -725.93e-4, 814.50e-6, -36.07e-7, 0.61, -0.16, -1.69e-2, 12.15e-3, -0.61e-4, 1.73e-3, -0.05e-5),
    "23": (241.69, 310.32e-2, -814.29e-4, 998.93e-6, -48.37e-7, 0.20, -0.20, -5.21e-2, 15.75e-3, -0.87e-4, 5.14e-3, 0.19e-5),
    "36": (239.45, 254.41e-2, -512.84e-4, 452.02e-6, -14.36e-7, 0.58, -0.57, -2.38e-2, 40.06e-3, -2.00e-4, 1.88e-3, 0.09e-5),
    "89": (242.58, 302.33e-2, -749.76e-4, 880.66e-6, -40.88e-7, 0.62, -0.57, -8.07e-2, 53.35e-3, -1.18e-4, 8.78e-3, 0.80e-5),
}


def atmosphere(freq, v_mm, t_surf):
    """Transmittance tau and upwelling emission TB_up (K) for a frequency key, water vapour V (mm) and a
    surface temperature proxy (K). Arrays broadcast."""
    b0, b1, b2, b3, b4, b5, b6, b7, ao1, ao2, av1, av2 = COEF[freq]
    v = np.asarray(v_mm, dtype=float)
    vc = np.minimum(v, 58.0)  # eq. 26a is extrapolated linearly above 58 mm; clamping is close enough here
    t_v = np.where(v <= 48, 273.16 + 0.8337 * v - 3.029e-5 * v ** 3.33, 301.16)  # eq. 27
    dts = t_surf - t_v
    zeta = np.where(np.abs(dts) <= 20, 1.05 * dts * (1 - dts ** 2 / 1200), np.sign(dts) * 14)
    t_d = b0 + b1 * vc + b2 * vc ** 2 + b3 * vc ** 3 + b4 * vc ** 4 + b5 * zeta
    t_u = t_d + b6 + b7 * v
    a_o = ao1 + ao2 * (t_d - 270)
    a_v = av1 * v + av2 * v ** 2
    tau = np.exp(-SEC_THETA * (a_o + a_v))
    return tau, (1 - tau) * t_u


def earth_leaving(tb, freq, v_mm, t_surf):
    tau, tb_up = atmosphere(freq, v_mm, t_surf)
    return (tb - tb_up) / tau


if __name__ == "__main__":  # sanity: humid tropics vs dry Sahel at 36.5 GHz
    for v, t in ((50.0, 298.0), (15.0, 305.0)):
        tau, up = atmosphere("36", v, t)
        print(f"V {v:4.0f} mm, T {t:.0f} K: tau {tau:.3f}, TB_up {up:.1f} K; land 285/255 K -> EL "
              f"{earth_leaving(285.0, '36', v, t):.1f}/{earth_leaving(255.0, '36', v, t):.1f}, "
              f"PRI {(285 - 255) / (285 + 255):.4f} -> {(earth_leaving(285.0, '36', v, t) - earth_leaving(255.0, '36', v, t)) / (earth_leaving(285.0, '36', v, t) + earth_leaving(255.0, '36', v, t)):.4f}")
