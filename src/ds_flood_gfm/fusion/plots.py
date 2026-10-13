"""Static report figures (PNG) for the validation results.

Colours follow the team dataviz method's reference palette, light mode
(validated with scripts/validate_palette.js: all hard gates pass; aqua and
yellow sit below 3:1 on the surface, so every bar carries a direct value
label and the numbers are also in the results CSVs):
  fusion #2a78d6, GFM #eb6834, microwave #1baf7a, VIIRS #eda100 (fixed by entity).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
REF = "#9b9a95"
COLOR = {"fusion": "#2a78d6", "gfm": "#eb6834", "microwave": "#1baf7a", "viirs": "#eda100"}
LABEL = {"fusion": "Fusion", "gfm": "GFM (SAR)", "microwave": "Microwave (FloodScan|GFDS)", "viirs": "VIIRS (optical)"}
STRATA = ["open_nosnow", "vegetated_nosnow", "urban_nosnow", "open_snow", "vegetated_snow", "urban_snow"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def reliability(bins: pd.DataFrame, summary: pd.DataFrame, stratum: str, out: Path) -> None:
    """One series (fusion) against the 1:1 reference; bin sizes labelled where the bin is thin."""
    a = bins[bins.stratum == stratum].groupby("bin")[["n", "sum_p", "n_flood"]].sum()
    if a.empty:
        return
    mp, of = a.sum_p / a.n, a.n_flood / a.n
    s = summary.set_index("stratum").loc[stratum]
    fig, ax = plt.subplots(figsize=(4.4, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(0.97, 0.90, "perfect calibration", color=INK2, fontsize=7, ha="right", rotation=45, rotation_mode="anchor")
    ax.plot(mp, of, color=COLOR["fusion"], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    for x, y, n in zip(mp, of, a.n):
        if n < 2000:  # selective labels: flag the bins whose frequency rests on few cells
            ax.annotate(f"n={n:,}", (x, y), fontsize=6.5, color=INK2, xytext=(4, -9), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(flooded)", color=INK2, fontsize=8)
    ax.set_ylabel("Observed flood frequency (gold)", color=INK2, fontsize=8)
    slope = s.reliability_slope_wls
    ax.set_title(f"Fusion reliability, {stratum.replace('_', ' ')}\n"
                 f"slope {slope:.2f} (target 0.8-1.2)\n{int(s.n):,} cell-days, {int(s.n_flood):,} flooded",
                 color=INK, fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def csi_by_stratum(skill: pd.DataFrame, out: Path, mode: str = "all") -> None:
    """Small multiples, one per stratum: CSI of fusion and each model source, value-labelled bars."""
    t = skill[(skill.model == "full") & (skill["mode"] == mode) & skill["product"].isin(COLOR)]
    if t.empty:
        return
    order = ["fusion", "gfm", "microwave", "viirs"]
    short = {"fusion": "Fusion", "gfm": "GFM", "microwave": "Microwave", "viirs": "VIIRS"}
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.8), sharex=True, sharey=True, facecolor=SURFACE)
    finite = t.csi.values[np.isfinite(t.csi.values)]
    xmax = max(0.1, float(finite.max()) * 1.3) if finite.size else 1.0
    y = np.arange(len(order))[::-1]
    for ax, st in zip(axes.ravel(), STRATA):
        _style(ax)
        ax.grid(axis="y", visible=False)
        g = t[t.stratum == st].set_index("product")
        n = int(g.truth_flood.max()) if len(g) else 0
        ax.set_title(f"{st.replace('_', ' ')} · {n:,} flood cells", fontsize=8, color=INK, loc="left")
        if n == 0:
            ax.text(0.5, 0.5, "no held-out gold flood cells", ha="center", va="center", color=INK2, fontsize=8, transform=ax.transAxes)
            continue
        vals = [g.csi.get(p, np.nan) for p in order]
        ax.barh(y, np.nan_to_num(vals), height=0.72, color=[COLOR[p] for p in order], edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y, vals):
            ax.text((v if np.isfinite(v) else 0) + xmax * 0.02, yy, "n/a" if not np.isfinite(v) else f"{v:.2f}",
                    va="center", fontsize=7.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([short[p] for p in order], fontsize=8, color=INK2)
    for ax in axes.ravel():
        ax.set_xlim(0, xmax)
    fig.supxlabel("CSI on held-out gold days" + (" (a source's NoData counts as not detected)" if mode == "all" else ""), fontsize=8, color=INK2)
    fig.suptitle("Fusion vs single sources by stratum (held-out events)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
