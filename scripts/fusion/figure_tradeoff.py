"""The trade-off between catching floods and raising false alarms (Next steps 10): F1 against the share of events
with a false event, for every version of fusion at every threshold, with the cascade as a point.

Reads results/cv/pattern_model_sweep.csv (cv_pattern_model.py --sweep). Writes docs/figures/tradeoff_f1_false_events.png.
Run: uv run python scripts/fusion/figure_tradeoff.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ds_flood_gfm.fusion.config import load_config  # noqa: E402
from ds_flood_gfm.fusion.plots import GRID, INK, INK2, SURFACE, _style  # noqa: E402

VERSIONS = [  # fixed order, fixed colour per version
    ("independent", 0.9, "each product separately, with memory (as built)", "#2a78d6"),
    ("pattern", 0.9, "combinations learned, with memory", "#1baf7a"),
    ("independent", 0.0, "each product separately, no memory", "#eb6834"),
    ("pattern", 0.0, "combinations learned, no memory", "#eda100"),
]
LABEL_AT = {0.5, 0.1, 0.02, 0.005, 0.001}


def main() -> None:
    cfg = load_config()
    s = pd.read_csv(cfg.results_dir / "cv" / "pattern_model_sweep.csv")
    cas = s[s.evidence == "cascade"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), facecolor=SURFACE, sharex=True)
    for ax, col, title in ((axes[0], "pooled_f1", "Pooled F1 (the reference)"), (axes[1], "basin_weighted_f1", "Basin-weighted F1 (each basin's flood counts once)")):
        _style(ax)
        for evid, mem, name, colour in VERSIONS:
            v = s[(s.evidence == evid) & (s.memory == mem)].sort_values("threshold")
            # off the chart: thresholds where a version flags almost nothing, or almost everything (below the base rate)
            v = v[(v.pooled_f1 > 0.05) & (v.false_event < 0.95)]
            ax.plot(v.false_event * 100, v[col], color=colour, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, label=name)
            if evid == "independent" and mem == 0.9:
                for _, r in v[v.threshold.isin(LABEL_AT)].iterrows():
                    ax.annotate(f"{r.threshold:g}", (r.false_event * 100, r[col]), fontsize=7.5, color=INK2, xytext=(5, -10), textcoords="offset points")
                b = v[v.threshold == 0.5].iloc[0]
                ax.plot(b.false_event * 100, b[col], "o", ms=11, mfc="none", mec=colour, mew=2)
                ax.annotate("fusion as built", (b.false_event * 100, b[col]), fontsize=8, color=INK, xytext=(-8, 10), textcoords="offset points", ha="right")
        ax.plot(cas.false_event * 100, cas[col], "D", ms=9, color=INK, mec=SURFACE, mew=1.5, label="cascade: VIIRS, else GFM, else microwave", zorder=5)
        ax.annotate("cascade", (cas.false_event * 100, cas[col]), fontsize=8, color=INK, xytext=(8, 4), textcoords="offset points")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Events with a false event (%)", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 100)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("F1 on gold-map days (all cells)", fontsize=8.5, color=INK2)
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK)
    fig.suptitle("With memory, lowering the threshold trades false events for F1; without memory, fusion lands on the cascade",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Each point is one threshold (P at or above which a cell is called flooded); numbers by the as-built line give it. "
             "False event: the footprint flagged 10 to 30 days before the first map (148 eligible events). Points flagging almost "
             "nothing or almost everything are left off. Five-fold cross-validation, 170 events.", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = cfg.local_root.parent.parent / "docs" / "figures" / "tradeoff_f1_false_events.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
