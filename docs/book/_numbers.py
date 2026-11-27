"""Every number the book prints, read from results/ at render time.

Chapters import this in a hidden cell and print tables with `md(...)` or single values with `f(...)` inline
(`{python} f(x)`), so a number in the book can never drift from the file that holds it. Nothing here has a
default: a missing file, column, product or row raises (KeyError / FileNotFoundError) and the render fails.

Grew out of scripts/fusion/ops/cv_doc_tables.py, which printed the same tables for the review doc.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
CV = RES / "cv"

NAME = {"fusion": "fusion", "viirs": "VIIRS", "microwave": "microwave vote", "floodscan": "FloodScan", "gfds": "GFDS",
        "gfm": "GFM", "fusion_literal_dry": "fusion, old dry-day rule"}
SINGLE = ["viirs", "microwave", "floodscan", "gfds", "gfm"]
PRODUCTS = ["fusion", *SINGLE]


def f(x, d: int = 2) -> str:
    """A number as the book prints it; NaN as an en dash (a stated absence, never a hidden zero)."""
    if pd.isna(x):
        return "–"
    return f"{x:.{d}f}"


def pct(x, d: int = 0) -> str:
    return "–" if pd.isna(x) else f"{x:.{d}%}"


def ci(r, s: str, d: int = 2) -> str:
    """value (lo–hi) from columns s, s_lo, s_hi of a row."""
    return f"{f(r[s], d)} ({f(r[s + '_lo'], d)}–{f(r[s + '_hi'], d)})"


def md(df: pd.DataFrame) -> str:
    """A pipe table. Print it from a cell with `#| output: asis`."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def csv(name: str, where: Path = CV) -> pd.DataFrame:
    p = where / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run the script named for it in docs/log.md")
    return pd.read_csv(p)


def skill(name: str = "skill_overall.csv", mode: str = "all") -> pd.DataFrame:
    """A skill table restricted to one scoring mode ('all' cells or 'conditional'), indexed by product."""
    t = csv(name)
    t = t[t["mode"] == mode]
    if t.empty:
        raise KeyError(f"{name}: no rows with mode={mode!r}")
    return t


def overall(product: str, col: str, mode: str = "all"):
    """One number from skill_overall.csv, e.g. overall('fusion', 'f1')."""
    return skill("skill_overall.csv", mode).set_index("product").loc[product, col]


def by(group: str, value: str, product: str, col: str, mode: str = "all"):
    """One number from skill_by_<group>.csv, e.g. by('continent', 'Asia', 'fusion', 'f1')."""
    t = skill(f"skill_by_{group}.csv", mode)
    r = t[(t[group] == value) & (t["product"] == product)]
    if len(r) != 1:
        raise KeyError(f"skill_by_{group}.csv: {len(r)} rows for {group}={value!r}, product={product!r}")
    return r.iloc[0][col]


def table_skill(name: str = "skill_overall.csv", mode: str = "all", products=None, intervals: bool = True) -> str:
    """recall · precision · F1 per product, sorted by F1, with 95% intervals."""
    a = skill(name, mode).set_index("product")
    products = products or [p for p in a.sort_values("f1", ascending=False).index if p in NAME and p != "fusion_literal_dry"]
    rows = []
    for p in products:
        r = a.loc[p]
        if intervals:
            rows.append({"product": NAME[p], "recall": ci(r, "recall"), "precision": ci(r, "precision"), "F1": ci(r, "f1")})
        else:
            rows.append({"product": NAME[p], "recall": f(r.recall), "precision": f(r.precision), "F1": ci(r, "f1")})
    return md(pd.DataFrame(rows))
