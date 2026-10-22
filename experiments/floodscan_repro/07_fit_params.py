"""Step 4: fit the chain's free parameters against FloodScan (the user's calibration idea), one knob at a time.

Runs 05 (variant 'full' only) and 06 per parameter set as subprocesses, on the real tables, and tabulates
the scores. Knobs (05's P): w_exp (AER 1/0.0012), qpow_offset (added to the table Q_pow), mdff_scale,
outlier_dev. One-at-a-time sweeps around AER's values first; a joint refinement can follow once the
sensitive knobs are known. Fit year: 2024 (the only year 05 runs so far); hold-out comes later.

    uv run python experiments/floodscan_repro/07_fit_params.py [sweep.json]

sweep.json: {"name": [values, ...], ...}; default below. Output: results/floodscan_repro/fit_sweep.csv and
the per-run scorer tables under data/floodscan_repro/fit/<run>/.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data/floodscan_repro/fit"
OUT = ROOT / "results/floodscan_repro/fit_sweep.csv"
TABLES = ROOT / "data/floodscan_repro/static_tables.npz"
DEFAULT = {
    "w_exp": [1 / 0.0012 / 4, 1 / 0.0012 / 2, 1 / 0.0012, 2 / 0.0012, 4 / 0.0012],
    "qpow_offset": [-0.02, -0.01, 0.01, 0.02, 0.035],
    "mdff_scale": [0.25, 0.5, 0.75, 1.5],
    "outlier_dev": [0.05, 0.2, 1.0],
}
sweep = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else DEFAULT
UV = ["uv", "run", "--no-project", "--with", "numpy", "--with", "scipy", "python"]


def run(name, params):
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    pj = d / "params.json"
    pj.write_text(json.dumps({**params, "variants": ["full"]}))
    ff = d / "ff.npz"
    if not ff.exists():
        subprocess.run(UV + [str(ROOT / "experiments/floodscan_repro/05_ff_chain.py"), str(TABLES), str(ff), str(pj)],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["uv", "run", "python", str(ROOT / "experiments/floodscan_repro/06_score_2024.py"), str(ff), str(d)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    row = pd.read_csv(d / "nga2024_variant_summary.csv").set_index("variant").loc["full"]
    row_in = pd.read_csv(d / "nga2024_variant_summary_inland.csv").set_index("variant").loc["full"]
    return {"run": name, **params, **row.to_dict(), **{f"in_{k}": v for k, v in row_in.to_dict().items()}}


rows = [run("baseline", {})]
print(pd.DataFrame(rows).round(3).to_string(), flush=True)
for knob, values in sweep.items():
    for v in values:
        rows.append(run(f"{knob}={v:g}", {knob: v}))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(header=False), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
print(f"wrote {OUT}")
