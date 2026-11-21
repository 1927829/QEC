"""
run_gross.py -- regenerates data/results_gross.csv, the gross code BP-OSD results used by notebooks 01 and 02.

What this file is for
---------------------
Runs the same decoder (BP-OSD with IBM's settings) on two gross code memory circuits:
    ibm  IBM's own circuit (from IBM's Relay-BP repo), with IBM's Z-detector filter
    sq   stabilizer-qec's circuit for the same code and schedule
and adds IBM's published BP-OSD numbers at the same p, so the three can be compared in one table.

    Check 1 (is the pipeline right?): the result on IBM's circuit should match IBM's published number.
    Check 2 (is stabilizer-qec's circuit right?): through the identical pipeline it should land in the same range.

Usage (from this folder)
------------------------
    python run_gross.py                      # the saved run: p = 0.005 (1,500 shots) and p = 0.004 (3,000 shots)
    python run_gross.py 0.004:500            # your own plan, as p:shots pairs
    python run_gross.py 0.004:500 --out results/my_run.csv

By default it overwrites data/results_gross.csv, so notebooks 01 and 02 then show the new numbers.
Time: about 6 minutes per 1,000 shots per circuit on 2 cores (BP-OSD order 10 is the slow part).

Where things are set
--------------------
    noise model and circuits   helpers/circuits.py      (IBM's circuits exist for p = 0.001-0.01 only)
    decoder settings           helpers/decode_bposd.py
    Z-detector filter          helpers/ibm_filter.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "helpers"))               # the shared helper functions

from circuits import ibm_gross_memory_z, sq_gross_memory_z   # noqa: E402
from decode_bposd import logical_error_rate, wilson          # noqa: E402
from ibm_filter import ibm_filtered_z                        # noqa: E402

# IBM's reference numbers (their repo's sweep CSV: BP-OSD, Z memory, 12 rounds): p -> (failures, shots)
IBM_PUBLISHED = {0.004: (265, 10000), 0.005: (651, 5000)}
DEFAULT_PLAN = [(0.005, 1500), (0.004, 3000)]          # (p, shots) per circuit: the saved run
DEFAULT_OUT = HERE / "data" / "results_gross.csv"
COLUMNS = ["circuit", "p", "shots", "failures", "P_L_12rounds", "ci_low", "ci_high", "seconds"]


def build_circuit(name: str, p: float):
    """The two circuits being compared. IBM's comes from a file with the noise already in it."""
    if name == "ibm":
        return ibm_filtered_z(ibm_gross_memory_z(p), f"ibm_p{p}")
    return sq_gross_memory_z(p)


def run(plan, out, workers=None, circuits=("ibm", "sq")):
    """Run every (p, shots) in `plan` on each circuit and write the table to `out`. Returns the rows."""
    workers = workers or os.cpu_count() or 2           # one decoding process per CPU core
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", newline="") as fh:              # a fresh file each run
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for p, shots in plan:
            for name in circuits:
                n, f, s = logical_error_rate(build_circuit(name, p), shots, workers=workers,
                                             seed=1000 + int(p * 1e4))   # fixed seed: repeatable
                lo, hi = wilson(f, n)                    # 95% interval
                rows.append([name, p, n, f, f / n, lo, hi, round(s, 1)])
                w.writerow(rows[-1]); fh.flush()         # saved as it goes
                print(f"{name} p={p}: {f}/{n} = {f/n:.4f} [{lo:.4f}, {hi:.4f}] in {s:.0f} s", flush=True)
        for p, (f, n) in IBM_PUBLISHED.items():          # IBM's own rows, for the same table
            if any(abs(p - q) < 1e-12 for q, _ in plan):
                lo, hi = wilson(f, n)
                rows.append(["ibm_published", p, n, f, f / n, lo, hi, ""])
                w.writerow(rows[-1])
    print("wrote", out)
    return rows


def parse_plan(pairs):
    """'0.004:3000' -> (0.004, 3000)."""
    plan = []
    for item in pairs:
        p, shots = item.split(":")
        plan.append((float(p), int(shots)))
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", nargs="*", help="p:shots pairs, e.g. 0.004:3000 (default: the saved run)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV (default: data/results_gross.csv)")
    ap.add_argument("--workers", type=int, default=None, help="decoding processes (default: all cores)")
    args = ap.parse_args()
    run(parse_plan(args.plan) if args.plan else DEFAULT_PLAN, args.out, args.workers)
