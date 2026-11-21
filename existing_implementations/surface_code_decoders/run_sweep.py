"""
run_sweep.py -- the rotated surface code decoded by every decoder we have, on the same shots.

What this file is for
---------------------
Last week's surface code results used only PyMatching (matching, MWPM). This script asks:
how do the qLDPC decoders (BP-OSD, BP-LSD, Relay-BP, Ambiguity Clustering), plain BP and
Union-Find do on the SAME surface code problem?

For every (distance, p):
  1. load IBM's rotated surface code Z-memory circuit (uniform circuit noise, d rounds),
  2. keep only the Z detectors (as IBM does), so every decoder gets the same problem,
  3. sample SHOTS shots once,
  4. give those same shots to each decoder; count failures and time.
Rows are appended to results_surface_decoders.csv; finished rows are skipped on rerun.

Usage:  python run_sweep.py            (see README for the measured run time)
        python run_sweep.py quick      (d = 3, 5 and fewer shots: a few minutes)

Sources: IBM Relay-BP repo circuits (github.com/trmue/relay, Apache-2.0); decoders: see decoders.py.
"""
import csv, sys, time
from pathlib import Path
import numpy as np, stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))   # for helpers/common.py
import common
import decoders as D

QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
DISTANCES = [3, 5] if QUICK else [3, 5, 7, 9]
NOISE = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008]
SHOTS = 2_000 if QUICK else 20_000
OUT = Path(__file__).resolve().parent / "results_surface_decoders.csv"
FIELDS = ["d", "p", "decoder", "shots", "failures", "seconds", "bp_not_converged", "detectors", "faults"]

done = set()
if OUT.exists():
    done = {(int(r["d"]), float(r["p"]), r["decoder"]) for r in csv.DictReader(open(OUT))}
else:
    csv.DictWriter(open(OUT, "w", newline=""), FIELDS).writeheader()

repo = common.get_relay_repo()
for d in DISTANCES:
    for p in NOISE:
        circuit = common.ibm_circuit("rotated_surface_code_memory_Z", d, p, z_detectors_only=True)
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, obs = circuit.compile_detector_sampler(seed=1000 * d + int(p * 1e4)).sample(SHOTS, separate_observables=True)
        for cls in D.ALL:
            if (d, p, cls.name) in done:
                continue
            t = time.time()
            dec = cls(dem)
            pred = np.asarray(dec.predict(dets)).reshape(SHOTS, -1)
            fails = int((pred.astype(bool) != obs).any(axis=1).sum())
            row = dict(d=d, p=p, decoder=cls.name, shots=SHOTS, failures=fails, seconds=round(time.time() - t, 2),
                       bp_not_converged=getattr(dec, "not_converged", ""), detectors=dem.num_detectors, faults=dem.num_errors)
            with open(OUT, "a", newline="") as fh:
                csv.DictWriter(fh, FIELDS).writerow(row)
            print(row, flush=True)
