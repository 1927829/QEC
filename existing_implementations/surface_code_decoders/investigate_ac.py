"""
investigate_ac.py -- why does BP-OSD tie with matching, and why does Ambiguity Clustering beat both?

What this file is for
---------------------
The sweep (run_sweep.py) showed, on the rotated surface code:
    BP-OSD  ~= matching (MWPM)            Ambiguity Clustering (AC) ~6% better at d = 7, 9.
This script decodes the SAME shots with variants that each change one thing, and keeps the per-shot
results, so the differences can be tested shot by shot (McNemar test) instead of by comparing totals.

Variants
    MWPM                  PyMatching on Stim's decomposed error model (what the sweep used)
    MWPM exact            PyMatching on the merged fault table (one column per fault, exact weights)
    BP-OSD-0              ldpc BP + OSD order 0 (no search)
    BP-OSD-CS10           ldpc BP + OSD combination sweep, order 10 (what the sweep used)
    AC kappa=0            stabilizer-qec AC with no cluster growth (= BP + OSD-0 on its own BP)
    AC kappa=0.05         default (what the sweep used)
    AC kappa=0.2          larger clusters

Extra measurement (the "degeneracy gap")
    For a sample of shots: w_best = weight of the most likely single error (MWPM exact), and
    w_other = weight of the most likely error in the OTHER logical class, found by BP-OSD on the
    check matrix with the logical observable appended as an extra check whose value is forced to the
    opposite of MWPM's answer. weight = sum over chosen faults of ln((1-p)/p), so
    gap = w_other - w_best = ln( P(best error) / P(best error of the other class) ).
    A small gap means the two logical classes are nearly tied, which is where counting how MANY
    errors each class contains (degeneracy) can change the decision.

Output
    results_ac_investigation.csv   one row per (d, p, variant): failures, seconds
    ac_investigation_shots.npz     per-shot failure flags and gaps, for the notebook

Run:  python investigate_ac.py          (about 30 minutes on 2 cores)
Sources: Wolanski & Barber arXiv:2406.14527 (AC); Roffe et al. arXiv:2005.07016 (BP-OSD);
Higgott & Gidney arXiv:2303.15933 (PyMatching).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import scipy.sparse as sp
import stabilizer_qec as sq
from ldpc import BpOsdDecoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))
import common                                      # ../helpers/common.py: IBM circuits
from decoders import problem_from_dem              # same fault table as the BP decoders

HERE = Path(__file__).resolve().parent
POINTS = [(5, 0.004), (5, 0.006), (7, 0.004), (7, 0.006), (9, 0.004), (9, 0.006)]
SHOTS = 20_000                                     # per point; `python investigate_ac.py 500` for a quick run
GAP_SAMPLE = 400                                   # random shots per point for the gap distribution


def ldpc_osd(H, pr, method, order=0):
    """ldpc BP-OSD with the sweep's BP settings (min-sum, scaling 0.625, 80 iterations)."""
    return BpOsdDecoder(H, error_channel=list(pr), max_iter=80, bp_method="minimum_sum",
                        ms_scaling_factor=0.625, osd_method=method, osd_order=order)


def decode_ldpc(dec, L, dets):
    """Shot-by-shot decode, returning predicted observables."""
    return np.array([(L @ dec.decode(s)) % 2 for s in dets.astype(np.uint8)], dtype=np.uint8)


def matched_faults(m, H, s):
    """Turn PyMatching's matched edges (detector pairs; -1 = boundary) back into a 0/1 fault vector."""
    Hc = H.tocsc()
    if not hasattr(matched_faults, "cols") or matched_faults.H is not H:
        matched_faults.H = H                       # cache: detector set -> column index
        matched_faults.cols = {tuple(sorted(Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]])): j
                               for j in range(H.shape[1])}
    e = np.zeros(H.shape[1], dtype=np.uint8)
    for a, b in m.decode_to_edges_array(s):
        e[matched_faults.cols[tuple(sorted(x for x in (a, b) if x >= 0))]] ^= 1
    return e


def other_class_weight(H, L, pr, w, dets, pred):
    """Weight of the best error found in the logical class opposite to `pred`, shot by shot."""
    Ha = sp.vstack([H, L]).tocsc()                 # logical observable appended as an extra check
    dec = ldpc_osd(Ha, pr, "osd_cs", 10)
    out = []
    for s, p in zip(dets.astype(np.uint8), pred):
        e = dec.decode(np.concatenate([s, 1 - p]).astype(np.uint8))
        ok = ((H @ e) % 2 == s).all() and ((L @ e) % 2 != p).all()
        out.append(w @ e if ok else np.nan)        # nan: OSD found no valid error (never seen so far)
    return np.array(out)


def main():
    global SHOTS
    if len(sys.argv) > 1:
        SHOTS = int(sys.argv[1])
    out_csv = HERE / "results_ac_investigation.csv"
    rows, store = [], {}
    for d, p in POINTS:
        circuit = common.ibm_circuit("rotated_surface_code_memory_Z", d, p, z_detectors_only=True)
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, obs = circuit.compile_detector_sampler(seed=1000 * d + int(p * 1e4)).sample(
            SHOTS, separate_observables=True)
        H, L, pr = problem_from_dem(dem)
        w = np.log((1 - pr) / pr)                  # weight of each fault

        variants = {
            "MWPM": lambda: pymatching.Matching.from_detector_error_model(dem).decode_batch(dets),
            "MWPM exact": lambda: pymatching.Matching(H, weights=w, faults_matrix=L).decode_batch(dets),
            "BP-OSD-0": lambda: decode_ldpc(ldpc_osd(H, pr, "osd0"), L, dets),
            "BP-OSD-CS10": lambda: decode_ldpc(ldpc_osd(H, pr, "osd_cs", 10), L, dets),
            "AC kappa=0": lambda: sq.AmbiguityClustering(dem, kappa=0.0).decode_batch(dets),
            "AC kappa=0.05": lambda: sq.AmbiguityClustering(dem).decode_batch(dets),
            "AC kappa=0.2": lambda: sq.AmbiguityClustering(dem, kappa=0.2).decode_batch(dets),
        }
        fails = {}
        for name, run in variants.items():
            t = time.time()
            pred = np.asarray(run(), dtype=np.uint8).reshape(SHOTS, -1)
            sec = time.time() - t
            fails[name] = (pred != obs).any(axis=1)
            if name == "MWPM exact":
                pred_exact = pred[:, 0]
            rows.append(dict(d=d, p=p, variant=name, shots=SHOTS, failures=int(fails[name].sum()),
                             seconds=round(sec, 1)))
            print(rows[-1], flush=True)

        # degeneracy gap on a random sample + every shot where MWPM and AC (default) disagree
        rng = np.random.default_rng(d)
        idx = np.union1d(rng.choice(SHOTS, GAP_SAMPLE, replace=False),
                         np.flatnonzero(fails["MWPM exact"] != fails["AC kappa=0.05"]))
        m = pymatching.Matching(H, weights=w, faults_matrix=L)
        w_best = np.array([w @ matched_faults(m, H, s) for s in dets[idx]])
        w_other = other_class_weight(H, L, pr, w, dets[idx], pred_exact[idx][:, None])
        key = f"d{d}_p{p}"
        store[key + "_fails"] = np.array([fails[n] for n in variants])
        store[key + "_gap_idx"] = idx
        store[key + "_gap"] = w_other - w_best
        print(key, "gap computed on", len(idx), "shots", flush=True)

    store["variants"] = np.array(list(variants))
    np.savez_compressed(HERE / "ac_investigation_shots.npz", **store)
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
