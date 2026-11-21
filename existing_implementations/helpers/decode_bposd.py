"""
decode_bposd.py -- sample a memory circuit and decode it with BP-OSD.

What this file is FOR :
    Turns a noisy Stim circuit into a logical error rate. Steps:
      1. Stim builds the detector error model (DEM): every possible fault, its
         probability, and which detectors / logical observables it flips.
      2. That DEM becomes a check matrix H (detectors x faults), a logical matrix
         L (observables x faults) and prior probabilities for each fault.
      3. Stim samples shots: which detectors fired, which logicals flipped.
      4. BP-OSD (Roffe's `ldpc`) guesses the faults from the detectors; we apply
         L to that guess and call the shot a FAILURE if any of the 12 predicted
         logical flips disagrees with the real ones.

    Decoder settings copy IBM's reference run in the Relay-BP repo
    (examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv):
    min-sum BP, 80 iterations, OSD-CS order 10.
"""

import multiprocessing as mp
import sys
import time

import numpy as np
import stim
from beliefmatching import detector_error_model_to_check_matrices
from ldpc import BpOsdDecoder

IBM_SETTINGS = dict(bp_method="minimum_sum", max_iter=80, osd_method="osd_cs", osd_order=10)


def _matrices(circuit: stim.Circuit):
    """DEM -> (H, L, priors) as sparse matrices / array."""
    dem = circuit.detector_error_model(decompose_errors=False)
    m = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    return m.check_matrix, m.observables_matrix, m.priors


def _worker(args):
    """Decode one batch of shots in a separate process; return (shots, failures)."""
    circuit_text, shots, seed, settings = args
    circuit = stim.Circuit(circuit_text)
    H, L, priors = _matrices(circuit)
    decoder = BpOsdDecoder(H, error_channel=list(priors), **settings)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    failures = 0
    for syndrome, actual in zip(dets.astype(np.uint8), obs.astype(np.uint8)):
        correction = decoder.decode(syndrome)
        predicted = (L @ correction) % 2
        failures += int(np.any(predicted != actual))
    return shots, failures


def logical_error_rate(circuit: stim.Circuit, shots: int, workers: int = 2,
                       seed: int = 1234, settings: dict = IBM_SETTINGS):
    """Run `shots` shots split over `workers` processes. Returns (shots, failures, seconds)."""
    per = [shots // workers + (i < shots % workers) for i in range(workers)]
    jobs = [(str(circuit), n, seed + i, settings) for i, n in enumerate(per) if n]
    t0 = time.time()
    with mp.Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    total = sum(r[0] for r in results)
    fails = sum(r[1] for r in results)
    return total, fails, time.time() - t0


def wilson(fails: int, shots: int, z: float = 1.96):
    """95% Wilson interval for a failure fraction (honest error bars on small counts)."""
    if shots == 0:
        return (0.0, 1.0)
    ph = fails / shots
    den = 1 + z * z / shots
    centre = (ph + z * z / (2 * shots)) / den
    half = z * np.sqrt(ph * (1 - ph) / shots + z * z / (4 * shots * shots)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


if __name__ == "__main__":
    # quick timing test: python3 decode_bposd.py <which> <p> <shots>
    from circuits import ibm_gross_memory_z, sq_gross_memory_z
    from ibm_filter import ibm_filtered_z
    which, p, shots = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    if which == "ibm":
        c = ibm_filtered_z(ibm_gross_memory_z(p), f"ibm_p{p}")
    else:
        c = sq_gross_memory_z(p)
    n, f, s = logical_error_rate(c, shots)
    lo, hi = wilson(f, n)
    print(f"{which} p={p} shots={n} failures={f} P_L={f/n:.4f} "
          f"[{lo:.4f}, {hi:.4f}] time={s:.1f}s ({s/n*2:.3f} s/shot/core)")
