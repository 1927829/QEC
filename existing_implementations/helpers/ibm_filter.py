"""
ibm_filter.py -- keep only the detectors a Z-memory decoder should see.

What this file is FOR (plain words):
    IBM's circuit files define detectors for BOTH the X checks and the Z checks.
    For a Z-basis memory, only the Z-check detectors carry information about the
    logical Z's (bit-flip errors). IBM's own workflow drops the rest with
    `filter_detectors_by_basis(circuit, "Z")` before decoding. That function and
    its helper `detect_data_qubits` are copied below from IBM's Relay-BP repo so
    you do not need to clone it, and the filtered circuit is cached to disk
    (the filter takes a few seconds per circuit).

License notice for the two copied functions:
    (C) Copyright IBM 2025. Licensed under the Apache License, Version 2.0
    (see data/IBM_LICENSE.txt). Sources:
      - filter_detectors_by_basis: tests/testdata/utils.py
      - detect_data_qubits:        python/relay_bp/stim/noise.py
    MODIFIED by Kanishka Padamata / Claude (24 Oct 2026): copied into this file,
    comments added, the unused `qubits=` argument removed. Logic unchanged.
"""

from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"   # filtered circuits are saved in ../cache/ (one folder above helpers/)


# ---------------------------------------------------------------- IBM code --
def detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Data qubits = qubits measured exactly once (only at the final readout).

    IBM's own note: a heuristic that works for typical memory circuits.
    """
    times_measured = [0] * circuit.num_qubits
    for inst in circuit:
        # count only noiseless measurements, as IBM does
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for q in inst.targets_copy():
                times_measured[q.qubit_value] += 1
    return [q for q, n in enumerate(times_measured) if n == 1]


def filter_detectors_by_basis(circuit: stim.Circuit, basis: str) -> stim.Circuit:
    """Return `circuit` without the detectors that cannot see `basis` errors.

    Method (IBM): inject a test error (X_ERROR for basis "Z") on every data
    qubit after every reset/measurement layer of the NOISELESS circuit, and
    mark each detector that ever flips. Unmarked detectors are dropped.
    """
    assert basis in ("X", "Z")
    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()
    noiseless = circuit.without_noise()

    # reference: detector values with no injected error
    ref_dets, _ = noiseless.compile_detector_sampler().sample(1, separate_observables=True)
    ref_dets = ref_dets[0, :]
    sensitive = np.full(len(ref_dets), False, dtype=bool)

    to_test = detect_data_qubits(noiseless)
    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            # copy the noiseless circuit and insert one certain error
            injected = stim.Circuit()
            injected += noiseless
            injected.insert(inst_idx, stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]))
            dets, _ = injected.compile_detector_sampler().sample(1, separate_observables=True)
            sensitive[np.where(ref_dets != dets[0, :])] = True

        # move to just after the next reset or measurement and test again
        to_test = []
        for inst in noiseless[inst_idx:]:
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    # rebuild the circuit, keeping only sensitive detectors
    filtered = stim.Circuit()
    det_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            keep = sensitive[det_idx]
            det_idx += 1
            if not keep:
                continue
        filtered.append(inst)
    return filtered
# ------------------------------------------------------------ end IBM code --


def ibm_filtered_z(circuit: stim.Circuit, tag: str) -> stim.Circuit:
    """Return `circuit` with non-Z detectors removed (cached under cache/<tag>)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}_filteredZ.stim"
    if path.exists():
        return stim.Circuit.from_file(str(path))
    out = filter_detectors_by_basis(circuit, "Z")
    out.to_file(str(path))
    return out
