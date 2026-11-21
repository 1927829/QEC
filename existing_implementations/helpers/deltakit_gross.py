"""
deltakit_gross.py -- the gross code built with Riverlane's Deltakit, as a noisy Stim circuit.

What this file is FOR (plain words):
    A third, independent implementation of the gross code to compare against IBM's
    circuits and stabilizer-qec's. Deltakit builds the code from the same polynomials
    (A = x^3 + y + y^2, B = y^3 + x + x^2 on a 12 x 6 torus, Bravyi et al.
    arXiv:2308.07915), writes a memory experiment, and adds a noise model.
    Everything here runs locally; no Deltakit account is needed.

    What runs where:
      - local (no account):  BivariateBicycleCode, css_code_memory_circuit,
                             QPU(...).compile_and_add_noise_to_circuit, as_stim_circuit
      - cloud (needs token): Deltakit's own circuit generator (client=...) and its
                             decoders BPOSDecoder / ACDecoder (see the notebook)

    Deltakit API used (deltakit 0.10.2; docs https://deltakit-docs.riverlane.com):
      deltakit_explorer.codes.BivariateBicycleCode(param_l, param_m, m_A_powers, m_B_powers)
          A = x^a + y^b + y^c for m_A_powers=[a, b, c]; B = y^d + x^e + x^f for [d, e, f]
      deltakit_explorer.codes.css_code_memory_circuit(code, num_rounds, logical_basis)
      deltakit_explorer.qpu.QPU(qubits, noise_model=ToyNoise(p)).compile_and_add_noise_to_circuit
          ToyNoise(p): depolarising noise p after gates and on idles, measurement flips
          (Deltakit's simple uniform model; close to, not identical to, IBM's).

    Detector filter: like IBM's circuits, Deltakit's Z memory defines detectors for both
    X and Z checks. IBM's filter (ibm_filter.py) guesses data qubits in a way that drops
    Deltakit's final-readout detectors (checked 7 Nov 2026: it leaves 45 single faults
    that flip a logical with no detector). `z_detectors_only` below classifies each
    detector from the circuit itself instead.
"""

import numpy as np
import stim

GROSS = dict(param_l=12, param_m=6, m_A_powers=[3, 1, 2], m_B_powers=[3, 1, 2])


def deltakit_gross_code():
    """The [[144, 12, 12]] gross code as a Deltakit BivariateBicycleCode object."""
    from deltakit_explorer.codes import BivariateBicycleCode
    return BivariateBicycleCode(**GROSS)


def deltakit_gross_memory_z(p: float, rounds: int = 12, noise: str = "toy") -> stim.Circuit:
    """Noisy Z-basis memory circuit of the gross code, built locally by Deltakit.

    noise: "toy" -> ToyNoise(p) (Deltakit's uniform depolarising model)
           "si1000" -> SI1000Noise(p) (superconducting-inspired model, Gidney et al.)
    Returns a plain stim.Circuit (Deltakit ships its own Stim build, so we re-parse
    the text to get the standard type the rest of our code uses).
    """
    from deltakit_circuit.gates import PauliBasis
    from deltakit_explorer.codes import css_code_memory_circuit
    from deltakit_explorer.qpu import QPU, SI1000Noise, ToyNoise

    code = deltakit_gross_code()
    circuit = css_code_memory_circuit(code, num_rounds=rounds, logical_basis=PauliBasis.Z)
    model = ToyNoise(p=p) if noise == "toy" else SI1000Noise(p=p)
    noisy = QPU(qubits=circuit.qubits, noise_model=model).compile_and_add_noise_to_circuit(circuit)
    return stim.Circuit(str(noisy.as_stim_circuit()))


def deltakit_gross_memory_z_noiseless(rounds: int = 12) -> stim.Circuit:
    """Deltakit's Z memory circuit with NO noise (to add our own noise model)."""
    from deltakit_circuit.gates import PauliBasis
    from deltakit_explorer.codes import css_code_memory_circuit
    circuit = css_code_memory_circuit(deltakit_gross_code(), num_rounds=rounds, logical_basis=PauliBasis.Z)
    return stim.Circuit(str(circuit.as_stim_circuit()))


def add_uniform_circuit_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Bravyi et al.'s / IBM's "uniform circuit" noise, added to any noiseless circuit.

    The model (arXiv:2308.07915, Methods): every operation fails with probability p:
      CX            -> DEPOLARIZE2(p) after it
      reset to |0>  -> X_ERROR(p) after it;  reset to |+> -> Z_ERROR(p) after it
      measurement   -> its result flipped with probability p
      idle qubit    -> DEPOLARIZE1(p) in every layer (between TICKs) where it does nothing
    Deltakit's ToyNoise differs: idle noise is p/10 and resets are noiseless, which is why
    its error rates come out lower at the same p (checked 7 Nov 2026).
    """
    out = stim.Circuit()
    all_q = set(range(circuit.num_qubits))
    layer: list[stim.CircuitInstruction] = []

    def flush(layer):
        busy = set()
        for inst in layer:
            name, t = inst.name, [x.value for x in inst.targets_copy() if x.is_qubit_target]
            if name in ("M", "MZ", "MX"):
                out.append(name, t, p)                          # measurement flip
            else:
                out.append(inst)
            if name in ("CX", "CNOT", "CZ"):
                out.append("DEPOLARIZE2", t, p)
            elif name in ("R", "RZ"):
                out.append("X_ERROR", t, p)
            elif name == "RX":
                out.append("Z_ERROR", t, p)
            elif name in ("H", "I"):
                out.append("DEPOLARIZE1", t, p)                 # 1-qubit gate error / explicit idle
            busy.update(t)
        idle = sorted(all_q - busy)
        if busy and idle:
            out.append("DEPOLARIZE1", idle, p)

    for inst in circuit.flattened():
        if inst.name == "TICK":
            flush(layer); layer = []
            out.append("TICK")
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            flush(layer); layer = []                            # annotations close the layer
            out.append(inst)
        else:
            layer.append(inst)
    flush(layer)
    return out


def z_detectors_only(circuit: stim.Circuit) -> stim.Circuit:
    """Keep only the detectors a Z-memory decoder needs (Z checks + final data readout).

    How each detector is classified, using only the circuit:
      - each measurement record is mapped to the qubit it measured;
      - a detector that uses a measurement of a DATA qubit (a qubit that is a CX
        control on Z-check ancillas... simpler: measured only once, at the end) is a
        final Z detector -> keep;
      - otherwise it compares measurements of one ANCILLA. In a CSS syndrome circuit an
        X-check ancilla is the CONTROL of its CX gates, a Z-check ancilla is the TARGET.
        Keep Z-check detectors, drop X-check ones.
    """
    flat = circuit.flattened()
    meas_qubit = []                 # measurement index -> qubit
    n_meas_per_qubit = {}
    cx_control, cx_target = set(), set()
    for inst in flat:
        name = inst.name
        if name in ("CX", "CNOT", "ZCX"):
            t = [x.value for x in inst.targets_copy()]
            cx_control.update(t[0::2])
            cx_target.update(t[1::2])
        elif name in ("M", "MZ", "MX", "MR", "MRX", "MRZ"):
            for x in inst.targets_copy():
                meas_qubit.append(x.value)
                n_meas_per_qubit[x.value] = n_meas_per_qubit.get(x.value, 0) + 1
    data = {q for q, n in n_meas_per_qubit.items() if n == 1}
    x_anc = cx_control - cx_target - data      # controls only: X-check ancillas
    out = stim.Circuit()
    m_count = 0
    for inst in flat:
        if inst.name == "DETECTOR":
            qs = {meas_qubit[m_count + t.value] for t in inst.targets_copy()}   # rec[-k]
            if qs & data or not (qs & x_anc):
                out.append(inst)
            continue
        out.append(inst)
        if inst.name in ("M", "MZ", "MX", "MR", "MRX", "MRZ"):
            m_count += len(inst.targets_copy())
    return out


def undetectable_single_faults(circuit: stim.Circuit) -> int:
    """How many single faults flip a logical without firing any detector (should be 0)."""
    dem = circuit.detector_error_model(decompose_errors=False)
    n = 0
    for inst in dem.flattened():
        if inst.type == "error":
            t = [x for x in inst.targets_copy() if not x.is_separator()]
            if t and all(x.is_logical_observable_id() for x in t):
                n += 1
    return n
