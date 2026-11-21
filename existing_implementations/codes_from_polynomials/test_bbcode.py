"""
test_bbcode.py -- checks for bbcode.py.

What this file is for: every claim the walkthrough makes about the code written from the
polynomials is tested here on cases where the right answer is known.
Run:  python -m pytest -q test_bbcode.py        (about a minute)
stabilizer-qec is used in two tests only, as an outside reference; they are skipped without it.
"""
import numpy as np
import pytest
import stim

import bbcode as bb


def test_gf2_rank_and_nullspace():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])      # third row = sum of the first two
    assert bb.gf2_rank(M) == 2
    ns = bb.gf2_nullspace(M)
    assert ns.shape == (1, 3) and not ((M @ ns.T) % 2).any()


def test_shift_is_a_permutation_and_inverse_undoes_it():
    p = bb.shift(12, 6, 3, 1)
    assert sorted(p) == list(range(72))
    assert [bb.inverse(p)[p[i]] for i in range(72)] == list(range(72))


@pytest.mark.parametrize("code, n, k", [(bb.gross_code(), 144, 12), (bb.bb72_code(), 72, 12)])
def test_parameters_match_the_paper(code, n, k):
    HX, HZ = code.check_matrices()
    assert code.n == n and code.k == k and code.checks_commute()
    assert set(HX.sum(axis=1)) == {6} and set(HX.sum(axis=0)) == {3}     # weight-6 checks, 3 per qubit
    assert set(HZ.sum(axis=1)) == {6} and set(HZ.sum(axis=0)) == {3}


def test_toric_code_has_two_logical_qubits():
    # A = 1 + x, B = 1 + y on a torus is the toric code (the third term is repeated to keep 3 terms
    # per polynomial out of the picture: use a 2-term version through the matrices directly)
    l, m = 6, 6
    A = sum(np.eye(l * m, dtype=int)[bb.shift(l, m, a, b)] for a, b in [(0, 0), (1, 0)]) % 2
    B = sum(np.eye(l * m, dtype=int)[bb.shift(l, m, a, b)] for a, b in [(0, 0), (0, 1)]) % 2
    HX, HZ = np.hstack([A, B]), np.hstack([B.T, A.T])
    assert not ((HX @ HZ.T) % 2).any()
    assert 2 * l * m - bb.gf2_rank(HX) - bb.gf2_rank(HZ) == 2


def test_logical_operators():
    code = bb.gross_code()
    HX, HZ = code.check_matrices()
    LZ, LX = code.logical_z(), code.logical_x()
    assert LZ.shape == LX.shape == (12, 144)
    assert not ((HX.astype(int) @ LZ.T) % 2).any()        # logical Z commutes with every X check
    assert not ((HZ.astype(int) @ LX.T) % 2).any()
    assert bb.gf2_rank(np.vstack([HZ, LZ])) == bb.gf2_rank(HZ) + 12      # and is not a product of checks
    assert bb.gf2_rank((LX.astype(int) @ LZ.T) % 2) == 12                # X and Z logicals pair up fully


@pytest.mark.parametrize("schedule", ["depth8", "simple"])
@pytest.mark.parametrize("z_only", [True, False])
def test_noiseless_circuit_is_silent(schedule, z_only):
    c = bb.gross_code().memory_circuit(4, 0.0, schedule, z_detectors_only=z_only)
    dets, obs = c.compile_detector_sampler(seed=1).sample(100, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_circuit_counts_match_ibm():
    c = bb.gross_code().memory_circuit(12, 0.004, "depth8", z_detectors_only=False)
    cx = sum(len(i.targets_copy()) // 2 for i in c.flattened() if i.name == "CX")
    assert cx == 10368 and c.num_qubits == 288 and c.num_observables == 12
    assert c.num_detectors == 72 + 11 * 144 + 72          # round 1 Z only, 11 rounds both, final Z
    assert bb.gross_code().memory_circuit(12, 0.004).num_detectors == 936     # what IBM's filter keeps
    assert len(str(c).splitlines()) < 1000                # REPEAT keeps the file short (IBM: 26,411)


def test_noisy_circuit_has_a_valid_error_model():
    # Stim refuses to build a DEM if any detector is not deterministic without noise
    for schedule in ("depth8", "simple"):
        dem = bb.gross_code().memory_circuit(3, 0.003, schedule).detector_error_model()
        assert dem.num_errors > 0


def test_matrices_equal_stabilizer_qec():
    sq = pytest.importorskip("stabilizer_qec")
    HX, HZ = bb.gross_code().check_matrices()
    sHX, sHZ = sq.BivariateBicycleCode("gross").check_matrices()
    assert np.array_equal(HX, np.asarray(sHX)) and np.array_equal(HZ, np.asarray(sHZ))


def test_depth8_matches_stabilizer_qec_circuit_size():
    sq = pytest.importorskip("stabilizer_qec")
    ref = stim.Circuit(str(sq.BivariateBicycleCode("gross").memory_circuit(12, 0.004)))
    ours = bb.gross_code().memory_circuit(12, 0.004, "depth8")
    count = lambda c, name: sum(len(i.targets_copy()) for i in c.flattened() if i.name == name)
    for name in ("CX", "DEPOLARIZE1", "DEPOLARIZE2", "R", "RX"):
        assert count(ours, name) == count(ref, name), name
    assert ours.num_detectors == ref.num_detectors == 936


def test_flattened_error_model_merges_duplicate_faults():
    # the REPEAT version of the circuit gives an error model with unmerged duplicates; ours must not
    code = bb.gross_code()
    with_repeat = code.memory_circuit(12, 0.004).detector_error_model(decompose_errors=False)
    ours = code.detector_error_model(12, 0.004)
    assert with_repeat.num_errors > ours.num_errors == 8784
    symptoms = [frozenset(str(t) for t in e.targets_copy()) for e in ours if e.type == "error"]
    assert len(symptoms) == len(set(symptoms))          # every fault listed once


def test_depth8_order_satisfies_the_interleaving_rule_and_a_scrambled_one_does_not():
    code = bb.gross_code()
    assert code.schedule_conflicts() == 0
    z = list(code.DEPTH8_Z); z[1], z[4] = z[4], z[1]          # swap two Z steps
    assert code.schedule_conflicts(code.DEPTH8_X, z) > 0
