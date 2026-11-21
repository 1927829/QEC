"""
test_common.py -- checks for common.py.

Run:  python -m pytest -q test_common.py
The last two tests need IBM's repository (cloned to ~/relay on first use, about 110 MB).
"""
import pytest
import stim

import common


def test_wilson_interval():
    lo, hi = common.wilson(80, 3000)
    assert lo < 80 / 3000 < hi
    assert common.wilson(0, 100)[0] == pytest.approx(0, abs=1e-12)


def test_per_round_conversion():
    best, lo, hi = common.per_round(1000, 10_000, 12)          # P = 0.1 over 12 rounds
    assert best == pytest.approx(1 - 0.9 ** (1 / 12)) and lo < best < hi


def test_describe_and_operation_counts():
    # Stim's d = 3 repetition code, 1 round: 3 data + 2 check qubits, 4 CX
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=1, after_clifford_depolarization=0.01)
    info = common.describe(c)
    assert info["qubits"] == 5 and info["CX gates"] == 4 and info["logical observables"] == 1
    assert common.operation_counts(c)["CX"] == 4


def test_ibm_repository_is_complete():
    repo = common.get_relay_repo()
    assert sum(1 for _ in (repo / "tests" / "testdata").rglob("*.stim")) == 672


def test_gross_circuit_and_filter():
    g = common.ibm_circuit("bicycle_bivariate_144_12_12_memory_Z", 12, 0.003)
    gz = common.ibm_circuit("bicycle_bivariate_144_12_12_memory_Z", 12, 0.003, z_detectors_only=True)
    assert (g.num_detectors, gz.num_detectors) == (1728, 936)
    assert common.describe(g)["CX gates"] == 10368
