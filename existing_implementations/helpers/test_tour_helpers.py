"""
test_tour_helpers.py — checks for tour_helpers.py.

What this file is for: makes sure the helpers behind notebooks 01-04 give the
right answers on cases where we know the answer, before anyone shows a graph built on them.
Run from this folder:   python -m pytest -q test_tour_helpers.py
(Uses IBM's Relay-BP clone in <home>/relay; tour.get_repo() clones it if missing.)
"""
import numpy as np
import pytest
import sinter
import stim

import tour_helpers as tour


@pytest.fixture(scope="module")
def repo():
    return tour.get_repo()


def test_catalogue_counts(repo):
    # IBM ships 672 circuits: 168 BB, 168 color, 336 surface, each at 14 noise levels
    cat = tour.catalogue(repo)
    assert len(cat) == 672
    assert cat.groupby("family").size().to_dict() == {"bicycle_bivariate": 168, "color": 168, "surface": 336}
    assert cat.p.nunique() == 14


def test_repetition_fallback_corrects_single_flips():
    # all 8 error patterns of the 3-bit code; only weight-0 and weight-1 should be corrected
    H = np.array([[1, 1, 0], [0, 1, 1]])
    errors = np.array([[(i >> b) & 1 for b in range(3)] for i in range(8)], dtype=np.uint8)
    syn = ((errors @ H.T) % 2).astype(np.uint8)
    true_flip = errors.sum(axis=1) % 2
    ok = tour.repetition_fallback(syn) == true_flip
    assert list(ok) == [w <= 1 for w in errors.sum(axis=1)]


def test_circuit_summary_on_known_circuit():
    # Stim's own d=3 repetition code: 3 data + 2 check qubits = 5 qubits, 1 observable
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3,
                               after_clifford_depolarization=0.01)
    s = tour.circuit_summary({"rep": c}).loc["rep"]
    assert s["qubits"] == 5
    assert s["logical observables"] == 1
    assert s["detectors"] == c.num_detectors


def test_per_round_conversion():
    # 12 rounds, whole-experiment failure 0.1 -> per round 1 - 0.9^(1/12)
    st = sinter.TaskStats(strong_id="x", decoder="d", json_metadata={"r": 12, "p": 0.003},
                          shots=10_000, errors=1_000)
    best, lo, hi = tour.per_round(st)
    assert best == pytest.approx(1 - 0.9 ** (1 / 12), rel=1e-3)
    assert lo < best < hi


def test_gross_filter_keeps_936_of_1728(repo):
    # IBM's Z-detector filter on the gross code: the count we found in our earlier runs
    from testdata import get_test_circuit, filter_detectors_by_basis
    g = get_test_circuit("bicycle_bivariate_144_12_12_memory_Z", distance=12, rounds=12, error_rate=0.003)
    assert (g.num_detectors, filter_detectors_by_basis(g, "Z").num_detectors) == (1728, 936)


def test_select_on_ibm_data(repo):
    # IBM's gross CSV has Relay-BP Z-only results at 5 noise levels
    stats = tour.ibm_data(repo, "bicycle_bivariate_144_12_12_fine_ler_sweep.csv")
    sel = tour.select(stats, decoder="relay-bp", basis="Z", xyz=False)
    assert sorted(s.json_metadata["p"] for s in sel) == [0.001, 0.002, 0.003, 0.004, 0.005]


def test_gate_counts_on_known_circuit():
    # Stim's d=3 repetition code, 1 round: 2 checks x 2 CX = 4 CX; 3 data + 2 check measurements
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=1)
    g = tour.gate_counts(c)
    assert g["CX"] == 4
    assert g["DETECTOR"] == c.num_detectors


def test_wilson_contains_estimate():
    lo, hi = tour.wilson(80, 3000)
    assert lo < 80 / 3000 < hi
    assert tour.wilson(0, 100)[0] == pytest.approx(0, abs=1e-12)


def test_load_surface_results_merges_rows(tmp_path):
    # two Sinter rows for the same (d, p) must be added together
    st = [sinter.TaskStats(strong_id=str(i), decoder="pymatching", json_metadata={"d": 3, "p": 0.003},
                           shots=1000, errors=10) for i in range(2)]
    f = tmp_path / "s.csv"
    f.write_text(sinter.CSV_HEADER + "\n" + "\n".join(s.to_csv_line() for s in st) + "\n")
    df = tour.load_surface_results(f)
    assert (df.shots.item(), df.failures.item()) == (2000, 20)


def test_results_path_keeps_relay_versions_apart(monkeypatch):
    """IBM's and stabilizer-qec's Relay-BP must never share a results file (both are named 'relay-bp')."""
    monkeypatch.setattr(tour, "relay_installed", lambda: False)
    assert tour.results_path("results/bb_relay.csv", ["relay-bp"]).name == "bb_relay_stabilizer_qec.csv"
    assert tour.results_path("results/surface.csv", ["pymatching"]).name == "surface.csv"
    monkeypatch.setattr(tour, "relay_installed", lambda: True)
    assert tour.results_path("results/bb_relay.csv", ["relay-bp"]).name == "bb_relay.csv"
