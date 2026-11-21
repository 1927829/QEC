"""
test_decoders.py -- checks for decoders.py.

What this file is for: before comparing decoders on the surface code, make sure every wrapper reads the
problem correctly and corrects what it must. Run:  python -m pytest -q test_decoders.py   (under a minute)
"""
import numpy as np
import pytest
import stim

import decoders as D

# A distance-5 repetition code memory from Stim: every single fault must be corrected by every decoder
# except plain BP (which may not converge).
CIRCUIT = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                 after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
DEM = CIRCUIT.detector_error_model(decompose_errors=True)


def single_fault_shots():
    """One shot per fault of the DEM: the detectors it flips and the observable it flips."""
    dets, obs = [], []
    for e in DEM.flattened():
        if e.type != "error":
            continue
        d = np.zeros(DEM.num_detectors, dtype=bool); o = np.zeros(DEM.num_observables, dtype=bool)
        for t in e.targets_copy():
            if t.is_relative_detector_id(): d[t.val] ^= True
            elif t.is_logical_observable_id(): o[t.val] ^= True
        dets.append(d); obs.append(o)
    return np.array(dets), np.array(obs)


def test_problem_from_dem_shapes_and_priors():
    H, L, priors = D.problem_from_dem(DEM)
    assert H.shape[0] == DEM.num_detectors and L.shape[0] == DEM.num_observables
    assert H.shape[1] == L.shape[1] == len(priors)
    assert ((priors > 0) & (priors < 0.5)).all()


@pytest.mark.parametrize("cls", [c for c in D.ALL if c is not D.BPOnly])
def test_every_single_fault_is_corrected(cls):
    dets, obs = single_fault_shots()
    pred = np.asarray(cls(DEM).predict(dets)).reshape(len(dets), -1).astype(bool)
    assert (pred == obs).all(), f"{cls.name} failed on a single fault"


@pytest.mark.parametrize("cls", D.ALL)
def test_no_detection_means_no_flip(cls):
    pred = np.asarray(cls(DEM).predict(np.zeros((3, DEM.num_detectors), dtype=bool))).reshape(3, -1)
    assert not pred.any()


def test_bposd_agrees_with_matching_on_most_random_shots():
    dets, obs = CIRCUIT.compile_detector_sampler(seed=3).sample(2000, separate_observables=True)
    m = np.asarray(D.Matching(DEM).predict(dets)).astype(bool)
    o = np.asarray(D.BPOSD(DEM).predict(dets)).astype(bool)
    assert (m == o).all(axis=1).mean() > 0.98


# ---- helpers of investigate_ac.py --------------------------------------------------------------
def test_matched_faults_and_other_class_gap():
    """matched_faults rebuilds a fault set with the shot's syndrome and matching's logical answer;
    other_class_weight finds a fault set in the opposite class that is never lighter (gap >= 0)."""
    import pymatching
    from investigate_ac import matched_faults, other_class_weight
    H, L, pr = D.problem_from_dem(DEM)
    w = np.log((1 - pr) / pr)
    m = pymatching.Matching(H, weights=w, faults_matrix=L)
    dets, _ = CIRCUIT.compile_detector_sampler(seed=7).sample(200, separate_observables=True)
    pred = m.decode_batch(dets)
    best = []
    for s, p in zip(dets, pred):
        e = matched_faults(m, H, s)
        assert ((H @ e) % 2 == s).all() and ((L @ e) % 2 == p).all()
        best.append(w @ e)
    other = other_class_weight(H, L, pr, w, dets, pred)
    assert not np.isnan(other).any()
    assert (other - np.array(best) >= -1e-9).all()
