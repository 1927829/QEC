"""
decoders.py -- one small wrapper per decoder, all with the same interface.

What this file is for
---------------------
Every decoder here is built from a Stim detector error model (DEM) and has one method,
    predict(detectors) -> predicted observable flips   (shots x observables, 0/1)
so the sweep script can hand the SAME shots to every decoder and compare them fairly.

The DEM is turned into three arrays first (by `problem_from_dem`):
    H       detectors x faults   H[d, f] = 1 if fault f lights detector d
    L       observables x faults L[o, f] = 1 if fault f flips logical observable o
    priors  probability of each fault
A matrix decoder returns a guess e for which faults happened; the predicted observable
flips are then L @ e (mod 2).

Sources
-------
- PyMatching (minimum-weight perfect matching): Higgott & Gidney, arXiv:2303.15933.
- Union-Find: Delfosse & Nickerson, arXiv:1709.06218 (here: stabilizer-qec's implementation).
- BP (min-sum belief propagation), BP-OSD: Roffe et al., arXiv:2005.07016; `ldpc` package.
- BP-LSD: Hillmann et al., arXiv:2406.18655; `ldpc` package.
- Relay-BP: Mueller et al., arXiv:2506.01779; IBM `relay_bp` package (optional; falls back to
  stabilizer-qec's own implementation of the same method).
- Ambiguity Clustering: Wolanski & Barber, arXiv:2406.14527 (stabilizer-qec's implementation).
- DEM -> matrices: `beliefmatching.detector_error_model_to_check_matrices`.
"""
from __future__ import annotations

import numpy as np
import stim


def problem_from_dem(dem: stim.DetectorErrorModel):
    """DEM -> (H, L, priors) as scipy sparse matrices and a numpy array.

    Faults that light the same detectors and flip the same observables are merged
    (their probabilities combined), which is what beliefmatching does.
    """
    from beliefmatching import detector_error_model_to_check_matrices
    m = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    return m.check_matrix.tocsc(), m.observables_matrix.tocsc(), np.asarray(m.priors, dtype=float)


class Matching:
    """Minimum-weight perfect matching (PyMatching). Needs every fault to light <= 2 detectors."""
    name = "MWPM (PyMatching)"

    def __init__(self, dem):
        import pymatching
        self.m = pymatching.Matching.from_detector_error_model(dem)

    def predict(self, dets):
        return self.m.decode_batch(dets)


class UnionFind:
    """Union-Find: grows clusters around lit detectors until each can be explained. Fast, slightly less accurate."""
    name = "Union-Find"

    def __init__(self, dem):
        import stabilizer_qec as sq
        self.d = sq.UnionFind(dem)

    def predict(self, dets):
        return np.asarray(self.d.decode_batch(dets), dtype=np.uint8)


class _LdpcBase:
    """Shared code for the three `ldpc` decoders: decode shot by shot, then map faults -> observables."""

    def __init__(self, dem):
        self.H, self.L, self.priors = problem_from_dem(dem)
        self.dec = self.make()
        self.not_converged = 0            # shots where plain BP did not find a valid answer

    def predict(self, dets):
        out = np.zeros((len(dets), self.L.shape[0]), dtype=np.uint8)
        for i, s in enumerate(dets.astype(np.uint8)):
            e = self.dec.decode(s)
            # ldpc sets .converge = False when BP alone failed (OSD / LSD then takes over)
            self.not_converged += int(not getattr(self.dec, "converge", True))
            out[i] = (self.L @ e) % 2
        return out


class BPOnly(_LdpcBase):
    """Plain min-sum BP, 80 iterations, no second step. If BP does not settle, its last guess is used."""
    name = "BP only"

    def make(self):
        from ldpc import BpDecoder
        return BpDecoder(self.H, error_channel=list(self.priors), max_iter=80,
                         bp_method="minimum_sum", ms_scaling_factor=0.625)


class BPOSD(_LdpcBase):
    """BP, then ordered-statistics decoding (combination sweep, order 10) when BP fails. IBM's gross-code settings."""
    name = "BP-OSD"

    def make(self):
        from ldpc import BpOsdDecoder
        return BpOsdDecoder(self.H, error_channel=list(self.priors), max_iter=80, bp_method="minimum_sum",
                            ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=10)


class BPLSD(_LdpcBase):
    """BP, then localised-statistics decoding (clusters solved separately) when BP fails."""
    name = "BP-LSD"

    def make(self):
        from ldpc import BpLsdDecoder
        return BpLsdDecoder(self.H, error_channel=list(self.priors), max_iter=80, bp_method="minimum_sum",
                            ms_scaling_factor=0.625, lsd_method="lsd_cs", lsd_order=10)


class RelayBP:
    """Relay-BP with IBM's default settings. Uses IBM's package if installed, else stabilizer-qec's."""
    name = "Relay-BP"

    def __init__(self, dem):
        self.n_det, self.n_obs = dem.num_detectors, dem.num_observables
        try:
            from relay_bp.stim import sinter_decoders
            self.kind = "IBM relay_bp"
            self.d = sinter_decoders()["relay-bp"].compile_decoder_for_dem(dem=dem)
        except ImportError:
            import stabilizer_qec as sq
            self.kind = "stabilizer-qec RelayBp"
            self.d = sq.RelayBp(dem)

    def predict(self, dets):
        if self.kind == "IBM relay_bp":          # IBM's sinter decoder takes bit-packed shots
            packed = np.packbits(dets, axis=1, bitorder="little")
            out = self.d.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
            return np.unpackbits(out, axis=1, bitorder="little")[:, : self.n_obs]
        return np.asarray(self.d.decode_batch(dets), dtype=np.uint8)


class AmbiguityClustering:
    """Ambiguity Clustering: BP, then split into independent blocks and resolve the uncertain ones."""
    name = "Ambiguity Clustering"

    def __init__(self, dem):
        import stabilizer_qec as sq
        self.d = sq.AmbiguityClustering(dem)

    def predict(self, dets):
        return np.asarray(self.d.decode_batch(dets), dtype=np.uint8)


ALL = [Matching, UnionFind, BPOnly, BPOSD, BPLSD, RelayBP, AmbiguityClustering]
