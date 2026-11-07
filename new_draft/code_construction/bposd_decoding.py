import numpy as np
import stim
from scipy.sparse import csc_matrix
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import load_npz
from pathlib import Path
file_path = Path('spacetime_graph.npz')

if not file_path.exists():
    raise FileNotFoundError("Please Run spacetime_construction first to generate matrices. matrix_checks edited for different BB codes")

spacetime_graph = load_npz('spacetime_graph.npz')
error_probs = np.load('error_probs.npy')
logical_matrix = load_npz("logical_matrix.npz") 

assert isinstance(logical_matrix,csc_matrix)
assert isinstance(spacetime_graph,csc_matrix)

decoder = BpOsdDecoder(
    spacetime_graph,
    error_channel=list(error_probs),
    max_iter=100,
    bp_method="minimum_sum",
    ms_scaling_factor=0.625,
    osd_method="OSD_0",
    osd_order=0
)

H = spacetime_graph.tocsr()
rng = np.random.default_rng()

assert H is not None
assert hasattr(H,"shape")
assert H.shape is not None
assert hasattr(H.shape, "__getitem__")
assert H.shape[1]==len(error_probs)


def sample_shots(shots):
    errors = (rng.random((shots,H.shape[1])) < error_probs).astype(np.uint8)

    syndromes = np.asarray(H @ errors.T).T
    syndromes %= 2

    return syndromes, errors
failures = 0
total = 0

for batch in range(100):
    print(batch)
    syndromes, actual_errors = sample_shots(1)

    for syndrome, actual_error in zip(syndromes,actual_errors):
        estimated_error = decoder.decode(syndrome).copy()
        residual = actual_error ^ estimated_error

        syndrome_bad = np.any(np.asarray(spacetime_graph @ residual).ravel() % 2)

        logical_bad = np.any(np.asarray(logical_matrix @ residual).ravel() % 2)

        failures += int(syndrome_bad or logical_bad)
        total += 1

print("Failure rate:",failures/total)