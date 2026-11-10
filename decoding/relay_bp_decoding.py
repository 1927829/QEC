#preferably, run on google colab with !pip install relay_bp
#you could download relay_bp on desktop if u want, but takes up a lot of space
#when running on colab, attach the 3 .npv and .npz files
import numpy as np
import stim
from scipy.sparse import csc_matrix
from relay_bp import RelayDecoderF32
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

decoder = RelayDecoderF32(
    spacetime_graph.tocsr(),
    error_priors=np.asarray(error_probs, dtype=np.float64),
    gamma0=0.1,
    pre_iter=80,
    num_sets=60,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=1,
)

H = spacetime_graph.tocsr()
rng = np.random.default_rng()

assert H is not None
assert hasattr(H,"shape")
assert H.shape is not None
assert hasattr(H.shape, "__getitem__")
assert H.shape[1]==len(error_probs)


def sample_shots(shots):
    assert H.shape is not None
    errors = (rng.random((shots,H.shape[1])) < error_probs).astype(np.uint8)

    syndromes = np.asarray(H @ errors.T).T
    syndromes %= 2

    return syndromes, errors
failures = 0
total = 0

for batch in range(5):
    print(batch)
    syndromes, actual_errors = sample_shots(1)

    for syndrome, actual_error in zip(syndromes,actual_errors):
        estimated_error = np.asarray(
    decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8)),
    dtype=np.uint8,
).copy()

        residual = actual_error ^ estimated_error

        syndrome_bad = np.any(np.asarray(spacetime_graph @ residual).ravel() % 2)

        logical_bad = np.any(np.asarray(logical_matrix @ residual).ravel() % 2)

        failures += int(syndrome_bad or logical_bad)
        total += 1
        

print("Failure rate:",failures/total)