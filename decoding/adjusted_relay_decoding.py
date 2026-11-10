import numpy as np
import time
from scipy.sparse import load_npz
from pathlib import Path
from relay_black_box import AdaptedRelayDecoder
if not Path("spacetime_graph.npz").exists():
    raise FileNotFoundError(
        "Run spacetime_construction first to generate the matrices."
    )

H = load_npz("spacetime_graph.npz").tocsr()
error_probs = np.asarray(
    np.load("error_probs.npy"), dtype=np.float64
)
logical_matrix = load_npz("logical_matrix.npz").tocsr()

decoder = AdaptedRelayDecoder(H,error_probabilities=error_probs,max_legs=61,seed=271828,)
decoder.warm_up()

rng = np.random.default_rng()


def sample_shots(shots):
    errors = (
        rng.random((shots, H.shape[1])) < error_probs
    ).astype(np.uint8)

    syndromes = np.asarray(H @ errors.T).T % 2
    return syndromes.astype(np.uint8), errors


failures = 0
nonconverged = 0
total = 0
time_spent_converging = 0
time_1=time.time()
for batch in range(20):
    print("Batch:", batch + 1)
    syndromes, actual_errors = sample_shots(10)

    for syndrome, actual_error in zip(syndromes, actual_errors):
        result = decoder.decode(
            np.ascontiguousarray(syndrome, dtype=np.uint8)
        )
        total += 1

        if not result.converged:
            failures += 1
            nonconverged += 1
            continue

        estimated_error = result.correction
        residual = actual_error ^ estimated_error

        syndrome_bad = np.any(
            np.asarray(H @ residual).ravel() % 2
        )
        logical_bad = np.any(
            np.asarray(logical_matrix @ residual).ravel() % 2
        )

        failures += int(syndrome_bad or logical_bad)

print("Failures:", failures, "/", total)
print("Nonconverged:", nonconverged)
print("Failure rate:", failures / total)
time_2=time.time()
print("Time Elapsed: ",time_2-time_1,"seconds.")
