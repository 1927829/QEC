import numpy as np
import time
from scipy.sparse import load_npz
from pathlib import Path
from decoders.relay_black_box import AdaptedRelayDecoder
from scipy.optimize import brentq
from scipy.special import expit
if not Path("spacetime_graph.npz").exists():
    raise FileNotFoundError(
        "Run spacetime_construction first to generate the matrices."
    )
H = load_npz("spacetime_graph.npz").tocsr()
error_probs = np.asarray(
    np.load("error_probs.npy"), dtype=np.float64
)
logical_matrix = load_npz("logical_matrix.npz").tocsr()

p=0.001
q = p / 15

multiplicity = np.maximum(
    1,
    np.rint(error_probs / q).astype(np.int64),
)

elementary_to_column = np.repeat(
    np.arange(H.shape[1]),
    multiplicity,
)

num_elementary = len(elementary_to_column)

error_probs = multiplicity * q


decoder = AdaptedRelayDecoder(
    H,
    error_probabilities=error_probs,
    max_legs=61,
    stop_nconv=1,         
    carry_messages=False,  
    seed=271828,
)
decoder.warm_up()

rng = np.random.default_rng()
print("Is the lowest elementary fault set correctly? If so, please hit enter. If not, interrupt.")
input()

def sample_shots(shots, weight=None):
    errors = np.zeros(
        (shots, H.shape[1]),
        dtype=np.uint8,
    )
    for shot in range(shots):
        if weight is None:
            w = int(rng.binomial(num_elementary, q))
        else:
            w = int(weight)
            assert w < num_elementary

        selected = rng.choice(
            num_elementary,
            size=w,
            replace=False,
        )

        columns = elementary_to_column[selected]

        errors[shot] = (np.bincount(columns, minlength=H.shape[1]) % 2).astype(np.uint8)

    syndromes = np.asarray(H @ errors.T).T % 2
    return syndromes.astype(np.uint8), errors
for _ in range(8,45):
    failures = 0
    nonconverged = 0
    total = 0
    for batch in range((80-(_))):
        syndromes, actual_errors = sample_shots(10,_)

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

    print("Failure rate:", failures / total,"at weight",_)
