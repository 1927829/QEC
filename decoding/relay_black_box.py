"""
No idea what the code is doing here, I'll just treat it as a black box
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numba import njit


@njit(cache=True)
def _relay_kernel(
    indptr, cols, prior, syndrome, seed, sets,
    nphys, order, split, schedule, carry, aux_scale,
    stop_nconv, gamma0, gamma_low, gamma_high,
    pre_iter, set_max_iter, clip,
):
    n = len(prior)
    m = len(syndrome)
    edges = len(cols)

    zero = True
    for r in range(m):
        if syndrome[r]:
            zero = False
            break

    # All-zero correction is already globally most likely here.
    if zero and np.all(prior >= 0):
        return (
            np.zeros(n, np.uint8),
            True, 0, 0, 1, 0.0, False, prior.copy(),
        )

    np.random.seed(seed)

    post = prior.copy()
    check = np.zeros(edges)
    proposal = np.zeros(edges)
    bias = np.zeros(n)
    gamma = np.zeros(n)

    total = 0
    saturated = False
    best = np.zeros(n, np.uint8)
    best_belief = prior.copy()
    best_score = np.inf
    nconv = 0

    for leg in range(sets + 1):
        for c in range(n):
            if leg == 0:
                g = gamma0
            else:
                g = gamma_low + (
                    gamma_high - gamma_low
                ) * np.random.random()

            if c >= nphys:
                g *= aux_scale

            gamma[c] = g

        if leg == 0 or not carry:
            check[:] = 0.0

        iterations_this_leg = pre_iter if leg == 0 else set_max_iter

        for _ in range(iterations_this_leg):
            # Freeze the memory contribution for this full sweep.
            for c in range(n):
                bias[c] = (
                    (1 - gamma[c]) * prior[c]
                    + gamma[c] * post[c]
                )

            post[:] = bias

            for k in range(edges):
                post[cols[k]] += check[k]

            for c in range(n):
                if abs(post[c]) > clip:
                    saturated = True
                post[c] = min(max(post[c], -clip), clip)

            groups = 2 if schedule == 1 else m

            for group in range(groups):
                if schedule == 1:
                    start = 0 if group == 0 else split
                    end = split if group == 0 else m
                else:
                    start = group
                    end = group + 1

                for q in range(start, end):
                    r = order[q]
                    a = indptr[r]
                    b = indptr[r + 1]

                    small = 1e300
                    second = 1e300
                    imin = -1
                    negative = int(syndrome[r])

                    for k in range(a, b):
                        v = post[cols[k]] - check[k]

                        if abs(v) > clip:
                            saturated = True

                        v = min(max(v, -clip), clip)
                        av = abs(v)
                        negative += int(v < 0)

                        if av < small:
                            second = small
                            small = av
                            imin = k
                        elif av < second:
                            second = av

                    for k in range(a, b):
                        v = min(
                            max(post[cols[k]] - check[k], -clip),
                            clip,
                        )

                        value = second if k == imin else small

                        if (negative - int(v < 0)) % 2:
                            value = -value

                        if abs(value) > clip:
                            saturated = True

                        proposal[k] = min(max(value, -clip), clip)

                for q in range(start, end):
                    r = order[q]

                    for k in range(indptr[r], indptr[r + 1]):
                        c = cols[k]

                        post[c] += proposal[k] - check[k]
                        check[k] = proposal[k]

                        if schedule == 2:
                            if abs(post[c]) > clip:
                                saturated = True
                            post[c] = min(max(post[c], -clip), clip)

                if schedule == 1:
                    for c in range(n):
                        if abs(post[c]) > clip:
                            saturated = True
                        post[c] = min(max(post[c], -clip), clip)

            total += 1

            # Check whether the hard decision matches the syndrome.
            good = True

            for r in range(m):
                parity = 0

                for k in range(indptr[r], indptr[r + 1]):
                    parity ^= int(post[cols[k]] < 0)

                if parity != syndrome[r]:
                    good = False
                    break

            if good:
                nconv += 1

                # Negative log likelihood, ignoring a common constant.
                score = 0.0
                for c in range(n):
                    if post[c] < 0:
                        score += prior[c]

                if score < best_score:
                    best_score = score

                    for c in range(n):
                        best[c] = int(post[c] < 0)

                    best_belief = post.copy()

                if nconv >= stop_nconv:
                    return (
                        best, True, total, leg + 1,
                        nconv, best_score, saturated, best_belief,
                    )

                # Count at most one convergence per leg.
                # Start another leg with newly sampled gammas.
                break

    # A valid correction remains usable even if the target was not reached.
    return (
        best, nconv > 0, total, sets + 1,
        nconv, best_score, saturated, best_belief,
    )


@dataclass(frozen=True)
class DecodeResult:
    correction: np.ndarray
    converged: bool
    iterations: int
    legs: int
    hit_llr_bound: bool
    belief_llr: np.ndarray
    n_convergences: int
    score: float
    reached_target: bool


class AdaptedRelayDecoder:
    """
    Custom serial layered Relay-style decoder.

    stop_nconv counts successful legs, including repeated corrections.
    The most likely candidate is selected using the fault priors.

    max_legs includes the initial leg.
    carry_messages=False resets check messages between legs while
    retaining posterior memory.
    """

    def __init__(
        self,
        parity_check,
        error_probabilities,
        *,
        max_legs=301,
        stop_nconv=5,
        carry_messages=True,
        gamma0=0.125,
        gamma_dist_interval=(-0.24, 0.66),
        pre_iter=80,
        set_max_iter=60,
        max_data_value=1e6,
        seed=271828,
    ):
        h = sp.csr_matrix(parity_check, copy=True)
        h.sum_duplicates()
        h.eliminate_zeros()
        h.sort_indices()

        if (
            h.shape[0] < 1
            or h.shape[1] < 1
            or not np.all(h.data == 1)
        ):
            raise ValueError(
                "parity_check must be a nonempty binary matrix."
            )

        self.H = h.astype(np.uint8)

        p = np.asarray(error_probabilities, dtype=np.float64)

        if p.ndim != 1 or len(p) != self.H.shape[1]:
            raise ValueError(
                "Provide one fault probability per matrix column."
            )

        if not np.all(np.isfinite(p) & (p > 0) & (p < 1)):
            raise ValueError(
                "Fault probabilities must be finite and between 0 and 1."
            )

        for name, value in (
            ("max_legs", max_legs),
            ("stop_nconv", stop_nconv),
            ("pre_iter", pre_iter),
            ("set_max_iter", set_max_iter),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if not isinstance(carry_messages, (bool, np.bool_)):
            raise ValueError("carry_messages must be a boolean.")

        interval = np.asarray(gamma_dist_interval, dtype=float)

        if (
            interval.shape != (2,)
            or not np.all(np.isfinite(interval))
            or interval[0] >= interval[1]
        ):
            raise ValueError(
                "gamma_dist_interval must contain finite increasing bounds."
            )

        if (
            not np.isfinite(gamma0)
            or not np.isfinite(max_data_value)
            or max_data_value <= 0
        ):
            raise ValueError(
                "gamma0 must be finite; max_data_value must be "
                "finite and positive."
            )

        self.max_legs = int(max_legs)
        self.stop_nconv = int(stop_nconv)
        self.carry_messages = bool(carry_messages)
        self.gamma0 = float(gamma0)
        self.gamma_dist_interval = tuple(interval)
        self.pre_iter = int(pre_iter)
        self.set_max_iter = int(set_max_iter)
        self.max_data_value = float(max_data_value)

        self.error_probabilities = p.copy()
        self._prior = np.ascontiguousarray(np.log((1 - p) / p))
        self._order = np.arange(self.H.shape[0], dtype=np.int32)

        self.seed = int(seed)
        self._calls = 0

    def decode(self, syndrome, *, seed=None):
        s = np.asarray(syndrome)

        if (
            s.ndim != 1
            or len(s) != self.H.shape[0]
            or not np.all((s == 0) | (s == 1))
        ):
            raise ValueError(
                "syndrome must be a binary vector with one entry per row."
            )

        s = np.ascontiguousarray(s, dtype=np.uint8)

        used_seed = (
            self.seed + self._calls if seed is None else int(seed)
        ) % (2**32)

        self._calls += 1

        out = _relay_kernel(
            self.H.indptr,
            self.H.indices,
            self._prior,
            s,
            used_seed,
            self.max_legs - 1,
            self.H.shape[1],
            self._order,
            0,
            2,
            self.carry_messages,
            0.5,
            self.stop_nconv,
            self.gamma0,
            *self.gamma_dist_interval,
            self.pre_iter,
            self.set_max_iter,
            self.max_data_value,
        )

        (
            hard, ok, iterations, legs,
            nconv, score, saturated, belief,
        ) = out

        if ok and not np.array_equal((self.H @ hard) % 2, s):
            raise RuntimeError(
                "Returned correction does not match the syndrome."
            )

        return DecodeResult(
            correction=hard,
            converged=bool(ok),
            iterations=int(iterations),
            legs=int(legs),
            hit_llr_bound=bool(saturated),
            belief_llr=belief,
            n_convergences=int(nconv),
            score=float(score),
            reached_target=int(nconv) >= self.stop_nconv,
        )

    def warm_up(self):
        """Compile without consuming a default decoding seed."""
        self.decode(
            np.zeros(self.H.shape[0], dtype=np.uint8),
            seed=0,
        )
        self._calls -= 1