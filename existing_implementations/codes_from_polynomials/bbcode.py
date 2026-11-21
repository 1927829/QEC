"""
bbcode.py -- bivariate bicycle (BB) codes written from their two polynomials. No QEC library.

What this file is for
---------------------
IBM's gross code circuit file is about 26,000 lines of Stim. It looks like something you could
never write yourself. This file shows that the whole thing follows from two short polynomials,

    A = x^3 + y + y^2        B = y^3 + x + x^2        on a 12 x 6 torus,

and that a memory circuit for it is a few loops. The only imports are numpy (arithmetic) and
stim (to hold the circuit text and simulate it). Everything else -- check matrices, the number
of logical qubits, the logical operators, the Tanner graph, the circuit -- is computed here.

The steps, in the order the file is written:

    1. shifts on the torus          x(a), y(b)                -> permutations of the 72 cells
    2. the code                     BBCode(l, m, A, B)        -> H_X = [A|B], H_Z = [B^T|A^T]
    3. arithmetic mod 2             gf2_rank, gf2_nullspace   -> k, logical operators
    4. the circuit                  BBCode.memory_circuit     -> a stim.Circuit
    5. the picture                  BBCode.draw_tanner        -> Tanner graph on the torus

References
----------
- S. Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory",
  Nature 627, 778 (2024), arXiv:2308.07915: the codes, H_X = [A|B], H_Z = [B^T|A^T], the
  depth-8 syndrome circuit and the circuit noise model.
- The depth-8 gate order used in `_round_depth8` was read off the circuit that the
  stabilizer-qec package generates for the gross code (which follows the paper), layer by layer.
- C. Gidney, "Stim: a fast stabilizer circuit simulator", Quantum 5, 497 (2021).
"""
from __future__ import annotations

import numpy as np
import stim


# =============================================================================================
# 3. Arithmetic mod 2 (placed first because the class below uses it)
# =============================================================================================
def gf2_row_reduce(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Gaussian elimination where 1 + 1 = 0. Returns (reduced matrix, list of pivot columns).

    Row operations are XORs of rows. The number of pivots is the rank.
    """
    M = (np.array(M, dtype=np.uint8) & 1).copy()
    pivots, row = [], 0
    for col in range(M.shape[1]):
        hits = np.flatnonzero(M[row:, col])
        if hits.size == 0:
            continue                                   # no 1 in this column below `row`
        M[[row, row + hits[0]]] = M[[row + hits[0], row]]   # bring a 1 up to the pivot row
        others = np.flatnonzero(M[:, col])
        others = others[others != row]
        M[others] ^= M[row]                            # clear the column everywhere else
        pivots.append(col)
        row += 1
        if row == M.shape[0]:
            break
    return M, pivots


def gf2_rank(M: np.ndarray) -> int:
    """Number of independent rows (mod 2)."""
    return len(gf2_row_reduce(M)[1])


def gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """All solutions v of M v = 0 (mod 2), as the rows of the returned matrix (a basis).

    After row reduction every non-pivot ("free") column gives one basis vector: set that free
    variable to 1, the other free variables to 0, and read the pivot variables off the rows.
    """
    R, pivots = gf2_row_reduce(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in pivots]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for r, pc in enumerate(pivots):
            basis[k, pc] = R[r, f]
    return basis


# =============================================================================================
# 1. Shifts on the torus
# =============================================================================================
def shift(l: int, m: int, a: int, b: int) -> list[int]:
    """The monomial x^a y^b as a permutation of the l*m cells of the torus.

    Cell (i, j) has index i*m + j, with i in 0..l-1 (the x direction) and j in 0..m-1 (y).
    x^a y^b moves cell (i, j) to cell (i + a mod l, j + b mod m). The returned list `perm`
    satisfies perm[cell] = shifted cell. As a matrix this is a permutation matrix with a single
    1 per row, at column perm[row].
    """
    return [((i + a) % l) * m + (j + b) % m for i in range(l) for j in range(m)]


def inverse(perm: list[int]) -> list[int]:
    """The opposite shift. For a permutation matrix the transpose is the inverse."""
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[j] = i
    return inv


# =============================================================================================
# 2. The code
# =============================================================================================
class BBCode:
    """A bivariate bicycle code.

    Parameters
    ----------
    l, m : size of the torus (l cells in x, m in y). There are 2*l*m data qubits.
    A, B : the two polynomials, each a list of three (a, b) pairs meaning x^a y^b.
           The gross code is BBCode(12, 6, A=[(3,0),(0,1),(0,2)], B=[(0,3),(1,0),(2,0)]).

    Layout of the qubits (N = l*m cells, one qubit of each kind per cell):
        L data  : 0 .. N-1          R data  : N .. 2N-1
        X check : 2N .. 3N-1        Z check : 3N .. 4N-1
    """

    def __init__(self, l: int, m: int, A: list[tuple[int, int]], B: list[tuple[int, int]], name: str = ""):
        self.l, self.m, self.N, self.n = l, m, l * m, 2 * l * m
        self.name = name or f"BB({l},{m})"
        self.A_terms, self.B_terms = list(A), list(B)
        # each term of a polynomial is one shift, stored as a permutation
        self.A = [shift(l, m, a, b) for a, b in A]
        self.B = [shift(l, m, a, b) for a, b in B]
        N = self.N
        self.L, self.R = list(range(0, N)), list(range(N, 2 * N))
        self.X, self.Z = list(range(2 * N, 3 * N)), list(range(3 * N, 4 * N))

    # ---- check matrices --------------------------------------------------------------------
    def _poly_matrix(self, terms: list[list[int]], transpose: bool = False) -> np.ndarray:
        """Sum (mod 2) of the permutation matrices of the terms; optionally transposed."""
        M = np.zeros((self.N, self.N), dtype=np.uint8)
        for perm in terms:
            for row, col in enumerate(inverse(perm) if transpose else perm):
                M[row, col] ^= 1
        return M

    def check_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """H_X = [A | B] and H_Z = [B^T | A^T]. Rows = checks, columns = data qubits (L then R).

        X check number i acts on L qubits A_t[i] and R qubits B_t[i], for each term t.
        Z check number j acts on L qubits B_t^{-1}[j] and R qubits A_t^{-1}[j].
        """
        HX = np.hstack([self._poly_matrix(self.A), self._poly_matrix(self.B)])
        HZ = np.hstack([self._poly_matrix(self.B, True), self._poly_matrix(self.A, True)])
        return HX, HZ

    def checks_commute(self) -> bool:
        """Every X check must overlap every Z check on an even number of qubits:
        H_X H_Z^T = A B + B A = 0 mod 2, true because shifts commute."""
        HX, HZ = self.check_matrices()
        return not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()

    @property
    def k(self) -> int:
        """Number of logical qubits: n - rank(H_X) - rank(H_Z)."""
        HX, HZ = self.check_matrices()
        return self.n - gf2_rank(HX) - gf2_rank(HZ)

    # ---- logical operators -----------------------------------------------------------------
    def logical_z(self) -> np.ndarray:
        """k logical Z operators, one per row (1 = a Z on that data qubit).

        A logical Z must commute with every X check (so it is a solution of H_X v = 0), and it
        must not be a product of Z checks (those do nothing to the stored state). So: take a
        basis of the solutions of H_X v = 0 and keep the vectors that are independent of the
        rows of H_Z and of the ones already kept.
        """
        HX, HZ = self.check_matrices()
        kept, rank = [], gf2_rank(HZ)
        stack = HZ.copy()
        for v in gf2_nullspace(HX):
            trial = np.vstack([stack, v])
            if gf2_rank(trial) > rank:                 # v adds something new
                stack, rank = trial, rank + 1
                kept.append(v)
        return np.array(kept, dtype=np.uint8)

    def logical_x(self) -> np.ndarray:
        """k logical X operators (same construction with X and Z exchanged)."""
        HX, HZ = self.check_matrices()
        kept, rank = [], gf2_rank(HX)
        stack = HX.copy()
        for v in gf2_nullspace(HZ):
            trial = np.vstack([stack, v])
            if gf2_rank(trial) > rank:
                stack, rank = trial, rank + 1
                kept.append(v)
        return np.array(kept, dtype=np.uint8)

    # =========================================================================================
    # 4. The circuit
    # =========================================================================================
    def _cx_layer(self, c: stim.Circuit, pairs: list[tuple[int, int]], p: float, idle: list[int] = ()) -> None:
        """One time step: CX on the given (control, target) pairs, then the noise of that step."""
        flat = [q for pair in pairs for q in pair]
        c.append("CX", flat)
        if p:
            c.append("DEPOLARIZE2", flat, p)           # two-qubit error after each gate
            if idle:
                c.append("DEPOLARIZE1", list(idle), p) # qubits that waited in this step
        c.append("TICK")

    def _x_pairs(self, side: str, t: int) -> list[tuple[int, int]]:
        """CX pairs for "every X check does the CX of its term t": control = X check, target = data.
        side "L" uses term t of A, side "R" uses term t of B (H_X = [A|B])."""
        perm, data = (self.A[t], self.L) if side == "L" else (self.B[t], self.R)
        return [(self.X[i], data[perm[i]]) for i in range(self.N)]

    def _z_pairs(self, side: str, t: int) -> list[tuple[int, int]]:
        """CX pairs for the Z checks: control = data, target = Z check.
        side "L" uses term t of B, side "R" uses term t of A (H_Z = [B^T|A^T]): data cell i is
        checked by Z check perm[i]."""
        perm, data = (self.B[t], self.L) if side == "L" else (self.A[t], self.R)
        return [(data[i], self.Z[perm[i]]) for i in range(self.N)]

    def _round_simple(self, c: stim.Circuit, p: float) -> tuple[str, str]:
        """One syndrome round, simplest valid order: all X checks, then all Z checks (14 steps).

        Always correct, for any code of this family, because X and Z gates are never mixed.
        The price is waiting: a qubit not in the current CX layer gets idle noise.
        Returns the order in which the two kinds of checks were measured.
        """
        data = self.L + self.R
        c.append("RX", self.X); c.append("R", self.Z)
        if p:
            c.append("Z_ERROR", self.X, p); c.append("X_ERROR", self.Z, p)
            c.append("DEPOLARIZE1", data, p)
        c.append("TICK")
        for side, other in (("L", self.R), ("R", self.L)):       # X checks: 3 terms on L, 3 on R
            for t in range(3):
                self._cx_layer(c, self._x_pairs(side, t), p, idle=other + self.Z)
        c.append("MX", self.X, p)
        for side, other in (("L", self.R), ("R", self.L)):       # Z checks
            for t in range(3):
                self._cx_layer(c, self._z_pairs(side, t), p, idle=other)
        c.append("M", self.Z, p)
        if p:
            c.append("DEPOLARIZE1", data, p)
        c.append("TICK")
        return ("X", "Z")

    # The depth-8 order of Bravyi et al.: in each of steps 2-6 the X checks and the Z checks act
    # at the same time on different halves of the data. (side, term index) per step; A1,A2,A3 and
    # B1,B2,B3 are terms 0,1,2 of A and B.
    DEPTH8_X = [None, ("L", 1), ("R", 1), ("R", 0), ("R", 2), ("L", 0), ("L", 2)]   # steps 1..7
    DEPTH8_Z = [("R", 0), ("R", 2), ("L", 0), ("L", 1), ("L", 2), ("R", 1), None]

    def schedule_conflicts(self, x_order: list | None = None, z_order: list | None = None) -> int:
        """Count X/Z check pairs for which an interleaved schedule measures the wrong thing.

        When X and Z checks run at the same time, an X check and a Z check that share data qubits
        are measured correctly only if, among their shared qubits, the number where the X check's
        CNOT comes first is EVEN. (Each such qubit swaps the order of two anticommuting gate
        actions; an odd number of swaps leaves a sign that makes the outcome random.)
        Returns the number of (X check, Z check) pairs that break the rule; 0 means the schedule is valid.
        Default: the depth-8 order.
        """
        x_order = self.DEPTH8_X if x_order is None else x_order
        z_order = self.DEPTH8_Z if z_order is None else z_order
        tx, tz = {}, {}                                # check -> {data qubit: time step}
        for step, (xs, zs) in enumerate(zip(x_order, z_order)):
            if xs:
                for chk, dq in self._x_pairs(*xs):
                    tx.setdefault(chk, {})[dq] = step
            if zs:
                for dq, chk in self._z_pairs(*zs):
                    tz.setdefault(chk, {})[dq] = step
        bad = 0
        for xa, xt in tx.items():
            for za, zt in tz.items():
                shared = xt.keys() & zt.keys()
                if shared and sum(xt[q] < zt[q] for q in shared) % 2:
                    bad += 1
        return bad

    def _round_depth8(self, c: stim.Circuit, p: float) -> tuple[str, str]:
        """One syndrome round in 8 time steps (Bravyi et al.). Z check qubits must already be reset.

        step 1    prepare X checks;         Z checks: first CX          (L data waits)
        steps 2-6 X checks and Z checks each do one CX layer
        step 7    measure Z checks;         X checks: last CX           (R data waits)
        step 8    measure X checks; reset Z checks for the next round   (all data waits)
        """
        c.append("RX", self.X)
        if p:
            c.append("Z_ERROR", self.X, p)
        for step in range(7):
            pairs, idle = [], []
            if step == 6:                                  # Z checks are measured in step 7
                c.append("M", self.Z, p)
            if self.DEPTH8_X[step]:
                pairs += self._x_pairs(*self.DEPTH8_X[step])
            if self.DEPTH8_Z[step]:
                pairs += self._z_pairs(*self.DEPTH8_Z[step])
            if step == 0:
                idle = self.L
            if step == 6:
                idle = self.R
            self._cx_layer(c, pairs, p, idle)
        c.append("MX", self.X, p)
        c.append("R", self.Z)
        if p:
            c.append("X_ERROR", self.Z, p)
            c.append("DEPOLARIZE1", self.L + self.R, p)
        c.append("TICK")
        return ("Z", "X")

    def memory_circuit(self, rounds: int, p: float = 0.0, schedule: str = "depth8",
                       z_detectors_only: bool = True) -> stim.Circuit:
        """Z-basis memory experiment: data in |0>, `rounds` syndrome rounds, measure the data.

        p         uniform circuit noise: every reset, gate, measurement and idle step fails with
                  probability p (p = 0 gives the noiseless circuit).
        schedule  "depth8" (Bravyi et al.) or "simple" (all X checks, then all Z checks).
        z_detectors_only  write detectors only for the Z checks (what a Z-memory decoder uses);
                  False also writes the X-check detectors, as IBM's files do.

        The circuit uses a Stim REPEAT block for rounds 2..rounds, so the file stays short.
        Observables: one per logical Z operator, read from the final data measurement.
        """
        N, body = self.N, (self._round_depth8 if schedule == "depth8" else self._round_simple)
        c = stim.Circuit()
        for cell in range(N):                              # coordinates, only used for drawing
            i, j = divmod(cell, self.m)
            c.append("QUBIT_COORDS", [self.X[cell]], [2 * i, 2 * j]);     c.append("QUBIT_COORDS", [self.L[cell]], [2 * i, 2 * j + 1])
            c.append("QUBIT_COORDS", [self.R[cell]], [2 * i + 1, 2 * j]); c.append("QUBIT_COORDS", [self.Z[cell]], [2 * i + 1, 2 * j + 1])
        c.append("R", self.L + self.R)
        if p:
            c.append("X_ERROR", self.L + self.R, p)
        if schedule == "depth8":                           # depth-8 rounds expect Z checks reset
            c.append("R", self.Z)
            if p:
                c.append("X_ERROR", self.Z, p)
        c.append("TICK")

        # --- round 1: Z checks should read 0 on |0...0>, so each is a detector on its own -------
        order = body(c, p)
        z_pos, x_pos = order.index("Z"), order.index("X")  # which half of a round's 2N records
        rec = lambda pos, j, rounds_back=0: stim.target_rec(-2 * N * (rounds_back + 1) + pos * N + j)
        for j in range(N):
            c.append("DETECTOR", [rec(z_pos, j)])

        # --- rounds 2..rounds: every check is compared with its own value one round earlier ----
        if rounds > 1:
            rep = stim.Circuit()
            body(rep, p)
            for j in range(N):
                rep.append("DETECTOR", [rec(z_pos, j), rec(z_pos, j, 1)])
            if not z_detectors_only:
                for j in range(N):
                    rep.append("DETECTOR", [rec(x_pos, j), rec(x_pos, j, 1)])
            c.append(stim.CircuitRepeatBlock(rounds - 1, rep))

        # --- final readout: measure all data; rebuild each Z check from it ----------------------
        c.append("M", self.L + self.R, p)                  # 2N new records: data qubit q is rec -2N + q
        _, HZ = self.check_matrices()
        for j in range(N):
            data_recs = [stim.target_rec(-2 * N + q) for q in np.flatnonzero(HZ[j])]
            last_round = stim.target_rec(-2 * N - 2 * N + z_pos * N + j)
            c.append("DETECTOR", data_recs + [last_round])
        for obs, row in enumerate(self.logical_z()):
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2 * N + q) for q in np.flatnonzero(row)], obs)
        return c

    def detector_error_model(self, rounds: int, p: float, schedule: str = "depth8") -> stim.DetectorErrorModel:
        """The decoder's input for this memory experiment, built from the FLATTENED circuit.

        Why flattened: with a REPEAT block in the circuit, Stim (checked with 1.16) lists some
        faults several times instead of merging those with identical symptoms (2,376 duplicates for
        the 12-round gross code: 11,160 entries instead of 8,784). BP-based decoders then treat the
        copies as separate possibilities and perform worse (BP-OSD at p = 0.004: 7.3% instead of
        3.1% failures on the same shots). Flattening first gives the merged model.
        """
        return self.memory_circuit(rounds, p, schedule).flattened().detector_error_model(decompose_errors=False)

    # =========================================================================================
    # 5. The picture
    # =========================================================================================
    _OFFSET = {"X": (-0.22, -0.22), "L": (-0.22, 0.22), "R": (0.22, -0.22), "Z": (0.22, 0.22)}
    _COLOUR = {"X": "#d1495b", "Z": "#2a6fdb", "L": "#333333", "R": "#999999"}

    def _pos(self, cell: int, kind: str) -> tuple[float, float]:
        return cell // self.m + self._OFFSET[kind][0], cell % self.m + self._OFFSET[kind][1]

    def draw_tanner(self, ax, x_checks: list[int] = (), z_checks: list[int] = (), all_x_terms: list[tuple[str, int]] = ()):
        """Tanner graph on the torus. Each cell holds an L and an R data qubit (dots) and an X and a
        Z check (squares).

        x_checks / z_checks : cells whose check should be drawn with its 6 edges, one colour per term.
        all_x_terms         : e.g. [("L", 1)] draws that term's edge for EVERY X check at once.
        Edges that leave the picture wrap around to the other side (dashed part).
        """
        l, m = self.l, self.m
        for cell in range(self.N):
            for kind in "XZLR":
                ax.scatter(*self._pos(cell, kind), s=16, color=self._COLOUR[kind],
                           marker="s" if kind in "XZ" else "o", zorder=3)
        colours = ["#e07a1f", "#2e9e5b", "#8a5cc2", "#1f9ec9", "#b5651d", "#c2185b"]   # A1 A2 A3 B1 B2 B3

        def edge(p0, p1, colour, lw=2.0, alpha=1.0):
            (x0, y0), (x1, y1) = p0, p1
            dx = (x1 - x0 + l / 2) % l - l / 2             # shortest way round the torus
            dy = (y1 - y0 + m / 2) % m - m / 2
            ax.plot([x0, x0 + dx], [y0, y0 + dy], color=colour, lw=lw, alpha=alpha, zorder=2)
            if abs(x0 + dx - x1) > 1e-9 or abs(y0 + dy - y1) > 1e-9:      # wrapped: show the other end too
                ax.plot([x1 - dx, x1], [y1 - dy, y1], color=colour, lw=lw, alpha=alpha, zorder=2, ls="--")

        for cell in x_checks:
            for t in range(3):
                edge(self._pos(cell, "X"), self._pos(self.A[t][cell], "L"), colours[t])
                edge(self._pos(cell, "X"), self._pos(self.B[t][cell], "R"), colours[3 + t])
        for cell in z_checks:
            for t in range(3):
                edge(self._pos(cell, "Z"), self._pos(inverse(self.B[t])[cell], "L"), colours[3 + t], 1.5, 0.7)
                edge(self._pos(cell, "Z"), self._pos(inverse(self.A[t])[cell], "R"), colours[t], 1.5, 0.7)
        for side, t in all_x_terms:
            perm, colour = (self.A[t], colours[t]) if side == "L" else (self.B[t], colours[3 + t])
            for cell in range(self.N):
                edge(self._pos(cell, "X"), self._pos(perm[cell], side), colour, 1.0, 0.8)
        ax.set_xlim(-0.8, l - 0.2); ax.set_ylim(-0.8, m - 0.2); ax.set_aspect("equal")
        ax.set_xticks(range(l)); ax.set_yticks(range(m)); ax.grid(alpha=0.25)
        ax.set_xlabel(f"x  ({l} cells, wraps around)"); ax.set_ylabel(f"y  ({m} cells, wraps around)")
        return colours


def term_name(a: int, b: int) -> str:
    """(3, 0) -> 'x^3', (0, 1) -> 'y', (0, 0) -> '1'."""
    part = lambda s, e: "" if e == 0 else (s if e == 1 else f"{s}^{e}")
    return (part("x", a) + part("y", b)) or "1"


# The codes of Bravyi et al., Table 3 (the ones used here), plus the toric code for comparison.
def gross_code() -> BBCode:
    return BBCode(12, 6, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)], name="gross [[144,12,12]]")


def bb72_code() -> BBCode:
    return BBCode(6, 6, A=[(3, 0), (0, 1), (0, 2)], B=[(0, 3), (1, 0), (2, 0)], name="BB [[72,12,6]]")
