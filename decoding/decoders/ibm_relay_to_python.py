# (C) Copyright IBM 2025
# Python/Numba adaptation prepared in 2026; this file has been altered from
# the Rust originals. It implements the float64 flooding/nconv configuration
# of https://github.com/trmue/relay
#  The full Apache-2.0 license is embedded below.

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


GAMMA_INTERVALS = {
    "gross_other": (-0.24, 0.66),
    "gross_xx": (-0.16, 0.66),
    "two_gross": (-0.161, 0.815),
}
_F64_MAX = np.finfo(np.float64).max


@njit(cache=True, nogil=True)
def _flooding_kernel(
    row_ptr, row_cols, col_ptr, col_edges, prior, score_weights, syndrome,
    rng, explicit_gammas, max_legs, stop_nconv, gamma0, gamma_low, gamma_high,
    pre_iter, set_max_iter,
):
    n = prior.size
    m = syndrome.size
    n_edges = row_cols.size
    post = prior.copy()
    q = np.empty(n_edges, dtype=np.float64)
    u = np.zeros(n_edges, dtype=np.float64)
    gamma = np.empty(n, dtype=np.float64)
    hard = np.zeros(n, dtype=np.uint8)
    best = hard.copy()
    best_post = post.copy()
    best_score = _F64_MAX
    nconv = 0
    total = 0
    legs = 0
    best_ok = False

    for leg in range(max_legs):
        legs = leg + 1
        for j in range(n):
            if leg == 0:
                gamma[j] = gamma0
            elif explicit_gammas.shape[0] != 0:
                gamma[j] = explicit_gammas[(leg - 1) % explicit_gammas.shape[0], j]
            else:
                gamma[j] = rng.uniform(gamma_low, gamma_high)

        # Original Relay resets BOTH edge-message directions each leg,
        # including legs following successful convergence. Keep post.
        for edge in range(n_edges):
            q[edge] = prior[row_cols[edge]]
            u[edge] = 0.0

        limit = pre_iter if leg == 0 else set_max_iter
        good = False
        for _ in range(limit):
            # Flooding CHECK phase: no q or post changes until all checks
            # have finished. These row computations are independent.
            for a in range(m):
                first = _F64_MAX
                second = _F64_MAX
                min_edge = -1
                negative = syndrome[a] != 0
                for edge in range(row_ptr[a], row_ptr[a + 1]):
                    value = q[edge]
                    negative = negative != np.signbit(value)
                    magnitude = abs(value)
                    if magnitude <= first:
                        second = first
                        first = magnitude
                        min_edge = edge
                    elif magnitude <= second:
                        second = magnitude
                for edge in range(row_ptr[a], row_ptr[a + 1]):
                    value = second if edge == min_edge else first
                    if negative != np.signbit(q[edge]):
                        value = -value
                    u[edge] = value

            # Flooding VARIABLE phase. Prefix/suffix accumulation follows
            # the Rust implementation rather than subtracting from a sum.
            # Memory is applied once per variable, using its OLD posterior.
            for j in range(n):
                if prior[j] == _F64_MAX:
                    bias = prior[j]  # Backend's sentinel for probability 0.
                else:
                    bias = (1.0 - gamma[j]) * prior[j] + gamma[j] * post[j]
                partial = bias
                for k in range(col_ptr[j], col_ptr[j + 1]):
                    edge = col_edges[k]
                    q[edge] = partial
                    partial += u[edge]
                post[j] = partial
                partial = 0.0
                for k in range(col_ptr[j + 1] - 1, col_ptr[j] - 1, -1):
                    edge = col_edges[k]
                    q[edge] += partial
                    partial += u[edge]
                hard[j] = post[j] <= 0.0

            total += 1
            good = True
            for a in range(m):
                parity = 0
                for edge in range(row_ptr[a], row_ptr[a + 1]):
                    parity ^= int(hard[row_cols[edge]])
                if parity != syndrome[a]:
                    good = False
                    break
            if good:
                break  # At most one convergence per leg, duplicates allowed.

        # Match the backend's diagnostic result if ALL legs fail: retain
        # the initial leg's unsuccessful hard decision and posterior.
        if leg == 0:
            best[:] = hard
            best_post[:] = post

        if good:
            nconv += 1
            score = 0.0
            for j in range(n):
                if hard[j] != 0:
                    score += score_weights[j]
            if score < best_score:
                best_score = score
                best[:] = hard
                best_post[:] = post
                best_ok = True
            if nconv >= stop_nconv:
                break

    return best, best_ok, total, legs, nconv, best_score, best_post


@dataclass(frozen=True)
class DecodeResult:
    correction: np.ndarray
    converged: bool
    iterations: int
    legs: int
    n_convergences: int
    score: float
    belief_llr: np.ndarray
    reached_target: bool

    # Aliases for the original package's detailed result interface.
    @property
    def decoding(self):
        return self.correction

    @property
    def success(self):
        return self.converged

    @property
    def posterior_ratios(self):
        return self.belief_llr


class RelayXYZDecoder:
    """Original flooding Relay-BP algorithm, with Tour de Gross defaults.

    max_legs includes the initial leg: 601 means 1 initial + 600 additional.
    profile='gross_xx' refers to the gross X tensor X series in Table 7.
    Additional-leg gamma values are independent across variables and legs,
    but remain fixed during a leg. Ordinary decoding stores O(edges+nodes)
    working data, without a precomputed 600-by-n gamma table.
    """

    def __init__(
        self, parity_check, error_probabilities, *, profile="gross_other",
        max_legs=601, stop_nconv=5, gamma0=0.125,
        pre_iter=80, set_max_iter=60, seed=0, explicit_gammas=None,
    ):
        H = csr_matrix(parity_check, copy=True)
        H.sum_duplicates()
        H.eliminate_zeros()
        H.sort_indices()
        if min(H.shape) < 1 or not np.all(H.data == 1):
            raise ValueError("parity_check must be a nonempty binary matrix.")
        self.H = H.astype(np.uint8)
        p = np.asarray(error_probabilities, dtype=np.float64)
        if p.ndim != 1 or p.size != H.shape[1]:
            raise ValueError("Provide one probability per joint fault column.")
        if not np.all(np.isfinite(p) & (p >= 0) & (p < 1)):
            raise ValueError("Fault probabilities must be finite and in [0, 1).")
        self.error_probabilities = p.copy()
        for name, value in (
            ("max_legs", max_legs), ("stop_nconv", stop_nconv),
            ("pre_iter", pre_iter), ("set_max_iter", set_max_iter),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64).")
        if profile not in GAMMA_INTERVALS:
            raise ValueError(f"profile must be one of {tuple(GAMMA_INTERVALS)}.")
        if not np.isfinite(gamma0):
            raise ValueError("gamma0 must be finite.")
        self.max_legs = int(max_legs)
        self.stop_nconv = int(stop_nconv)
        self.pre_iter = int(pre_iter)
        self.set_max_iter = int(set_max_iter)
        self.gamma0 = float(gamma0)
        self.gamma_dist_interval = GAMMA_INTERVALS[profile]
        self._rng = np.random.default_rng(int(seed))

        # Match the backend's prior calculation and probability-zero sentinel.
        self._prior = np.empty(p.size, dtype=np.float64)
        self._score_weights = np.empty(p.size, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore"):
            for j in range(p.size):
                llr = math.log(float((1.0 - p[j]) / p[j])) if p[j] != 0 else math.inf
                self._prior[j] = _F64_MAX if llr == math.inf else llr
                self._score_weights[j] = llr if math.isfinite(llr) else 0.0

        # Map each variable's incident checks back into the CSR edge arrays.
        # Identical fault columns remain separate; no graph transformation.
        mapping = self.H.copy()
        mapping.data = np.arange(1, H.nnz + 1, dtype=np.int64)
        by_column = mapping.tocsc()
        self._col_ptr = np.ascontiguousarray(by_column.indptr)
        self._col_edges = np.ascontiguousarray(by_column.data - 1)
        if explicit_gammas is None:
            gammas = np.empty((0, H.shape[1]), dtype=np.float64)
        else:
            gammas = np.asarray(explicit_gammas, dtype=np.float64)
            if (
                gammas.ndim != 2 or gammas.shape[0] < 1
                or gammas.shape[1] != H.shape[1] or not np.all(np.isfinite(gammas))
            ):
                raise ValueError("explicit_gammas must be a finite, nonempty legs-by-variables array.")
        self._gammas = np.ascontiguousarray(gammas)

    def _run(self, syndrome, rng, *, warmup=False):
        return _flooding_kernel(
            self.H.indptr, self.H.indices, self._col_ptr, self._col_edges,
            self._prior, self._score_weights, syndrome, rng, self._gammas,
            1 if warmup else self.max_legs,
            1 if warmup else self.stop_nconv,
            self.gamma0, *self.gamma_dist_interval,
            1 if warmup else self.pre_iter, self.set_max_iter,
        )

    def decode(self, syndrome, *, seed=None):
        """Decode one unpacked binary syndrome; return correction + status.

        A per-call seed uses a temporary RNG and preserves the default stream.
        A valid correction remains usable if the convergence target is missed.
        """
        s = np.asarray(syndrome)
        if s.ndim != 1 or s.size != self.H.shape[0] or not np.all((s == 0) | (s == 1)):
            raise ValueError("syndrome must be a binary vector with one entry per detector.")
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
                raise ValueError("seed must be an integer in [0, 2**64).")
        rng = self._rng if seed is None else np.random.default_rng(int(seed))
        hard, ok, total, legs, nconv, score, belief = self._run(
            np.ascontiguousarray(s, dtype=np.uint8), rng,
        )
        return DecodeResult(
            hard, bool(ok), int(total), int(legs), int(nconv), float(score),
            belief, bool(nconv >= self.stop_nconv),
        )

    decode_detailed = decode

    def warm_up(self):
        """Compile the kernel without consuming the decoder's RNG stream."""
        self._run(
            np.zeros(self.H.shape[0], dtype=np.uint8),
            np.random.default_rng(0), warmup=True,
        )


_APACHE_LICENSE = """

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
