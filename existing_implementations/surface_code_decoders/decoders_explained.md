# How the decoders work

**What this document is for.** It explains each decoder used in this project, with the maths, in one place: what problem a
decoder solves, how each algorithm attacks it, where each one works and where it fails, and how they compare on the
rotated surface code (`surface_code_decoders.ipynb`, results in `results_surface_decoders.csv`). Every claim about how an
algorithm works is taken from its paper, cited per section.

Contents:
1. The decoding problem
2. Minimum-weight perfect matching (PyMatching)
3. Union-Find
4. Belief propagation (BP)
5. BP + ordered statistics decoding (BP-OSD)
6. BP + localised statistics decoding (BP-LSD)
7. Relay-BP
8. Ambiguity Clustering
9. Side by side
10. Results on the surface code

---

## 1. The decoding problem

A noisy circuit has many places where something can go wrong. Stim turns it into a **detector error model** (DEM): a list of
independent faults $j = 1, \dots, n$, each with

- a probability $p_j$,
- the set of **detectors** it flips (a detector is a parity of measurements that is always 0 without noise),
- the set of **logical observables** it flips.

Collect these into two 0/1 matrices:

$$H \in \mathbb{F}_2^{m \times n}\ (H_{ij} = 1 \text{ if fault } j \text{ flips detector } i), \qquad L \in \mathbb{F}_2^{k \times n}\ (L_{oj} = 1 \text{ if fault } j \text{ flips observable } o).$$

In one shot, a set of faults $e \in \mathbb{F}_2^n$ happens. We see only the **syndrome** $s = He \pmod 2$ (which detectors
fired). We want the logical effect $Le$. A decoder is a function $s \mapsto \hat\ell$, and the shot fails if $\hat\ell \neq Le$.

**What would be optimal.** Many different $e$ give the same $s$. They split into classes by their logical effect. The best
possible decoder picks the most likely class:

$$\hat\ell = \arg\max_{\ell} \sum_{e:\ He = s,\ Le = \ell} \ \prod_j p_j^{e_j} (1 - p_j)^{1 - e_j}.$$

This sum is exponentially large, so every practical decoder approximates it. Most look instead for the single most likely
fault set,

$$\hat e = \arg\min_{e:\ He = s} \ \sum_j w_j e_j, \qquad w_j = \log\frac{1 - p_j}{p_j},$$

and output $\hat\ell = L \hat e$. The weight $w_j$ is the cost of assuming fault $j$ happened: rare faults cost more. This is
**minimum-weight decoding**. It is not the same as the optimal class decision, because it ignores **degeneracy**: two
different fault sets with the same syndrome and the same logical effect are equally good answers, and their probabilities
should add up. That difference matters most for qLDPC codes (sections 4–8).

**Graph or hypergraph.** If every column of $H$ has at most two 1s, each fault is an **edge** between two detectors (or between
a detector and a "boundary" node): the problem is a graph. Surface codes under circuit noise are graph-like: with only the
Z detectors kept (as here) no fault flips more than two; with both X and Z detectors, Stim splits the larger faults into
graph-like pieces (`decompose_errors=True`). Bivariate bicycle codes, including the gross
code, have faults that flip 3 to 6 detectors (Z detectors only) that cannot be split: the problem is a **hypergraph**, and matching does not apply.

---

## 2. Minimum-weight perfect matching (PyMatching)

**Idea.** On a graph, a fault set $e$ with syndrome $s$ is a set of edges whose endpoints of odd degree are exactly the fired
detectors. Every such set decomposes into paths that pair up the fired detectors (or connect one to the boundary). So
minimum-weight decoding becomes: **pair up the fired detectors so that the total length of the connecting paths is as small as
possible**, with edge lengths $w_j$.

**Steps.**
1. Build the graph: a node per detector plus a boundary node; an edge per fault, with weight $w_j = \log((1-p_j)/p_j)$. Parallel
   edges are merged (probabilities combined).
2. For every pair of fired detectors $u, v$, the cost of pairing them is the shortest-path distance $d(u, v)$ (Dijkstra). Also
   $d(u, \text{boundary})$.
3. Find a **minimum-weight perfect matching** on the fired detectors: a set of disjoint pairs covering every fired detector (or
   matching it to the boundary) with minimum total cost. Edmonds' **blossom algorithm** solves this exactly in polynomial time.
4. The correction is the union of the shortest paths; $\hat\ell = L\hat e$ is the parity of observable-flipping edges on them.

**PyMatching's version.** *Sparse blossom* (Higgott & Gidney 2023) runs the blossom algorithm directly on the detector graph,
growing regions around fired detectors, instead of building the complete graph of step 2. That is why it decodes surface code
shots in microseconds.

**Strengths and limits.** Exact minimum-weight decoding on graphs, and the standard decoder for surface codes. Within the
graph model it is close to optimal: the gap to the optimal class decision (degeneracy) is small for the surface code. It
needs a graph: on the gross code it does not apply. It also loses correlations between the X and Z parts of a Y error unless
extended (correlated matching).

*Sources:* J. Edmonds, *Paths, trees, and flowers*, Canad. J. Math. 17, 449 (1965); E. Dennis, A. Kitaev, A. Landahl,
J. Preskill, *Topological quantum memory*, J. Math. Phys. 43, 4452 (2002); O. Higgott and C. Gidney, *Sparse Blossom*,
arXiv:2303.15933.

---

## 3. Union-Find

**Idea.** Replace the global optimisation of matching with local growth. Fired detectors grow clusters; a cluster stops when it
can be explained on its own; then any correction inside each cluster is found by peeling.

**Steps** (Delfosse & Nickerson).
1. Every fired detector starts a cluster.
2. A cluster is **valid** if it contains an even number of fired detectors, or touches the boundary (then any parity can be
   absorbed). Invalid clusters **grow** by half an edge in every direction per step.
3. When two clusters touch, they **merge**. Merging is done with a union-find (disjoint-set) data structure, which makes each
   merge almost constant time.
4. When every cluster is valid, take a spanning forest of each cluster and **peel** it: walk from the leaves inwards, adding an
   edge to the correction whenever its leaf is a fired detector, and toggling the inner node.

**Complexity and accuracy.** $O(n\,\alpha(n))$, with $\alpha$ the inverse Ackermann function ($\le 3$ in practice). It is
provably correct for up to $(d-1)/2$ errors. On the toric code it reaches a threshold of about 9.9% with perfect measurements
(MWPM: about 10.3%) and about 2.6% with faulty ones. In short: almost as accurate as matching, and simpler to put in hardware.
Like matching, it needs a graph. (Here we use `stabilizer-qec`'s implementation, with weighted growth.)

*Source:* N. Delfosse and N. H. Nickerson, *Almost-linear time decoding algorithm for topological codes*, Quantum 5, 595
(2021), arXiv:1709.06218.

---

## 4. Belief propagation (BP)

**Idea.** Work on the **factor graph** (Tanner graph) of $H$: a node per fault $j$, a node per detector $i$, an edge where
$H_{ij} = 1$. Instead of searching for one answer, estimate for every fault the probability that it happened, given the
syndrome, by passing messages along the edges.

**Quantities** (log-likelihood ratios; positive means "probably did not happen"):

- prior of fault $j$: $\lambda_j = \log\frac{1 - p_j}{p_j}$,
- message from fault $j$ to detector $i$: $\mu_{j \to i}$,
- message from detector $i$ to fault $j$: $\nu_{i \to j}$.

**Updates**, repeated for $t = 1, 2, \dots$ up to a maximum:

$$\mu_{j \to i} = \lambda_j + \sum_{i' \in N(j) \setminus i} \nu_{i' \to j}$$

$$\nu_{i \to j} = (-1)^{s_i} \, 2 \operatorname{atanh} \prod_{j' \in N(i) \setminus j} \tanh\frac{\mu_{j' \to i}}{2} \qquad \text{(sum-product)}$$

$$\nu_{i \to j} = (-1)^{s_i} \, \alpha \prod_{j' \in N(i) \setminus j} \operatorname{sign} \mu_{j' \to i} \ \cdot \min_{j' \in N(i) \setminus j} |\mu_{j' \to i}| \qquad \text{(min-sum, scaling } \alpha \le 1)$$

The detector message says: "given what the other faults on my edges are doing, and whether I fired, here is what fault $j$
should be". Then each fault combines everything it heard:

$$\Lambda_j = \lambda_j + \sum_{i \in N(j)} \nu_{i \to j}, \qquad \hat e_j = [\Lambda_j < 0].$$

BP stops as soon as $H \hat e = s$ ("converged"), otherwise at the iteration limit.

**When it works.** BP computes exact marginals on a graph with no cycles. On quantum codes it is cheap (linear in the number of
edges per iteration) and fully parallel.

**When it fails, and why that matters for quantum codes.**
- **Short cycles.** Messages come back to where they started and reinforce or contradict themselves; beliefs oscillate.
- **Degeneracy.** If two different fault sets explain the syndrome equally well (e.g. two equal-length paths around a square in
  the surface code), BP is pulled towards both, gives each fault about 50%, and the hard decision $\hat e$ satisfies neither. In
  classical LDPC codes this is rare; in quantum codes it is built in, because stabilizers make many errors equivalent.
- **Trapping sets.** Small subgraphs where BP converges to a wrong fixed point.

That is why every qLDPC decoder below is "BP plus something": BP narrows down where the faults probably are, and a second step
finishes the job when BP does not converge. In our surface code runs, plain BP failed to converge on 6% (d = 3, p = 0.002) up
to 90% (d = 9, p = 0.005) of shots, and its error rate grows with distance: it has no threshold on its own.

*Sources:* R. G. Gallager, *Low-density parity-check codes* (1963); D. Poulin and Y. Chung, *On the iterative decoding of
sparse quantum codes*, Quant. Inf. Comput. 8, 987 (2008), arXiv:0801.1241; J. Roffe et al., arXiv:2005.07016.

---

## 5. BP + ordered statistics decoding (BP-OSD)

**Idea.** When BP does not converge, its beliefs are still useful: they rank the faults from "very likely" to "very unlikely".
OSD uses that ranking to solve $He = s$ exactly with linear algebra.

**OSD-0.**
1. Sort the columns of $H$ by BP's soft output, most likely fault first (most negative $\Lambda_j$).
2. Going down that list, pick the first $r = \operatorname{rank} H$ linearly independent columns: the **information set** $S$.
   The rest is $T$.
3. Set $e_T = 0$ and solve $H_S e_S = s$. $H_S$ is an invertible $r \times r$ matrix, so the solution is unique (Gaussian
   elimination over $\mathbb{F}_2$, $O(n^3)$ in the worst case).

$\hat e = (e_S, 0)$ always satisfies the syndrome. It uses only faults BP believed in most.

**Higher order.** OSD-0 assumed none of the less likely faults in $T$ happened. Higher-order OSD also tries a few of them:
for each small pattern $e_T$, recompute $e_S = H_S^{-1}(s + H_T e_T)$ and keep the solution with the lowest weight
$\sum_j w_j e_j$.
- **OSD-E** (exhaustive): all $2^w$ patterns on the first $w$ columns of $T$.
- **OSD-CS** (combination sweep, used here and by IBM with order 10): all weight-1 patterns on $T$, plus all weight-2 patterns
  among the first $w = 10$ columns of $T$.

**Strengths and limits.** Accurate on all qLDPC codes; the standard baseline in the literature. The Gaussian elimination is the
cost: it is slow, sequential and hard to build in hardware for real-time decoding. In our surface code runs it was the slowest
decoder at large d (about 17 ms per shot at d = 9 on 2 cores), and as accurate as matching.

*Sources:* M. P. C. Fossorier and S. Lin, *Soft-decision decoding of linear block codes based on ordered statistics*, IEEE
Trans. Inf. Theory 41, 1379 (1995); P. Panteleev and G. Kalachev, *Degenerate quantum LDPC codes with good finite length
performance*, Quantum 5, 585 (2021), arXiv:1904.02703; J. Roffe et al., arXiv:2005.07016 (`ldpc` package).

---

## 6. BP + localised statistics decoding (BP-LSD)

**Idea.** Do OSD's linear algebra only where it is needed. A fault far away from every fired detector cannot help explain the
syndrome, so the matrix can be split into small independent pieces around the fired detectors, each solved separately (and in
parallel).

**Steps** (Hillmann et al.).
1. Run BP. If it converges, stop.
2. Seed a **cluster** at each fired detector.
3. Grow each cluster by adding fault columns adjacent to it, in order of BP's soft output (most likely first). Each cluster
   keeps its own submatrix and runs Gaussian elimination **incrementally** ("on-the-fly elimination") as columns are added.
4. A cluster stops growing when its local syndrome lies in the column span of its local submatrix: it can be explained on its
   own.
5. Clusters that touch are merged (their eliminations combined).
6. Each cluster's local system is solved (OSD-0 inside the cluster, or a higher-order variant such as LSD-CS); the corrections are
   combined.

**Strengths and limits.** Below threshold, the clusters are small, so the work is much less than global OSD and parallelisable;
the authors report accuracy matching BP-OSD on the codes they test. In our surface code runs with `ldpc`'s implementation and
order 10, BP-LSD matched BP-OSD at d = 3 but was 20–40% worse at d = 5–9 (and similar to BP-OSD in speed on 2 cores). We have
not tuned its settings; this is a finding to check, not a verdict on the method.

*Source:* T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, J. Roffe, *Localized statistics decoding for quantum
low-density parity-check codes*, Nature Communications (2025), arXiv:2406.18655.

---

## 7. Relay-BP

**Idea.** Fix BP's oscillations without any linear algebra, so that the decoder stays fast and parallel enough for hardware.
Two ingredients: **memory** (each fault remembers its previous belief, which damps oscillation and can break symmetric ties)
and a **relay** of BP runs with randomly chosen memory strengths, each starting where the last one stopped.

**Memory BP.** In ordinary BP the bias of fault $j$ is fixed at its prior $\Lambda_j(0) = \lambda_j$. In memory BP it is updated
every iteration:

$$\Lambda_j(t) = (1 - \gamma_j)\,\Lambda_j(0) + \gamma_j\, M_j(t - 1),$$

where $M_j(t-1)$ is the fault's marginal (posterior) from the previous iteration and $\gamma_j$ its **memory strength**.
$\gamma_j = 0$ is plain BP; a constant $0 < \gamma < 1$ is Mem-BP; a different $\gamma_j$ per fault is "disordered" memory BP
(DMem-BP). Negative $\gamma_j$ (pushing away from the previous belief) turn out to be essential: they help BP escape a
symmetric, undecided state.

**The relay.**
1. **Leg 0:** DMem-BP with every $\gamma_j = \gamma_0$, starting from the priors, up to $T_0$ iterations.
2. **Legs $r = 1, 2, \dots$:** draw each $\gamma_j$ independently and uniformly from $[\gamma_{\min}, \gamma_{\max}]$. Reset the
   biases to the priors, but start the marginals from the last leg's final marginals, so information carries forward. Run up to
   $T_r$ iterations.
3. A leg whose hard decision satisfies $H\hat e = s$ is a **solution**; record its weight $\sum_j w_j \hat e_j$.
4. Stop after $S$ solutions or $R$ legs; return the lowest-weight solution.

IBM's settings for the gross code (used here through `relay_bp`'s defaults): $T_0 = 80$, $T_r = 60$, $\gamma_0 \approx 0.1$,
$[\gamma_{\min}, \gamma_{\max}] = [-0.24, 0.66]$, $S = 5$, up to about 300 legs. The interval was tuned for the gross code; for
the surface code the paper uses $\gamma_0 = 0.35$ and $[-0.254, 0.985]$. We used the gross code defaults on the surface code,
so our surface code numbers for Relay-BP are not tuned.

**Strengths and limits.** On bivariate bicycle codes the paper reports Relay-BP clearly beating BP-OSD-CS order 10 in accuracy,
with only BP-style message passing, so it can run on FPGAs or ASICs. On a 2-core CPU it is slow (many legs per hard shot): in our
gross code test, 140 s for 300 shots vs 20 s for BP-OSD.

*Source:* T. Müller, T. Alexander, M. E. Beverland, M. Bühler, B. R. Johnson, T. Maurer, D. Vandeth, *Improved belief
propagation is sufficient for real-time decoding of quantum memory*, arXiv:2506.01779; code: github.com/trmue/relay.

---

## 8. Ambiguity Clustering (AC)

**Idea.** Like LSD, split the problem into independent clusters after BP. The difference: AC does not need one fault set per
cluster, only each cluster's **logical effect**. Usually that is determined uniquely ("unambiguous"), with no search at all; a
small search is done only in the rare ambiguous cluster.

**Steps** (Wolanski & Barber).
1. Run BP (9 sum-product iterations in the paper) to get posteriors $p_j$. If BP explains the syndrome, stop.
2. **Initial solution.** Repeatedly pivot on a pair (fired check row, adjacent fault column) that is not yet a pivot, choosing the
   column with the largest $p_j$, and update the syndrome. Stop when every fired check is a pivot row: this gives an initial
   solution. Unlike OSD, the pivot order follows the current syndrome, not a fixed ranking.
3. **Grow clusters.** $K = \kappa n$ more times, add the most likely column adjacent to an already-pivoted row: if it has a 1 in a
   new row, pivot on it and start a new block; otherwise add it and merge any blocks it touches. The result is a set of blocks =
   connected clusters of the Tanner graph. Larger $\kappa$ explores more and is more accurate (the paper uses $\kappa \le 0.05$).
4. **Analyse each cluster** $C_i e_i = \sigma_i$ with logical matrix $L_i$:
   - **unambiguous**, if every logical row is a combination of the cluster's check rows, $L_i = R_i C_i$: then the logical effect
     is $\lambda_i = R_i \sigma_i$, computed directly;
   - **ambiguous**, otherwise: write the solutions as $e_i = (f_i, g_i)$ with $f_i = \sigma_i + B_i g_i$, enumerate all $g_i$ of
     weight $\le 2$, add up the probability of the candidates that flip each observable and of those that don't, and choose the
     larger. This is a small step towards the optimal class decision of section 1, not just the single most likely fault set.
5. Output $\lambda = \sum_i \lambda_i$.

**Strengths and limits.** On bivariate bicycle codes at 0.3% circuit noise, the paper reports BP-OSD accuracy at up to 27×
the speed (135 µs per round for the gross code on an M2 CPU). In our tests it was the fastest of the BP-based decoders and
among the most accurate, on the gross code (8.7% vs BP-OSD 16.3%, 300 shots, $p = 0.005$) and on the surface code (at or below
matching at every d and p tested). It is the decoder behind Deltakit's cloud `ACDecoder`; here we use `stabilizer-qec`'s
implementation.

*Source:* S. Wolanski and B. Barber, *Ambiguity Clustering: an accurate and efficient decoder for qLDPC codes*,
arXiv:2406.14527.

---

## 9. Side by side

| Decoder | Works on | What it finds | Second step after BP | Parallel / hardware | Cost per shot |
|---|---|---|---|---|---|
| MWPM (PyMatching) | graphs (surface, repetition, colour after splitting) | exact min-weight fault set | – | partly (sparse blossom) | very low |
| Union-Find | graphs | a valid fault set inside small clusters | – | yes | lowest |
| BP | any $H$ | marginals; answer only if it converges | – | yes | low |
| BP-OSD | any $H$ | min-weight within BP's top choices | global Gaussian elimination | no | high |
| BP-LSD | any $H$ | the same, cluster by cluster | local, incremental elimination | yes | medium |
| Relay-BP | any $H$ | lowest-weight of several converged BP runs | none (more BP, with memory) | yes | medium–high on CPU |
| Ambiguity Clustering | any $H$ | logical effect per cluster, with a small probability sum when ambiguous | local elimination + small search | partly | low–medium |

The ideas line up as one story. Matching and Union-Find exploit the graph structure the surface code gives. qLDPC codes do
not have it, so BP is the natural starting point, but degeneracy and short cycles stop BP from converging. The four BP decoders
differ only in what they do then: solve globally (OSD), solve locally (LSD), keep doing BP with memory (Relay-BP), or reason
about each cluster's logical effect (AC).

---

## 10. Results on the surface code

Setup (`run_sweep.py`): IBM's rotated surface code Z-memory circuits (uniform circuit noise, $d$ rounds) for
$d = 3, 5, 7, 9$ and $p = 0.002 \dots 0.008$; Z detectors only (as IBM does); one Stim DEM per point with
`decompose_errors=True`; 20,000 shots sampled once per point and **the same shots given to every decoder**.

See `surface_code_decoders.ipynb` for the plots. The main findings:

Results from `run_sweep.py`: $d$ = 3, 5, 7, 9; $p$ = 0.002 to 0.008; 20,000 shots per point, the same shots for every decoder.
Mean of (decoder's error rate ÷ matching's) over the points with at least 20 failures:

| Decoder | d = 3 | d = 5 | d = 7 | d = 9 | ms per shot at d = 9 |
|---|---|---|---|---|---|
| MWPM (PyMatching) | 1.00 | 1.00 | 1.00 | 1.00 | 0.013 |
| Union-Find | 1.00 | 1.09 | 1.09 | 1.14 | 0.029 |
| BP only | 2.7 | 9.1 | 38 | 270 | 1.9 |
| BP-OSD | 0.99 | 1.00 | 0.99 | 1.02 | 10.9 |
| BP-LSD | 1.01 | 1.31 | 1.29 | 1.37 | 2.1 |
| Relay-BP | 1.02 | 1.02 | 1.06 | 1.17 | 11.4 |
| Ambiguity Clustering | 0.99 | 0.97 | 0.94 | 0.94 | 0.88 |

1. **Plain BP has no threshold on the surface code.** It does not converge on 6–29% of shots at $d = 3$ and 35–99% at $d = 9$,
   and its error rate rises with distance (at $p = 0.004$: 3.5% at $d = 3$, 19% at $d = 9$). Short cycles and degeneracy
   (section 4) are the cause. Every working BP decoder is BP plus a step for the shots where BP fails.
2. **BP-OSD equals matching** within the statistical error at every distance, and is about 800 times slower at $d = 9$
   (section 11 explains why).
3. **Ambiguity Clustering is slightly better than matching** at larger $d$ (about 6% fewer failures at $d$ = 7 and 9; e.g.
   $d = 9$, $p = 0.006$: 1.39% vs 1.48%), and the fastest of the BP-based decoders. Single points are within error of each
   other; the advantage is the consistency across points. Section 11 tests why: AC
   makes a (local) class decision on near-tied shots, where matching only picks the single most likely fault set.
4. **Union-Find** is 9–14% worse than matching at $d \ge 5$, as expected for an almost-linear-time approximation to it.
5. **BP-LSD** is 30–40% worse than matching at $d \ge 5$ with order 10. Its clusters are built from BP's soft output, which is
   poor when BP does not converge, which on the surface code is most shots at large $d$. We did not tune it.
6. **Relay-BP** (IBM's defaults for the surface code) is close to matching but falls behind as $d$ grows (17% worse at $d = 9$)
   and is the slowest here. Its parameters were tuned by IBM for the gross code, not for these runs.

For the surface code, matching remains the right choice: as accurate as anything else here and 100–1000 times faster. The
BP-based decoders are built for codes where matching is not possible, such as the gross code.

---

---

## 11. Why BP-OSD ties with matching, and why Ambiguity Clustering beats both

`investigate_ac.py` decodes the same 20,000 shots per point ($d$ = 5, 7, 9; $p$ = 0.004, 0.006; 120,000 shots in all) with
variants that each change one thing, and compares them shot by shot.

### Two different questions a decoder can answer

Section 1 gave two goals. A **minimum-weight** decoder answers "which single fault set is most likely?":

$$\hat e = \arg\min_{e:\ He = s} \sum_j w_j e_j, \qquad \hat\ell = L\hat e.$$

A **class** (degenerate) decoder answers "which logical outcome is most likely?", adding up every fault set that gives it:

$$\hat\ell = \arg\max_{\ell} \sum_{e:\ He = s,\ Le = \ell} P(e).$$

The second is what we want; the first is what most decoders compute. They disagree only when the two logical classes are
nearly tied. Define, for a shot, the **gap**

$$g = w(\text{best fault set in the other class}) - w(\text{best fault set overall}) = \ln\frac{P(\hat e)}{P(\hat e_{\text{other}})} \ \ge 0.$$

If $g$ is large, one class wins by a wide margin and both kinds of decoder agree. If $g$ is close to 0, the best fault sets of
the two classes are almost equally likely, and which class has **more** likely fault sets in total decides the right answer.
(We find the other class's best fault set with BP-OSD on $H$ with the row $L$ appended, its value forced to the opposite
outcome; this gives an upper bound on $g$.)

### BP-OSD and matching solve the same problem

With only the Z detectors kept, every fault on this circuit flips at most two detectors (checked: no fault flips three), so
the problem is a graph and matching finds the exact minimum-weight fault set. BP-OSD looks for the same minimum-weight fault
set by different means: on the surface code BP rarely converges at large $d$, so OSD does most of the work, and OSD's search
is a good but not exhaustive approximation of the minimum. So BP-OSD can at best equal matching here, and it does:

| Pair (120,000 shots) | only the first fails | only the second fails | $z$ |
|---|---|---|---|
| MWPM (Stim's graph) vs MWPM (exact weights) | 115 | 93 | 1.5 |
| MWPM (exact) vs BP-OSD-CS10 | 30 | 53 | −2.5 |
| MWPM (exact) vs BP-OSD-0 | — | — | BP-OSD-0 fails 4–40% more (more at larger $d$) |

($z = (b - c)/\sqrt{b + c}$, McNemar's test on the shots where exactly one of the two fails; $|z| > 2$ is a real difference.)
BP-OSD succeeds or fails together with matching on 99.9% of shots, and loses the few it does by not quite reaching the
minimum. Without the order-10 search (OSD-0) it is clearly worse. "MWPM exact" is matching built from the same merged fault
table as the BP decoders: Stim splits 98 faults at $d = 7$ into two separate edges (e.g. `D44 ^ D2`), and PyMatching then
treats the pieces as independent. That approximation turns out not to matter ($z = 1.5$).

**BP-OSD tying with matching is therefore expected, not odd:** on a graph, BP-OSD is an approximate minimum-weight decoder
competing with an exact one. Its value is on hypergraphs (the gross code), where matching cannot run at all.

### Ambiguity Clustering adds a class decision

AC first finds a fault set the way BP-OSD-0 does, then grows clusters and, inside each cluster whose logical effect is not
fixed by the syndrome, compares the total probability of the alternatives (up to two extra columns) for each logical value
(section 8). That is a small, local version of the class decision above. The variants separate the two steps:

| Pair (120,000 shots) | only the first fails | only the second fails | $z$ |
|---|---|---|---|
| MWPM (exact) vs AC with $\kappa = 0$ (BP + OSD-0 only) | 167 | 229 | −3.1 |
| MWPM (exact) vs AC ($\kappa = 0.05$, default) | 192 | 128 | 3.6 |
| AC $\kappa = 0$ vs AC $\kappa = 0.05$ | 266 | 140 | 6.3 |

Without its cluster step AC is worse than matching; the cluster step is what makes it better (larger clusters, $\kappa = 0.2$,
change nothing further). And the shots it wins are exactly the near-ties:

| | median gap $g$ |
|---|---|
| random shots | 8–18 |
| shots AC gets right and matching wrong | 0.1–0.4 |
| shots matching gets right and AC wrong | 0.1–0.5 |

A gap of 0.3 means the best fault sets of the two classes differ in probability by a factor of only $e^{0.3} \approx 1.35$.
In random samples (400 shots per point), about 1% of shots have $g < 1$, matching fails on 12 of those 28, and they account
for about half of matching's failures. Of the 320 shots where matching and AC disagree, 281 have $g < 1$. On them matching picks the single best fault set, which is close to a coin flip, while AC weighs
how much probability each class has nearby and wins 192 of the 320 contested shots. That is where its ~6% lower error
rate at $d$ = 7–9 comes from.

**Limits.** AC's class decision is local (inside clusters, at most two extra columns), so it is not the optimal decoder; a
tensor-network or other maximum-likelihood decoder would show how much more there is to gain. The difference is a few
percent here because the surface code's near-ties are rare; it should matter more on codes and noise where degeneracy is
larger.


## References

- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. 43, 4452 (2002).
- Q. McNemar, *Note on the sampling error of the difference between correlated proportions*, Psychometrika 12, 153 (1947).
- J. Edmonds, *Paths, trees, and flowers*, Canad. J. Math. 17, 449 (1965).
- O. Higgott and C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching*,
  arXiv:2303.15933.
- N. Delfosse and N. H. Nickerson, *Almost-linear time decoding algorithm for topological codes*, Quantum 5, 595 (2021),
  arXiv:1709.06218.
- R. G. Gallager, *Low-density parity-check codes*, MIT Press (1963).
- D. Poulin and Y. Chung, *On the iterative decoding of sparse quantum codes*, arXiv:0801.1241.
- M. P. C. Fossorier and S. Lin, IEEE Trans. Inf. Theory 41, 1379 (1995).
- P. Panteleev and G. Kalachev, *Degenerate quantum LDPC codes with good finite length performance*, Quantum 5, 585 (2021),
  arXiv:1904.02703.
- J. Roffe, D. R. White, S. Burton, E. Campbell, *Decoding across the quantum LDPC code landscape*, Phys. Rev. Research 2,
  043423 (2020), arXiv:2005.07016.
- T. Hillmann et al., *Localized statistics decoding for quantum low-density parity-check codes*, arXiv:2406.18655.
- T. Müller et al., *Improved belief propagation is sufficient for real-time decoding of quantum memory*, arXiv:2506.01779.
- S. Wolanski and B. Barber, *Ambiguity Clustering*, arXiv:2406.14527.
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021).
