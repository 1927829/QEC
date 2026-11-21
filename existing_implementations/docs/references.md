# References

Everything cited in the notebooks and documents of this folder. The project presentation and report can be
added to this `docs/` folder next to this file.

## Codes

- S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, T. J. Yoder, *High-threshold and low-overhead
  fault-tolerant quantum memory*, Nature 627, 778 (2024), [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
  The gross code, its depth-8 syndrome circuit and the uniform circuit noise model.
- T. J. Yoder et al., *Tour de gross: a modular quantum computer based on bivariate bicycle codes*,
  [arXiv:2506.03094](https://arxiv.org/abs/2506.03094). Automorphisms as logical operations; instruction error rates.
- D. J. Williamson and T. J. Yoder, *Low-overhead fault-tolerant quantum computation by gauging logical operators*,
  [arXiv:2410.02213](https://arxiv.org/abs/2410.02213).
- A. Cross, Z. He, P. Rall, T. Yoder, *Improved QLDPC surgery: logical measurements and bridging codes*,
  [arXiv:2407.18393](https://arxiv.org/abs/2407.18393).
- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. 43, 4452 (2002).
- P. Panteleev and G. Kalachev, *Degenerate quantum LDPC codes with good finite length performance*, Quantum 5, 585
  (2021), [arXiv:1904.02703](https://arxiv.org/abs/1904.02703).

## Decoders

- O. Higgott and C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight
  matching*, [arXiv:2303.15933](https://arxiv.org/abs/2303.15933) (PyMatching).
- J. Edmonds, *Paths, trees, and flowers*, Canad. J. Math. 17, 449 (1965) (blossom algorithm).
- N. Delfosse and N. H. Nickerson, *Almost-linear time decoding algorithm for topological codes*, Quantum 5, 595
  (2021), [arXiv:1709.06218](https://arxiv.org/abs/1709.06218) (Union-Find).
- D. Poulin and Y. Chung, *On the iterative decoding of sparse quantum codes*, Quant. Inf. Comput. 8, 987 (2008),
  [arXiv:0801.1241](https://arxiv.org/abs/0801.1241) (BP on quantum codes).
- J. Roffe, D. R. White, S. Burton, E. Campbell, *Decoding across the quantum low-density parity-check code
  landscape*, Phys. Rev. Research 2, 043423 (2020), [arXiv:2005.07016](https://arxiv.org/abs/2005.07016)
  (BP-OSD; `ldpc` package).
- T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, J. Roffe, *Localized statistics decoding for quantum
  low-density parity-check codes*, [arXiv:2406.18655](https://arxiv.org/abs/2406.18655) (BP-LSD).
- T. Müller, T. Alexander, M. E. Beverland, M. Bühler, B. R. Johnson, T. Maurer, D. Vandeth, *Improved belief propagation is sufficient for real-time decoding of quantum memory*,
  [arXiv:2506.01779](https://arxiv.org/abs/2506.01779) (Relay-BP).
- S. Wolanski and B. Barber, *Ambiguity Clustering: an accurate and efficient decoder for qLDPC codes*,
  [arXiv:2406.14527](https://arxiv.org/abs/2406.14527).
- O. Higgott, T. C. Bohdanowicz, A. Kubica, S. T. Flammia, E. T. Campbell, *Improved decoding of circuit noise and
  fragile boundaries of tailored surface codes*, Phys. Rev. X 13, 031007 (2023),
  [arXiv:2203.04948](https://arxiv.org/abs/2203.04948) (belief matching).
- B. Barber et al., *A real-time, scalable, fast and highly resource efficient decoder for a quantum computer*,
  Nature Electronics (2025), doi:10.1038/s41928-024-01319-5, [arXiv:2309.05558](https://arxiv.org/abs/2309.05558) (Collision Clustering).

## Noise and simulation

- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021),
  [arXiv:2103.02202](https://arxiv.org/abs/2103.02202).
- C. Gidney, M. Newman, A. Fowler, M. Broughton, *A fault-tolerant honeycomb memory*, Quantum 5, 605 (2021),
  [arXiv:2108.10457](https://arxiv.org/abs/2108.10457) (defines the SI1000 noise model).

## Statistics

- E. B. Wilson, *Probable inference, the law of succession, and statistical inference*, J. Am. Stat. Assoc. 22, 209
  (1927) (error bars).
- Q. McNemar, *Note on the sampling error of the difference between correlated proportions*, Psychometrika 12, 153
  (1947) (paired comparison of decoders).

## Software

| Package / repository | Version | Link |
|---|---|---|
| IBM Relay-BP (circuits, decoder, published sweeps) | commit d185194 (21 Jul 2026) | https://github.com/trmue/relay (Apache-2.0) |
| `stabilizer-qec` | 2.1.0 | https://pypi.org/project/stabilizer-qec/ |
| Riverlane Deltakit | 0.10.2 | https://deltakit-docs.riverlane.com |
| Stim, Sinter | 1.16.0, 1.15.0 | https://github.com/quantumlib/Stim |
| PyMatching | 2.4.0 | https://github.com/oscarhiggott/PyMatching |
| `ldpc` | 2.4.1 | https://github.com/quantumgizmos/ldpc |
| `beliefmatching` | 0.2.0 | https://github.com/oscarhiggott/BeliefMatching |
| Bravyi et al.'s original code | – | https://github.com/sbravyi/BivariateBicycleCodes |
