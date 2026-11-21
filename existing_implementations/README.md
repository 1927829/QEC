# Existing implementations of the gross code

QCSA Project 1 compares the [[144,12,12]] gross code, a bivariate bicycle (BB) code, with the rotated surface
code. This folder surveys the existing open implementations of the gross code: IBM's Relay-BP repository,
Riverlane's Deltakit and the `stabilizer-qec` package. We run their circuits and decoders, check their numbers
against each other and against IBM's published results, and compare one gross code block with twelve surface
code patches in memory experiments. Two further studies build on this: writing the gross code circuit directly
from its two polynomials, and decoding the surface code with every available decoder.

## Contents

| | What it is for |
|---|---|
| `01_ibm_relay_bp.ipynb` | IBM's Relay-BP repository: its 672 circuits, the gross code circuit, why only Z detectors are decoded, surface code and small BB codes run live, our gross code result against IBM's published one |
| `02_memory_experiments.ipynb` | Three memory experiments: IBM's gross code circuit, `stabilizer-qec`'s gross code circuit, Stim's surface code; one gross block against twelve d = 12 surface patches; decoder speed |
| `03_deltakit.ipynb` | Riverlane's Deltakit: codes, noise models, local decoding, the gross code in Deltakit against IBM's and `stabilizer-qec`'s circuits, what its cloud decoders do |
| `04_stabilizer_qec.ipynb` | `stabilizer-qec`: the gross code object, logical operators, automorphisms, memory and logical-measurement circuits, four decoders on the same shots |
| `codes_from_polynomials/` | The gross code written from its two polynomials, up to the full Stim circuit, checked against IBM's. Notebook, `bbcode.py`, tests, and `polynomials_to_circuits.md` (the maths) |
| `surface_code_decoders/` | The surface code decoded by seven decoders on the same shots, and why BP-OSD ties with matching while Ambiguity Clustering beats it. Notebook, scripts, results, tests, and `decoders_explained.md` (each decoder with its maths) |
| `run_gross.py` | Regenerates `data/results_gross.csv` (the gross code BP-OSD results shown in notebooks 01 and 02). Shots and noise levels are set on the command line, e.g. `python run_gross.py 0.004:3000` |
| `helpers/` | Functions shared by the notebooks. Each notebook lists, at the top, exactly which ones it uses |
| `data/` | Saved results of the long runs, and IBM's 14 gross code circuits (Apache-2.0, see `data/IBM_LICENSE.txt`) |
| `results/` | Cached results of the notebooks' own runs; a notebook re-uses them instead of re-running |
| `figures/` | Figures written by the notebooks |
| `docs/` | References, and the project presentation and report |

## Running

```
python -m venv .venv            # Python 3.12 recommended
.venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install jupyterlab
jupyter lab
```

Open a notebook and run all cells: the first code cell installs its packages (`requirements.txt` lists them all).
Run the notebooks from this folder, since they read `helpers/`, `data/` and `results/` by relative path.

- **IBM's repository** is cloned on first use into `~/relay` (about 110 MB; needs `git`). It is not cloned into this
  folder because IBM's circuit file names are up to 139 characters long and Windows limits a full path to 260.
- **IBM's `relay_bp` package** has no ready-made Windows install and needs the Rust compiler (https://rustup.rs).
  Without it, the notebooks use `stabilizer-qec`'s Relay-BP, a separate implementation of the same method, and
  say so.
- **Deltakit's cloud decoders** need an account token (https://deltakit.riverlane.com), pasted into the first code
  cell of notebook 03. Without one, the local parts run and the cloud parts show saved results. Never commit a
  notebook with a token in it.

Tests: `python -m pytest -q test_run_gross.py helpers codes_from_polynomials surface_code_decoders`.

## Main results

- Our BP-OSD pipeline on IBM's gross code circuit reproduces IBM's published BP-OSD numbers (2.7% vs 2.65% over
  12 rounds at p = 0.004).
- `stabilizer-qec`'s independently built gross circuit agrees within error. Deltakit builds the same check
  matrices, but its general-purpose schedule idles more and fails about twice as often under the same noise.
- One gross block (288 qubits) vs twelve d = 12 surface patches (3,444 qubits): the patches fail 2–3× less often at
  p = 0.002–0.005, with 12× more qubits.
- The gross code circuit written from its two polynomials (586 lines of Stim, against 26,411 in IBM's file) matches
  IBM's error rate (2.9% vs 2.65% at p = 0.004).
- On the surface code, BP-OSD equals matching: both look for the single most likely set of faults, matching exactly
  and BP-OSD approximately. Ambiguity Clustering is about 6% better at d = 7–9, because on nearly tied shots it
  weighs the total probability of each logical outcome. Plain BP has no threshold.
- A Stim circuit written with `REPEAT` gives a detector error model with unmerged duplicate faults, which makes
  BP-OSD worse; build the model from `circuit.flattened()`.

References: `docs/references.md`.
