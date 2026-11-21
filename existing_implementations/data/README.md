# data/ — IBM reference material (copied, unmodified)

Source: IBM's Relay-BP repository, https://github.com/trmue/relay
(commit d185194ba0cb4101ced4340d82b2ee6d42f225f0, 21 Jul 2026), Apache License 2.0 (IBM_LICENSE.txt).

- ibm_circuits/gross_memory_Z_p<p>.stim
    = tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,
      rounds=12,error_rate=<p>,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim
    Only the file NAMES were shortened (Windows path-length limit); contents are untouched.
- ibm_published_sweep_gross.csv
    = examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv (IBM's own decoder sweep).

Our own saved results (used by notebooks 01 and 02):
- results_gross.csv   gross code memory, BP-OSD with IBM's settings: IBM's circuit, stabilizer-qec's circuit, and
                      IBM's published numbers (runs of 23 Oct and 7 Nov 2026).
                      Regenerate with `python run_gross.py` from the folder above.
- results_surface.csv rotated surface code memory (Stim's circuit), PyMatching, d = 3-12, up to 200,000 shots
                      or 200 failures per point.
