"""
test_run_gross.py -- checks for run_gross.py (about a minute).

Run from this folder:  python -m pytest -q test_run_gross.py
"""
import csv

import run_gross


def test_parse_plan():
    assert run_gross.parse_plan(["0.004:3000", "0.005:10"]) == [(0.004, 3000), (0.005, 10)]


def test_small_run_writes_the_table(tmp_path):
    """A 20-shot run on both circuits writes the same columns the notebooks read, plus IBM's published row."""
    out = tmp_path / "results_gross.csv"
    rows = run_gross.run([(0.004, 20)], out, workers=2)
    with open(out) as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0]) == run_gross.COLUMNS
    assert [r["circuit"] for r in table] == ["ibm", "sq", "ibm_published"]
    for r in table[:2]:
        assert int(r["shots"]) == 20 and 0 <= int(r["failures"]) <= 20
        assert float(r["ci_low"]) <= float(r["P_L_12rounds"]) <= float(r["ci_high"])
    assert len(rows) == 3
