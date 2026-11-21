"""
common.py -- helpers used by surface_code_decoders/ (IBM circuits, describing circuits, Wilson intervals, plotting).

What this file is for
---------------------
Three things that would otherwise be repeated in every notebook:
  1. getting IBM's Relay-BP repository (its Stim circuit files and published results),
  2. loading one of IBM's circuits, optionally keeping only its Z detectors,
  3. small statistics and plotting helpers (Wilson interval, saving figures).
Nothing here is specific to one notebook, and nothing here decodes.

Sources: IBM Relay-BP repo, https://github.com/trmue/relay (Apache-2.0), Mueller et al. arXiv:2506.01779.
The Z-detector filter is IBM's own function (tests/testdata/utils.py), imported from the clone.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import stim

RELAY_URL = "https://github.com/trmue/relay.git"


def get_relay_repo(path: str | Path | None = None) -> Path:
    """Path of IBM's Relay-BP clone; clones it on first use (about 110 MB, needs git).

    Default location is <home>/relay. It is kept short on purpose: IBM's circuit file names
    are up to 139 characters and Windows limits a full path to 260.
    """
    path = Path(path).expanduser().resolve() if path else Path.home() / "relay"
    if not (path / "tests" / "testdata").exists():
        print(f"Cloning {RELAY_URL} into {path} ...")
        subprocess.run(["git", "clone", "-c", "core.longpaths=true", "--depth", "1", RELAY_URL, str(path)], check=True)
    n = sum(1 for _ in (path / "tests" / "testdata").rglob("*.stim"))
    if n != 672:
        raise RuntimeError(f"{path} has {n} of IBM's 672 circuit files (incomplete clone). Delete it and retry.")
    if str(path / "tests") not in sys.path:
        sys.path.append(str(path / "tests"))
    if not relay_bp_installed():
        _use_noise_module_from_clone(path)
    return path


def relay_bp_installed() -> bool:
    """True if IBM's relay_bp decoder package is importable (it needs Rust to build on Windows)."""
    try:
        import relay_bp
        return hasattr(relay_bp, "RelayDecoderF32")
    except ImportError:
        return False


def _use_noise_module_from_clone(repo: Path) -> None:
    """IBM's test helpers import one function from relay_bp (detect_data_qubits). Without the
    package, load IBM's own src/relay_bp/stim/noise.py from the clone under that name."""
    spec = importlib.util.spec_from_file_location("relay_bp.stim.noise", repo / "src" / "relay_bp" / "stim" / "noise.py")
    noise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(noise)
    pkg, sub = types.ModuleType("relay_bp"), types.ModuleType("relay_bp.stim")
    pkg.__path__, sub.__path__ = [], []
    pkg.stim, sub.noise = sub, noise
    sys.modules.update({"relay_bp": pkg, "relay_bp.stim": sub, "relay_bp.stim.noise": noise})


def ibm_circuit(name: str, distance: int, p: float, z_detectors_only: bool = False) -> stim.Circuit:
    """One of IBM's stored memory circuits (rounds = distance), e.g.
    ibm_circuit("rotated_surface_code_memory_Z", 5, 0.003)."""
    get_relay_repo()
    from testdata import filter_detectors_by_basis, get_test_circuit
    c = get_test_circuit(name, distance=distance, rounds=distance, error_rate=p)
    return filter_detectors_by_basis(c, "Z") if z_detectors_only else c


def wilson(failures: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a failure fraction (Wilson, JASA 1927)."""
    f = failures / shots
    den = 1 + z * z / shots
    mid = (f + z * z / (2 * shots)) / den
    half = z * np.sqrt(f * (1 - f) / shots + z * z / (4 * shots * shots)) / den
    return mid - half, mid + half


def savefig(name: str, folder: str | Path = "figures") -> None:
    """Save the current matplotlib figure as a 200-dpi PNG in ./figures."""
    import matplotlib.pyplot as plt
    Path(folder).mkdir(exist_ok=True)
    plt.savefig(Path(folder) / name, dpi=200, bbox_inches="tight", facecolor="white")


# ----------------------------------------------------------------------------------------------
# Looking at circuits
# ----------------------------------------------------------------------------------------------
def describe(circuit: stim.Circuit) -> dict:
    """Size of a circuit in plain numbers: qubits, CX gates, measurements, detectors, observables,
    and the number of distinct faults in its detector error model."""
    cx = meas = 0
    used = set()                                   # qubits that actually appear (Stim's num_qubits is max index + 1)
    for inst in circuit.flattened():
        targets = inst.targets_copy()
        n = len(targets)
        if inst.name in ("CX", "CZ"):
            cx += n // 2
        elif inst.name in ("M", "MX", "MR", "MRX"):
            meas += n
        if inst.name in ("CX", "CZ", "M", "MX", "MR", "MRX", "R", "RX", "H"):
            used.update(t.value for t in targets)
    return {"qubits": len(used), "CX gates": cx, "measurements": meas,
            "detectors": circuit.num_detectors, "logical observables": circuit.num_observables,
            "possible faults": circuit.detector_error_model(decompose_errors=False).num_errors}


def operation_counts(circuit: stim.Circuit) -> dict:
    """How many times each instruction is applied (a CX on one pair counts once)."""
    import collections
    counts = collections.Counter()
    for inst in circuit.flattened():
        n = len(inst.targets_copy())
        if inst.name in ("CX", "CZ", "DEPOLARIZE2"):
            n //= 2
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            n = 1
        counts[inst.name] += n
    return dict(counts.most_common())


def print_lines(circuit: stim.Circuit, n: int = 10, start: str | None = None) -> None:
    """Print n lines of the circuit file, optionally starting at the first line that begins with `start`."""
    lines = str(circuit).splitlines()
    first = next((i for i, l in enumerate(lines) if l.startswith(start)), 0) if start else 0
    if first:
        print(f"... lines 1-{first} skipped ...")
    for line in lines[first:first + n]:
        print(line if len(line) <= 110 else line[:110] + " ...")
    print(f"... {len(lines)} lines in total")


# ----------------------------------------------------------------------------------------------
# Running and plotting
# ----------------------------------------------------------------------------------------------
def relay_decoders() -> dict:
    """Sinter decoders keyed "relay-bp": IBM's package if installed, otherwise stabilizer-qec's
    independent implementation of the same method (prints which one is used)."""
    if relay_bp_installed():
        from relay_bp.stim import sinter_decoders
        return sinter_decoders()
    import sinter
    import stabilizer_qec as sq

    class _Compiled(sinter.CompiledDecoder):
        def __init__(self, dem):
            self.dec, self.nd, self.no = sq.RelayBp(dem), dem.num_detectors, dem.num_observables

        def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
            dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, : self.nd]
            pred = np.asarray(self.dec.decode_batch(dets.astype(bool)), dtype=np.uint8).reshape(len(dets), self.no)
            return np.packbits(pred, axis=1, bitorder="little")

    class _Decoder(sinter.Decoder):
        def compile_decoder_for_dem(self, *, dem):
            return _Compiled(dem)

    print("relay_bp is not installed: using stabilizer-qec's Relay-BP")
    return {"relay-bp": _Decoder()}


def collect(tasks, decoder: str, max_shots: int, max_errors: int):
    """Sample and decode a list of sinter.Task with one decoder ("pymatching" or "relay-bp")."""
    import multiprocessing
    import sinter
    return sinter.collect(tasks=tasks, decoders=[decoder], num_workers=multiprocessing.cpu_count(),
                          custom_decoders=relay_decoders() if decoder == "relay-bp" else None,
                          max_shots=max_shots, max_errors=max_errors, print_progress=False)


def ibm_published(csv_name: str, **where) -> list:
    """IBM's own results (examples/data/<csv_name> in the repo), filtered by e.g.
    decoder="relay-bp", basis="Z", xyz=False."""
    import sinter
    stats = sinter.read_stats_from_csv_files(str(get_relay_repo() / "examples" / "data" / csv_name))
    def ok(s):
        m = s.json_metadata
        have = {"decoder": s.decoder, "basis": m["circuit"][-1], **m}
        return all(have.get(k) == v for k, v in where.items())
    return [s for s in stats if ok(s)]


def per_round(failures: int, shots: int, rounds: int) -> tuple[float, float, float]:
    """Failure rate per syndrome round with a 95% interval: 1 - (1 - P)^(1/rounds)."""
    lo, hi = wilson(failures, shots)
    f = lambda P: 1 - (1 - P) ** (1 / rounds)
    return f(failures / shots), f(max(lo, 0.0)), f(hi)


def plot_error_rates(series: dict, title: str, ax=None, styles: dict | None = None):
    """Logical error per round against physical error rate p, one line per entry of `series`
    (label -> list of sinter.TaskStats). Bars are 95% intervals. Points with zero failures are skipped."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))
    for label, stats in series.items():
        stats = sorted((s for s in stats if s.errors), key=lambda s: s.json_metadata["p"])
        if not stats:
            continue
        p = [s.json_metadata["p"] for s in stats]
        y, lo, hi = map(np.array, zip(*(per_round(s.errors, s.shots, s.json_metadata.get("r", s.json_metadata.get("d")))
                                         for s in stats)))
        ax.errorbar(p, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=label, **(styles or {}).get(label, {}))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate per round")
    ax.set_title(title); ax.grid(which="both", alpha=0.3); ax.legend(fontsize=9)
    return ax
