"""
tour_helpers.py — shared helpers for notebooks 01-04 in existing_implementations/.

What this file is for
---------------------
The notebooks should read like a walk through each reference repo: short cells,
clear parameters, a graph after every run. Anything longer than a few lines
(plotting, tables, the fallback decoder) lives here so the notebook cells stay
short. Nothing in here changes the reference code; it only wraps it.

Sources
-------
- IBM Relay-BP repository: https://github.com/trmue/relay  (Apache-2.0)
  Müller et al., "Improved belief propagation is sufficient for real-time decoding
  of quantum memory", arXiv:2506.01779.
- Stim (Gidney, Quantum 5, 497, 2021) and Sinter: https://github.com/quantumlib/Stim
- PyMatching (Higgott & Gidney, arXiv:2303.15933).
- stabilizer-qec 2.1.0 (fallback Relay-BP if IBM's package is not installed).
"""

from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
import stim

RELAY_URL = "https://github.com/trmue/relay.git"

# One colour per code family, reused in every plot so the team can follow them.
FAMILY_COLOURS = {
    "surface": "#2a6fdb",
    "color": "#e07a1f",
    "bicycle_bivariate": "#2e9e5b",
    "repetition": "#8a5cc2",
}


# --------------------------------------------------------------------------
# Step 1 · getting the repo
# --------------------------------------------------------------------------
def get_repo(path: str | Path | None = None) -> Path:
    """Return the path of IBM's Relay-BP clone; clone it first if it is missing.

    Default location: a folder called `relay` in your home folder (C:\\Users\\<you>\\relay on
    Windows). It is kept short on purpose: Windows limits a full file path to 260
    characters, IBM's BB circuit file names alone are 139 characters, so a clone inside a
    deep project folder loses files.

    Also puts the repo's `tests/` folder on sys.path so that IBM's own helpers
    (`from testdata import get_test_circuit, filter_detectors_by_basis`) can be
    imported exactly as IBM's GettingStarted notebook does.
    """
    path = Path(path).expanduser().resolve() if path else Path.home() / "relay"
    if not (path / "tests" / "testdata").exists():
        print(f"Cloning {RELAY_URL} into {path} (about 110 MB, mostly circuits) ...")
        subprocess.run(["git", "clone", "-c", "core.longpaths=true", "--depth", "1", RELAY_URL, str(path)],
                       check=True)
    # Make sure the whole circuit catalogue arrived (a cut-short clone is missing files).
    n = sum(1 for _ in (path / "tests" / "testdata").rglob("*.stim"))
    if n != 672:
        raise RuntimeError(f"{path} has {n} of IBM's 672 circuit files: the clone is incomplete. "
                           f"Delete that folder and run this cell again (or pass a shorter path).")
    tests = str(path / "tests")
    if tests not in sys.path:
        sys.path.append(tests)
    if not relay_installed():
        _load_ibm_noise_module(path)
    return path


def _load_ibm_noise_module(repo: Path):
    """Make `from relay_bp.stim.noise import detect_data_qubits` work without the package.

    IBM's test helpers (tests/testdata/utils.py) import one function, detect_data_qubits,
    from the relay_bp package. When that package is not installed (on Windows it needs
    Rust to build), we load IBM's own file src/relay_bp/stim/noise.py from the clone,
    unchanged, under the same module name. It only needs Stim. The decoder itself is not
    loaded this way; relay_decoders() falls back to stabilizer-qec for that.
    """
    import importlib.util
    import types
    src = repo / "src" / "relay_bp" / "stim" / "noise.py"
    spec = importlib.util.spec_from_file_location("relay_bp.stim.noise", src)
    noise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(noise)
    pkg, sub = types.ModuleType("relay_bp"), types.ModuleType("relay_bp.stim")
    pkg.__path__, sub.__path__ = [], []          # mark them as packages
    pkg.stim, sub.noise = sub, noise
    sys.modules.update({"relay_bp": pkg, "relay_bp.stim": sub, "relay_bp.stim.noise": noise})
    print("relay_bp package not installed: loaded IBM's relay_bp/stim/noise.py from the clone "
          "(needed by IBM's test helpers); decoding will use stabilizer-qec's Relay-BP")


def relay_installed() -> bool:
    """True if IBM's own `relay_bp` package can be imported."""
    try:
        import relay_bp
        return hasattr(relay_bp, "RelayDecoderF32")   # False for the stand-in made by _load_ibm_noise_module
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Step 2 · what is inside the repo
# --------------------------------------------------------------------------
def catalogue(repo: Path) -> pd.DataFrame:
    """One row per stored Stim circuit, read from the file names.

    IBM's file names are lists of key=value pairs, e.g.
    circuit=rotated_surface_code_memory_Z,distance=5,rounds=5,error_rate=0.003,...stim
    """
    rows = []
    for f in (repo / "tests" / "testdata").rglob("*.stim"):
        kv = dict(part.split("=", 1) for part in f.stem.split(",") if "=" in part)
        name = kv["circuit"]
        rows.append({
            "family": f.parent.name,                 # folder: surface / color / bicycle_bivariate
            "circuit": name,
            "code": name.rsplit("_memory_", 1)[0],   # circuit name without the basis
            "basis": name.rsplit("_memory_", 1)[1],  # X, Z or choi_XZ
            "distance": int(kv["distance"]),
            "rounds": int(kv["rounds"]),
            "p": float(kv["error_rate"]),
            "noise_model": kv.get("noise_model", ""),
            "file": f.name,
        })
    return pd.DataFrame(rows).sort_values(["family", "code", "distance", "basis", "p"]).reset_index(drop=True)


def plot_catalogue(cat: pd.DataFrame):
    """Two panels: how many circuits each code has, and which distances exist."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))

    # Left: number of stored circuits per code.
    counts = cat.groupby(["family", "code"]).size().reset_index(name="n")
    labels = [c.replace("bicycle_bivariate_", "BB ").replace("_code", "").replace("_", " ") for c in counts.code]
    a1.barh(labels, counts.n, color=[FAMILY_COLOURS[f] for f in counts.family])
    a1.set_xlabel("number of Stim circuits in tests/testdata")
    a1.set_title("Circuits shipped with IBM's Relay-BP repo")
    a1.invert_yaxis()

    # Right: distance vs family, one dot per (code, distance) available.
    fams = list(dict.fromkeys(cat.family))
    for i, fam in enumerate(fams):
        ds = sorted(cat[cat.family == fam].distance.unique())
        a2.scatter(ds, [i] * len(ds), s=120, color=FAMILY_COLOURS[fam])
    a2.set_yticks(range(len(fams)), fams)
    a2.set_xlabel("code distance d (= number of syndrome rounds)")
    a2.set_title("Distances available (each at 14 noise levels)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Step 4 · the 3-bit repetition code (IBM GettingStarted example)
# --------------------------------------------------------------------------
def repetition_fallback(syndromes: np.ndarray) -> np.ndarray:
    """Decode IBM's 3-bit example with stabilizer-qec's RelayBp when relay_bp is missing.

    The same problem written as a Stim error model: bit 1 lights check D0, bit 2 lights
    D0 and D1, bit 3 lights D1; every bit flip also flips the logical bit L0.
    Returns the decoder's guess of whether L0 flipped, one entry per syndrome.
    """
    import stabilizer_qec as sq
    dem = stim.DetectorErrorModel("error(0.003) D0 L0\nerror(0.003) D0 D1 L0\nerror(0.003) D1 L0")
    pred = np.asarray(sq.RelayBp(dem).decode_batch(syndromes.astype(bool)), dtype=np.uint8)
    return pred.reshape(len(syndromes), -1)[:, 0]


def plot_repetition(errors: np.ndarray, syndromes: np.ndarray, corrected: np.ndarray):
    """One bar per error pattern: green if the decoder got the logical bit right."""
    labels = ["".join(map(str, e)) for e in errors]
    weights = errors.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(labels, weights, color=["#2e9e5b" if c else "#d1495b" for c in corrected])
    for i, s in enumerate(syndromes):
        ax.text(i, weights[i] + 0.05, f"checks {s[0]}{s[1]}", ha="center", fontsize=9)
    ax.set_xlabel("error pattern (bit 1, bit 2, bit 3 flipped?)")
    ax.set_ylabel("number of flipped bits")
    ax.set_ylim(0, 3.6)
    ax.set_title("3-bit repetition code: green = decoder corrected it, red = logical error")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Step 5 · looking at circuits
# --------------------------------------------------------------------------
def circuit_summary(circuits: dict[str, stim.Circuit]) -> pd.DataFrame:
    """Size of each circuit: qubits, gates, detectors, observables, error mechanisms."""
    rows = []
    for name, c in circuits.items():
        flat = c.flattened()
        ops = collections.Counter()
        for inst in flat:
            n_targets = len(inst.targets_copy())
            if inst.name in ("CX", "CZ"):
                ops["two-qubit gates"] += n_targets // 2
            elif inst.name in ("M", "MX", "MR", "MRX"):
                ops["measurements"] += n_targets
        rows.append({
            "circuit": name,
            "qubits": c.num_qubits,
            "two-qubit gates": ops["two-qubit gates"],
            "measurements": ops["measurements"],
            "detectors": c.num_detectors,
            "logical observables": c.num_observables,
            "error mechanisms (DEM)": c.detector_error_model(decompose_errors=False).num_errors,
        })
    return pd.DataFrame(rows).set_index("circuit")


def plot_summary(summary: pd.DataFrame):
    """Grouped bar chart (log scale) of the circuit sizes from circuit_summary."""
    cols = ["qubits", "two-qubit gates", "detectors", "error mechanisms (DEM)"]
    ax = summary[cols].plot.bar(figsize=(12, 4.2), logy=True, rot=0, width=0.8)
    ax.set_ylabel("count (log scale)")
    ax.set_title("How big each circuit is")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rounds(c: stim.Circuit, title: str):
    """Cumulative number of CX gates along the circuit file, with a line at every
    measurement layer (= end of a syndrome round).

    IBM's files list one gate per line and put no TICK (time-step) markers inside the
    rounds, so we count along the file instead of per time step.
    """
    cx_so_far, total, xs, ys, rounds = 0, 0, [], [], []
    for inst in c.flattened():
        if inst.name in ("CX", "CZ"):
            cx_so_far += len(inst.targets_copy()) // 2
        elif inst.name in ("M", "MX", "MR", "MRX"):
            if not rounds or rounds[-1] != cx_so_far:
                rounds.append(cx_so_far)   # several measurement lines at the same point = one layer
        total += 1
        xs.append(total)
        ys.append(cx_so_far)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(ys, color="#2a6fdb", lw=2)
    ax.set_xticks([])
    for k, r in enumerate(rounds):
        ax.axhline(r, color="#e07a1f", lw=0.8, alpha=0.7)
    ax.set_xlabel("position in the circuit file (start → end)")
    ax.set_ylabel("CX gates so far")
    per_round = np.diff([0] + rounds)
    ax.set_title(f"{title}\n{len(rounds)} measurement layers (orange); {int(np.median(per_round))} CX gates per round")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# Step 6 · X and Z detectors
# --------------------------------------------------------------------------
def plot_detector_split(rows: list[tuple[str, int, int]], save: str | None = None):
    """rows = [(label, all detectors, Z-only detectors)] -> stacked bar chart."""
    labels = [r[0] for r in rows]
    total = np.array([r[1] for r in rows])
    z = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(labels, z, color="#2a6fdb", label="kept: Z-type detectors (detect X errors)")
    ax.bar(labels, total - z, bottom=z, color="#c8c8c8", label="dropped: X-type detectors")
    for i, (t, k) in enumerate(zip(total, z)):
        ax.text(i, t, f"{k} of {t}", ha="center", va="bottom")
    ax.set_ylabel("detectors")
    ax.set_title("filter_detectors_by_basis(circuit, 'Z') on IBM's Z-memory circuits")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save:
        savefig(save)
    plt.show()


# --------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------
class _SqRelayCompiled(sinter.CompiledDecoder):
    """Sinter wrapper around stabilizer-qec's RelayBp (used only if relay_bp is missing)."""

    def __init__(self, dem: stim.DetectorErrorModel):
        import stabilizer_qec as sq
        self.dec = sq.RelayBp(dem)
        self.num_det = dem.num_detectors
        self.num_obs = dem.num_observables

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        dets = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")[:, : self.num_det]
        pred = np.asarray(self.dec.decode_batch(dets.astype(bool)), dtype=np.uint8)
        return np.packbits(pred.reshape(len(dets), self.num_obs), axis=1, bitorder="little")


class _SqRelayDecoder(sinter.Decoder):
    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        return _SqRelayCompiled(dem)


def relay_decoders() -> dict:
    """Sinter custom decoders, keyed 'relay-bp'.

    Uses IBM's own `relay_bp.stim.sinter_decoders()` (IBM's default settings) when
    installed. Otherwise falls back to stabilizer-qec's RelayBp, a separate
    implementation of the same algorithm, so the notebook still runs on Windows.
    """
    if relay_installed():
        from relay_bp.stim import sinter_decoders
        return sinter_decoders()
    print("relay_bp not installed -> using stabilizer-qec's RelayBp instead")
    return {"relay-bp": _SqRelayDecoder()}


def results_path(save_to: str | Path, decoders: list[str]) -> Path:
    """Where to save Sinter results. Sinter records both Relay-BP versions under the same decoder name
    'relay-bp', so when IBM's package is missing (stabilizer-qec's version is used instead) the results go
    to a separate '<name>_stabilizer_qec.csv' file. Otherwise one would be re-used as if it were the other."""
    save_to = Path(save_to)
    if "relay-bp" in decoders and not relay_installed():
        return save_to.with_name(save_to.stem + "_stabilizer_qec" + save_to.suffix)
    return save_to


def collect(tasks: list[sinter.Task], decoders: list[str], max_shots: int, max_errors: int,
            save_to: str | Path | None = None, workers: int | None = None) -> list[sinter.TaskStats]:
    """sinter.collect with our defaults. Results are saved to `save_to`; Sinter resumes from
    that file, so re-running a cell re-uses finished work instead of redoing it."""
    import multiprocessing
    if save_to:
        save_to = results_path(save_to, decoders)
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    return sinter.collect(
        tasks=tasks,
        decoders=decoders,
        custom_decoders=relay_decoders() if "relay-bp" in decoders else None,
        num_workers=workers or multiprocessing.cpu_count(),
        max_shots=max_shots,
        max_errors=max_errors,
        save_resume_filepath=str(save_to) if save_to else None,
        print_progress=False,
    )


def per_round(stat: sinter.TaskStats) -> tuple[float, float, float]:
    """Logical error rate per syndrome round, with an uncertainty band.

    Whole-experiment failure P over r rounds -> per round 1 - (1 - P)^(1/r), the same
    conversion IBM uses in its notebooks (failure_units_per_shot_func = rounds).
    The band is Sinter's standard one: all rates within a likelihood ratio of 1000 of
    the best estimate (Sinter's plotting default).
    """
    r = stat.json_metadata.get("r", stat.json_metadata.get("d"))
    fit = sinter.fit_binomial(num_shots=stat.shots - stat.discards, num_hits=stat.errors,
                              max_likelihood_factor=1000)
    conv = lambda P: 1 - (1 - P) ** (1 / r)
    return conv(fit.best), conv(fit.low), conv(fit.high)


def plot_ler(series: dict[str, list[sinter.TaskStats]], title: str, styles: dict | None = None,
             ax=None, ylabel="logical error rate per round"):
    """One line per entry in `series`; x = physical error rate p, y = error per round."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    styles = styles or {}
    for label, stats in series.items():
        stats = sorted(stats, key=lambda s: s.json_metadata["p"])
        stats = [s for s in stats if s.errors > 0]
        if not stats:
            continue
        ps = [s.json_metadata["p"] for s in stats]
        y, lo, hi = zip(*(per_round(s) for s in stats))
        st = {"marker": "o"} | styles.get(label, {})
        line, = ax.plot(ps, y, label=label, **st)
        ax.fill_between(ps, lo, hi, alpha=0.18, color=line.get_color())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=9)
    return ax


def ibm_data(repo: Path, name: str) -> list[sinter.TaskStats]:
    """IBM's own published results that ship with the repo (examples/data/*.csv)."""
    return sinter.read_stats_from_csv_files(str(repo / "examples" / "data" / name))


def select(stats, **match) -> list[sinter.TaskStats]:
    """Filter sinter stats: decoder='relay-bp', basis='Z', xyz=False, d=12, ..."""
    out = []
    for s in stats:
        m = s.json_metadata
        ok = True
        for k, v in match.items():
            have = s.decoder if k == "decoder" else (m["circuit"][-1] if k == "basis" else m.get(k))
            ok &= have == v
        if ok:
            out.append(s)
    return out


# ==========================================================================
# Looking inside a circuit (used by every notebook)
# ==========================================================================
def show_head(c: stim.Circuit, n: int = 12, start: str | None = None) -> None:
    """Print n lines of the circuit file, as Stim writes it.

    start: begin at the first line that starts with this operation name (e.g. "CX"),
    to skip past long blocks of coordinates or resets.
    """
    lines = str(c).splitlines()
    first = 0
    if start:
        first = next((i for i, l in enumerate(lines) if l.strip().startswith(start)), 0)
        print(f"... (lines 1-{first} skipped)")
    for line in lines[first:first + n]:
        print(line[:140] + (" ..." if len(line) > 140 else ""))
    print(f"... ({len(lines)} lines in total)")


def gate_counts(c: stim.Circuit) -> pd.Series:
    """How many times each operation is applied (one CX on two qubits counts once).

    Noise channels (DEPOLARIZE, X_ERROR, ...) are counted too, which shows where
    the noise model puts errors. REPEAT blocks are unrolled first.
    """
    counts = collections.Counter()
    for inst in c.flattened():
        n = len(inst.targets_copy())
        if inst.name in ("CX", "CZ", "DEPOLARIZE2"):
            n //= 2
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            n = 1
        counts[inst.name] += n
    return pd.Series(counts).sort_values(ascending=False)


def plot_gate_counts(circuits: dict[str, stim.Circuit], title: str, save: str | None = None):
    """Grouped horizontal bars: operation counts per circuit (log scale)."""
    df = pd.DataFrame({k: gate_counts(c) for k, c in circuits.items()}).fillna(0).astype(int)
    df = df.drop(index=[x for x in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS") if x in df.index])
    df = df.loc[df.max(axis=1).sort_values().index]
    ax = df.plot.barh(figsize=(10, 0.45 * len(df) + 1.5), logx=True, width=0.8)
    ax.set_xlabel("how many times it appears (log scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(save)
    plt.show()
    return df


def savefig(name: str):
    """Save the current figure into ./figures at slide resolution (PNG, 200 dpi)."""
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(Path("figures") / name, dpi=200, bbox_inches="tight", facecolor="white")


def wilson(fails: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a failure fraction (Wilson, JASA 1927)."""
    p = fails / shots
    den = 1 + z * z / shots
    mid = (p + z * z / (2 * shots)) / den
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / den
    return mid - half, mid + half


def load_surface_results(path) -> pd.DataFrame:
    """Sinter CSV of the surface code runs -> one row per (d, p), repeated rows merged."""
    rows = collections.defaultdict(lambda: [0, 0])
    for s in sinter.read_stats_from_csv_files(str(path)):
        m = s.json_metadata
        rows[(m["d"], m["p"])][0] += s.shots
        rows[(m["d"], m["p"])][1] += s.errors
    df = pd.DataFrame([(d, p, n, e) for (d, p), (n, e) in rows.items()],
                      columns=["d", "p", "shots", "failures"]).sort_values(["d", "p"])
    df["P_L"] = df.failures / df.shots
    df[["ci_low", "ci_high"]] = [wilson(e, n) for e, n in zip(df.failures, df.shots)]
    return df.reset_index(drop=True)


# ==========================================================================
# Deltakit (notebook 03)
# ==========================================================================
def deltakit_client(token: str):
    """Connect to Riverlane's Deltakit cloud with your token; return the client or None.

    The token is never printed. Without a token (or without network access to
    deltakit.riverlane.com) the cloud cells are skipped and saved results are shown.
    """
    import os
    token = (token or os.environ.get("DELTAKIT_TOKEN", "")).strip()
    if not token:
        print("No token: cloud decoders will be skipped (saved results are shown if there are any).")
        return None
    try:
        from deltakit.explorer import Client
        Client.set_token(token)          # stores it for this machine and checks it with the server
        print("Deltakit cloud client ready.")
        return Client.get_instance()
    except Exception as e:               # wrong token, no internet, firewall ...
        print("Could not connect to the Deltakit cloud:", str(e)[:200])
        return None


def dk_noisy(dk_circuit, noise_model):
    """Add a Deltakit noise model to a Deltakit circuit.

    Returns (noisy Deltakit circuit, the same circuit as a standard stim.Circuit).
    Deltakit bundles its own build of Stim, so we re-read the text into the normal
    `stim` package to use it with Stim/Sinter tools.
    """
    from deltakit.explorer.qpu import QPU
    noisy = QPU(qubits=dk_circuit.qubits, noise_model=noise_model).compile_and_add_noise_to_circuit(dk_circuit)
    return noisy, stim.Circuit(str(noisy.as_stim_circuit()))


def count_failures(predictions, observables) -> int:
    """Shots where the decoder got at least one logical observable wrong."""
    return int((np.asarray(predictions, dtype=bool) != np.asarray(observables, dtype=bool)).any(axis=1).sum())


def cached_run(path, key: dict, run):
    """Look up `key` in the results CSV at `path`; if missing, call run() -> dict and save it.

    `run` may return None (e.g. no cloud token), in which case nothing is saved.
    Lets the cloud cells show results even when re-run without a token.
    """
    path = Path(path)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        hit = df
        for k, v in key.items():
            hit = hit[hit[k].astype(str) == str(v)]
        if len(hit):
            return hit.iloc[-1].to_dict()
    row = run()
    if row is None:
        return None
    row = {**key, **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return row


def cloud_decoder(cls, circuit: stim.Circuit, cloud, **parameters):
    """Build one of Deltakit's cloud decoder classes for a standard stim.Circuit.

    Deltakit's decoders recognise only Deltakit's own Stim build (deltakit_stim); given
    plain text they don't know how many logical observables there are and refuse to
    decode. So the circuit is converted to deltakit_stim first.
    """
    import deltakit_stim
    return cls(circuit=deltakit_stim.Circuit(str(circuit)), parameters=parameters, client=cloud)
