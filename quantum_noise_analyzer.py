"""
Quantum Hardware Noise Analyzer
===============================
Compares ideal vs. noisy simulation of a 2-qubit Bell state using Qiskit + Qiskit Aer.

Steps
-----
1. Build a Bell-state circuit (H + CNOT).
2. Run an ideal simulation (1,000 shots by default).
3. Build a noise model (depolarizing gate errors + readout errors) and re-run.
4. Compare with metrics (total variation distance, Hellinger fidelity, Bell-state fidelity)
   and plot results.
5. Sweep the error rate to show how fidelity degrades as noise grows.
6. (Optional) Run on real IBM Quantum hardware with --hardware.

Install:
    pip install qiskit qiskit-aer matplotlib numpy
    pip install qiskit-ibm-runtime        # only needed for --hardware

Usage:
    python quantum_noise_analyzer.py
    python quantum_noise_analyzer.py --shots 4000 --p1 0.002 --p2 0.03 --readout 0.04
    python quantum_noise_analyzer.py --hardware      # needs a saved IBM Quantum account
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # works without a display; plots are saved to PNG files
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

STATES = ["00", "01", "10", "11"]
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Circuit
# --------------------------------------------------------------------------- #
def build_bell_circuit() -> QuantumCircuit:
    """Bell state (|00> + |11>)/sqrt(2): Hadamard on q0, then CNOT q0 -> q1."""
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# --------------------------------------------------------------------------- #
# 2/3. Simulation helpers
# --------------------------------------------------------------------------- #
def build_noise_model(p1: float, p2: float, p_readout: float) -> NoiseModel:
    """
    p1        : depolarizing error probability for single-qubit gates
    p2        : depolarizing error probability for two-qubit gates (CNOT)
    p_readout : probability a measured bit is flipped (symmetric readout error)
    """
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "x", "sx", "rz", "id"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    noise.add_all_qubit_readout_error(
        ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    )
    return noise


def run_counts(qc: QuantumCircuit, shots: int, noise_model: NoiseModel | None = None) -> dict:
    backend = AerSimulator(noise_model=noise_model, seed_simulator=SEED)
    # Transpile to a basis the noise model knows about
    tqc = transpile(qc, backend, basis_gates=["rz", "sx", "x", "cx", "id", "h"], seed_transpiler=SEED)
    return backend.run(tqc, shots=shots).result().get_counts()


def to_probs(counts: dict) -> np.ndarray:
    """Convert a counts dict to a probability vector ordered as STATES."""
    total = sum(counts.values())
    return np.array([counts.get(s, 0) / total for s in STATES])


# --------------------------------------------------------------------------- #
# 4. Metrics
# --------------------------------------------------------------------------- #
def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)) ** 2)


def bell_fidelity_proxy(p: np.ndarray) -> float:
    """Probability of landing in the correct correlated subspace {00, 11}."""
    return float(p[0] + p[3])


def ideal_probs() -> np.ndarray:
    return np.array([0.5, 0.0, 0.0, 0.5])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_comparison(series: dict, shots: int, path: str) -> None:
    """Grouped bar chart of probabilities for each named series."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(STATES))
    for i, (name, probs) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * width, probs, width, label=name)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"|{s}>" for s in STATES])
    ax.set_xlabel("Measurement outcome")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.65)
    ax.set_title(f"Bell state: ideal vs. noisy ({shots} shots)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_sweep(rates, fidelities, correlated, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates, fidelities, "o-", label="Hellinger fidelity vs. ideal")
    ax.plot(rates, correlated, "s--", label="P(00) + P(11)")
    ax.set_xlabel("Two-qubit (CNOT) depolarizing error rate")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Degradation of Bell-state quality with gate noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Noise sweep
# --------------------------------------------------------------------------- #
def noise_sweep(qc: QuantumCircuit, shots: int, p_readout: float):
    rates = np.linspace(0.0, 0.30, 11)
    fid, corr = [], []
    ideal = ideal_probs()
    for p2 in rates:
        nm = build_noise_model(p1=p2 / 10, p2=p2, p_readout=p_readout)
        probs = to_probs(run_counts(qc, shots, nm))
        fid.append(hellinger_fidelity(ideal, probs))
        corr.append(bell_fidelity_proxy(probs))
    return rates, np.array(fid), np.array(corr)


# --------------------------------------------------------------------------- #
# 6. Optional real hardware
# --------------------------------------------------------------------------- #
def run_on_hardware(qc: QuantumCircuit, shots: int) -> dict:
    """
    Requires: pip install qiskit-ibm-runtime and a saved IBM Quantum account, e.g.
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(token="YOUR_API_KEY", overwrite=True)
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
    print(f"Running on real hardware: {backend.name}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc)
    job = Sampler(mode=backend).run([isa], shots=shots)
    print(f"Job id: {job.job_id()} (waiting for results...)")
    result = job.result()[0]
    return result.data.c.get_counts()  # classical register is named "c" by default


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Quantum Hardware Noise Analyzer")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--p1", type=float, default=0.001, help="1-qubit gate depolarizing error")
    ap.add_argument("--p2", type=float, default=0.02, help="2-qubit gate depolarizing error")
    ap.add_argument("--readout", type=float, default=0.03, help="readout bit-flip probability")
    ap.add_argument("--hardware", action="store_true", help="also run on real IBM Quantum hardware")
    args = ap.parse_args()

    qc = build_bell_circuit()
    print("Bell-state circuit:")
    print(qc.draw(output="text"))

    # Ideal + noisy runs
    ideal_counts = run_counts(qc, args.shots)
    noise_model = build_noise_model(args.p1, args.p2, args.readout)
    noisy_counts = run_counts(qc, args.shots, noise_model)

    p_ideal = to_probs(ideal_counts)
    p_noisy = to_probs(noisy_counts)
    series = {"Ideal simulation": p_ideal, "Noisy simulation": p_noisy}

    hw_probs = None
    if args.hardware:
        try:
            hw_probs = to_probs(run_on_hardware(qc, args.shots))
            series["Real hardware"] = hw_probs
        except Exception as e:  # keep the rest of the analysis usable
            print(f"[!] Hardware run skipped: {e}")

    # Report
    print(f"\nShots: {args.shots}")
    print(f"Noise: 1q depol={args.p1}, 2q depol={args.p2}, readout flip={args.readout}\n")
    print(f"{'State':<8}" + "".join(f"{name:>20}" for name in series))
    for i, s in enumerate(STATES):
        print(f"|{s}>   " + "".join(f"{p[i]:>20.4f}" for p in series.values()))

    print("\nMetrics (vs. theoretical ideal [0.5, 0, 0, 0.5]):")
    ref = ideal_probs()
    for name, p in series.items():
        print(f"  {name:<18} TVD={total_variation_distance(ref, p):.4f}  "
              f"Hellinger fidelity={hellinger_fidelity(ref, p):.4f}  "
              f"P(00)+P(11)={bell_fidelity_proxy(p):.4f}")

    print("\nError outcomes (should be 0 for an ideal Bell state):")
    for name, p in series.items():
        print(f"  {name:<18} P(01)={p[1]:.4f}  P(10)={p[2]:.4f}")

    # Plots
    plot_comparison(series, args.shots, "bell_comparison.png")
    rates, fid, corr = noise_sweep(qc, args.shots, args.readout)
    plot_noise_sweep(rates, fid, corr, "noise_sweep.png")
    print("\nSaved plots: bell_comparison.png, noise_sweep.png")


if __name__ == "__main__":
    main()
