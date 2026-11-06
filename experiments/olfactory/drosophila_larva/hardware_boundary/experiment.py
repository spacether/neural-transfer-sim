"""Hardware boundary experiment for the Drosophila larva stage.

Sweeps hardware impairment axes — read noise, write precision, channel
dropout, and transmit delay — to find where ORN-to-ORN transfer breaks
down in the larval olfactory circuit.

Uses the connectome-based larva model (Winding et al. 2023).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
from neural_transfer.models.drosophila_larva import ConnectomeLarvaVariantGenerator
from neural_transfer.models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE
from neural_transfer.hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
    bootstrap_ci,
)
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_hardware_boundary

ODOR_SET: List[Dict[str, object]] = [
    {"odor_name": name, "odor_intensity": intensity}
    for name in ODOR_PROFILES
    for intensity in (0.5, 1.0, 1.5)
]

DEFAULT_JITTER = 0.75
DEFAULT_SEEDS = 5

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def _run_one_setting(
    sender: ConnectomeLarvaModel,
    receiver: ConnectomeLarvaModel,
    hardware: HardwareChain,
    rng: np.random.Generator,
) -> Dict[str, float]:
    orn_order = sender._orn_order
    odor_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    transferred_states: List[Dict[str, float]] = []
    successes = 0

    for odor in ODOR_SET:
        name = odor["odor_name"]
        intensity = odor["odor_intensity"]
        odor_names.append(name)

        sender_orn = sender._orn_rates(name, intensity)
        sender_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(orn_order)}
        sender_layers = sender._forward(name, intensity)
        sender_beh = sender._behavior_from_output(sender_layers["output"])

        degraded = hardware.apply(sender_signal, rng)
        result = receiver.simulate_from_receptors(degraded)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, intensity)
        natural_states.append(receiver._layers_to_dict(natural_layers))
        transferred_states.append(result["neurons"])

    return {
        "transfer_rate": successes / len(ODOR_SET),
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def run_hardware_sweep(
    jitter: float = DEFAULT_JITTER,
    seeds: int = DEFAULT_SEEDS,
    noise_levels: List[float] | None = None,
    precision_levels: List[float] | None = None,
    dropout_levels: List[float] | None = None,
    delay_levels: List[float] | None = None,
) -> Dict[str, List[Dict]]:
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    if precision_levels is None:
        precision_levels = [float("inf"), 8, 6, 5, 4, 3, 2, 1]
    if dropout_levels is None:
        dropout_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    if delay_levels is None:
        delay_levels = [0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]

    sender = ConnectomeLarvaModel()
    gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)

    def _sweep_axis(make_chain, param_values, param_name):
        rows = []
        for val in param_values:
            chain = make_chain(val)
            seed_metrics = []
            for seed in range(seeds):
                receiver = gen.generate_variant(sender, seed=seed)
                if not gen.sanity_check(receiver):
                    continue
                rng = np.random.default_rng(seed * 1000 + hash(str(val)) % 10000)
                m = _run_one_setting(sender, receiver, chain, rng)
                seed_metrics.append(m)
            if not seed_metrics:
                continue
            row = {param_name: val, "n_seeds": len(seed_metrics)}
            row.update(_agg_with_ci(seed_metrics))
            rows.append(row)
        return rows

    results = {}

    results["read_noise"] = _sweep_axis(
        lambda sigma: HardwareChain(ReadNoise(sigma=sigma), WritePrecision(), ChannelDropout()),
        noise_levels, "noise_sigma",
    )
    results["write_precision"] = _sweep_axis(
        lambda bits: HardwareChain(ReadNoise(), WritePrecision(bits=bits, clip_max=100.0), ChannelDropout()),
        precision_levels, "precision_bits",
    )
    results["channel_dropout"] = _sweep_axis(
        lambda frac: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(drop_fraction=frac)),
        dropout_levels, "dropout_fraction",
    )
    results["transmit_delay"] = _sweep_axis(
        lambda ms: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(), TransmitDelay(delay_ms=ms, tau_ms=500.0)),
        delay_levels, "delay_ms",
    )

    combined_steps = [
        {"noise_sigma": 0.0, "precision_bits": float("inf"), "dropout_fraction": 0.0, "delay_ms": 0.0},
        {"noise_sigma": 0.05, "precision_bits": 8, "dropout_fraction": 0.05, "delay_ms": 0.1},
        {"noise_sigma": 0.1, "precision_bits": 6, "dropout_fraction": 0.1, "delay_ms": 1.0},
        {"noise_sigma": 0.2, "precision_bits": 5, "dropout_fraction": 0.15, "delay_ms": 5.0},
        {"noise_sigma": 0.3, "precision_bits": 4, "dropout_fraction": 0.2, "delay_ms": 20.0},
        {"noise_sigma": 0.5, "precision_bits": 3, "dropout_fraction": 0.3, "delay_ms": 100.0},
        {"noise_sigma": 0.75, "precision_bits": 2, "dropout_fraction": 0.4, "delay_ms": 500.0},
        {"noise_sigma": 1.0, "precision_bits": 1, "dropout_fraction": 0.5, "delay_ms": 2000.0},
    ]
    combined_rows = []
    for step_idx, params in enumerate(combined_steps):
        chain = HardwareChain(
            ReadNoise(sigma=params["noise_sigma"]),
            WritePrecision(bits=params["precision_bits"], clip_max=100.0),
            ChannelDropout(drop_fraction=params["dropout_fraction"]),
            TransmitDelay(delay_ms=params["delay_ms"]),
        )
        seed_metrics = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            rng = np.random.default_rng(seed * 1000 + step_idx)
            m = _run_one_setting(sender, receiver, chain, rng)
            seed_metrics.append(m)
        if seed_metrics:
            row: Dict = {"step": step_idx, **params}
            row.update(_agg_with_ci(seed_metrics))
            combined_rows.append(row)
    results["combined"] = combined_rows

    return results


RESULTS_DIR = Path(__file__).parent / "results"

AXIS_LABELS = {
    "read_noise": ("noise_sigma", "sigma"),
    "write_precision": ("precision_bits", "bits"),
    "channel_dropout": ("dropout_fraction", "drop"),
    "transmit_delay": ("delay_ms", "ms"),
    "combined": ("step", "step"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva hardware boundary sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.75)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = run_hardware_sweep(jitter=args.jitter, seeds=args.seeds)

    print("=== Drosophila Larva: Hardware Boundary ===")
    for axis, rows in all_results.items():
        param_key, short = AXIS_LABELS[axis]
        print(f"\n  --- {axis} ---")
        print(f"  {short:>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'odor_id':>6}")
        print(f"  {'-' * 48}")
        for row in rows:
            val = row[param_key]
            val_str = f"{val:6.2f}" if isinstance(val, float) and val != float("inf") else f"{val!s:>6}"
            print(
                f"  {val_str}  {row['transfer_rate']:6.3f}  "
                f"{row['correlation']:6.3f}  {row['mse']:8.2f}  "
                f"{row['odor_id']:6.3f}"
            )

    if not args.no_save:
        for axis, rows in all_results.items():
            json_path = write_results(
                experiment=f"larva_hardware_{axis}",
                description=f"Drosophila larva hardware boundary: {axis} sweep",
                parameters={"axis": axis, "jitter": args.jitter, "seeds": args.seeds},
                results=rows,
                output_dir=RESULTS_DIR,
            )
            print(f"  {axis} results: {json_path}")

        plot_path = RESULTS_DIR / "hardware_boundary_larva.png"
        plot_hardware_boundary(all_results, plot_path)
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
