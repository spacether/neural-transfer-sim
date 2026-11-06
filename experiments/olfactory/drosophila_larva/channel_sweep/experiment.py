"""Channel sweep experiment for the Drosophila larva stage.

Measures transfer quality versus number of connected neurons at each
level of brain difference.  At each channel count, the experiment
randomly selects which ORNs to include and zeros out the rest.

Uses the connectome-based larva model (Winding et al. 2023).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
from neural_transfer.models.drosophila_larva import ConnectomeLarvaVariantGenerator
from neural_transfer.models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE
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
from neural_transfer.analysis.plotting import plot_channel_sweep

ODOR_SET: List[Dict[str, object]] = [
    {"odor_name": name, "odor_intensity": 1.0}
    for name in ODOR_PROFILES
]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _mask_signal(signal: Dict[str, float], active_orns: set) -> Dict[str, float]:
    return {k: (v if k in active_orns else 0.0) for k, v in signal.items()}


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def run_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    """Sweep channel count x brain difference.

    For each (channel_count, jitter_level), randomly pick which ORNs to
    include, transfer the masked signal, and measure quality.
    """
    if channel_counts is None:
        channel_counts = [1, 2, 3, 5, 7, 10, 14, 18, 21]
    if jitter_levels is None:
        jitter_levels = [0.0, 0.5, 1.0]

    sender = ConnectomeLarvaModel()
    orn_order = sender._orn_order
    results: List[Dict] = []

    for n_channels in channel_counts:
        for jitter in jitter_levels:
            gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)
            all_metrics: List[Dict[str, float]] = []

            for ch_seed in range(channel_seeds):
                ch_rng = np.random.default_rng(ch_seed)
                active = set(ch_rng.choice(orn_order, size=min(n_channels, len(orn_order)), replace=False))

                for seed in range(seeds_per_level):
                    receiver = gen.generate_variant(sender, seed=seed)
                    if not gen.sanity_check(receiver):
                        continue

                    odor_names: List[str] = []
                    natural_states: List[Dict[str, float]] = []
                    transferred_states: List[Dict[str, float]] = []
                    successes = 0

                    for odor in ODOR_SET:
                        name = odor["odor_name"]
                        odor_names.append(name)

                        sender_orn = sender._orn_rates(name, 1.0)
                        full_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(orn_order)}
                        masked = _mask_signal(full_signal, active)

                        sender_layers = sender._forward(name, 1.0)
                        sender_beh = sender._behavior_from_output(sender_layers["output"])

                        result = receiver.simulate_from_receptors(masked)
                        if result["behavior"] == sender_beh:
                            successes += 1

                        natural_layers = receiver._forward(name, 1.0)
                        natural_states.append(receiver._layers_to_dict(natural_layers))
                        transferred_states.append(result["neurons"])

                    all_metrics.append({
                        "transfer_rate": successes / len(ODOR_SET),
                        "correlation": correlation(transferred_states, natural_states),
                        "mse": mean_mse(transferred_states, natural_states),
                        "rsa": representational_similarity(natural_states, transferred_states),
                        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
                    })

            if not all_metrics:
                continue

            row: Dict = {
                "n_channels": n_channels,
                "jitter_scale": jitter,
                "valid_trials": len(all_metrics),
            }
            row.update(_agg_with_ci(all_metrics))
            results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva channel sweep")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--channel-seeds", type=int, default=3)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_channel_sweep(
        seeds_per_level=args.seeds,
        channel_seeds=args.channel_seeds,
    )

    print("=== Drosophila Larva: Channel Sweep ===")
    print(f"{'ch':>3}  {'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'odor_id':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['n_channels']:3d}  {r['jitter_scale']:6.2f}  "
            f"{r['transfer_rate']:6.3f}  {r['correlation']:6.3f}  "
            f"{r['mse']:8.2f}  {r['rsa']:6.3f}  {r['odor_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="larva_channel_sweep",
            description="Drosophila larva transfer quality vs number of connected ORNs",
            parameters={
                "seeds_per_level": args.seeds,
                "channel_seeds": args.channel_seeds,
            },
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
