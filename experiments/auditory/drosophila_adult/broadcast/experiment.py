"""Direct broadcast baseline for adult Drosophila auditory transfer.

Copies sender AMMC bottleneck activity to receiver AMMC layer with no
learned translation, across a range of brain-difference levels.
Uses temporal encoding — each sound is a 500 ms stimulus processed
through the sender's temporal pipeline before time-averaging.

Uses the connectome-based auditory model (FlyWire v783, Dorkenwald et al. 2024).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_adult import AuditoryAdultModel
from neural_transfer.models.drosophila_adult import AuditoryAdultVariantGenerator
from neural_transfer.models.drosophila_adult import (
    SOUND_PROFILES, SOUND_VALENCE, SOUND_SUBTYPES,
    make_temporal_stimulus, temporal_avg_window,
)
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    temporal_stimulus_identification,
    bootstrap_ci,
)
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_receptor_to_receptor_sweep


SOUND_SET: List[Dict[str, object]] = [
    {"sound_name": name, "sound_intensity": intensity}
    for name in SOUND_PROFILES
    for intensity in (0.5, 1.0, 1.5)
]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "sound_id")


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def run_broadcast_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
    subtypes: tuple | None = SOUND_SUBTYPES,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = AuditoryAdultModel(subtypes=subtypes)
    ammc_names = sender.ammc_names
    avg_w = temporal_avg_window()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = AuditoryAdultVariantGenerator(jitter_scale=jitter)
        seed_metrics: List[Dict[str, float]] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue

            sound_names: List[str] = []
            natural_states: List[Dict[str, float]] = []
            transferred_states: List[Dict[str, float]] = []
            natural_wv_ts: List[np.ndarray] = []
            transferred_wv_ts: List[np.ndarray] = []
            successes = 0

            for stim in SOUND_SET:
                name = stim["sound_name"]
                intensity = stim["sound_intensity"]
                sound_names.append(name)

                stimulus = make_temporal_stimulus(name, intensity)
                sender_result = sender.simulate_timesteps(stimulus)
                sender_beh = sender_result["behavior"]
                sender_ammc_ts = sender_result["ammc"]

                trans_result = receiver.simulate_timesteps_from_ammc(sender_ammc_ts)
                if trans_result["behavior"] == sender_beh:
                    successes += 1

                recv_result = receiver.simulate_timesteps(stimulus)
                recv_wv_avg = recv_result["wedvlp"][-avg_w:].mean(axis=0)
                natural_state = {f"WV_{i:03d}": float(recv_wv_avg[i]) for i in range(len(recv_wv_avg))}
                natural_states.append(natural_state)
                natural_wv_ts.append(recv_result["wedvlp"])
                trans_wv_avg = trans_result["wedvlp"][-avg_w:].mean(axis=0)
                trans_state = {f"WV_{i:03d}": float(trans_wv_avg[i]) for i in range(len(trans_wv_avg))}
                transferred_states.append(trans_state)
                transferred_wv_ts.append(trans_result["wedvlp"])

            if len(SOUND_SET) > 0:
                seed_metrics.append({
                    "transfer_rate": successes / len(SOUND_SET),
                    "correlation": correlation(transferred_states, natural_states),
                    "mse": mean_mse(transferred_states, natural_states),
                    "rsa": representational_similarity(natural_states, transferred_states),
                    "sound_id": temporal_stimulus_identification(sound_names, natural_wv_ts, transferred_wv_ts),
                })

        if not seed_metrics:
            continue

        row: Dict = {
            "jitter_scale": jitter,
            "valid_seeds": len(seed_metrics),
        }
        row.update(_agg_with_ci(seed_metrics))
        results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def _print_table(label: str, results: list) -> None:
    print(f"\n{label}")
    print(f"{'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'snd_id':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['jitter_scale']:6.2f}  {r['transfer_rate']:6.3f}  "
            f"{r['correlation']:6.3f}  {r['mse']:8.2f}  "
            f"{r['rsa']:6.3f}  {r['sound_id']:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly auditory broadcast baseline sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    configs = [
        ("A+B sound-sensitive (359 JO)", SOUND_SUBTYPES, "adult_auditory_broadcast_ab"),
        ("All subtypes (1103 JO)", None, "adult_auditory_broadcast_all"),
    ]

    for label, subtypes, name in configs:
        results = run_broadcast_sweep(seeds_per_level=args.seeds, subtypes=subtypes)
        n_jo = results[0].get("n_jo", "?") if results else "?"
        _print_table(f"=== {label} ===", results)

        if not args.no_save:
            json_path = write_results(
                experiment=name,
                description=f"Adult Drosophila auditory broadcast ({label})",
                parameters={"seeds_per_level": args.seeds, "subtypes": list(subtypes) if subtypes else "all"},
                results=results,
                output_dir=RESULTS_DIR,
            )
            plot_path = plot_receptor_to_receptor_sweep(results, json_path.with_suffix(".png"))
            print(f"Results: {json_path}")
            print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
