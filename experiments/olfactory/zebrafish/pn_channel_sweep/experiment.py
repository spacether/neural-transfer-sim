"""Mitral cell channel sweep for the zebrafish stage.

2D sweep of read/write mitral cell counts with calibration.  The
zebrafish equivalent of the fly PN channel sweep — tests how many
mitral cells (the bulb output) need to be connected on each side.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from neural_transfer.models.zebrafish import (
    ZebrafishModel, ODOR_PROFILES, ODOR_VALENCE, OR_TYPES, MITRAL_NAMES, N_MITRAL,
)
from neural_transfer.models.zebrafish.variants import ZebrafishVariantGenerator
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import correlation, mean_mse, odor_identification
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_pn_channel_sweep

ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]


def _mitral_signal(model: ZebrafishModel, odor_name: str, intensity: float) -> Dict[str, float]:
    layers = model._forward(odor_name, intensity)
    return {mc: float(layers["mitral"][i]) for i, mc in enumerate(MITRAL_NAMES)}


def _mask_to_set(signal: Dict[str, float], active: Set[str]) -> Dict[str, float]:
    return {k: (v if k in active else 0.0) for k, v in signal.items()}


def run_mitral_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter: float = 1.0,
    seeds: int = 3,
) -> List[Dict]:
    if channel_counts is None:
        channel_counts = [1, 5, 10, 15, 20, 30, 40, 50, 60, 80]

    sender = ZebrafishModel()
    gen = ZebrafishVariantGenerator(jitter_scale=jitter)
    results: List[Dict] = []

    for n_read in channel_counts:
        for n_write in channel_counts:
            all_metrics: List[Dict[str, float]] = []

            for seed in range(seeds):
                receiver = gen.generate_variant(sender, seed=seed)
                if not gen.sanity_check(receiver):
                    continue

                rng = np.random.default_rng(seed * 100)
                read_set = set(rng.choice(MITRAL_NAMES, size=min(n_read, N_MITRAL), replace=False))
                write_set = set(rng.choice(MITRAL_NAMES, size=min(n_write, N_MITRAL), replace=False))

                sender_cal = [_mask_to_set(_mitral_signal(sender, n, i), read_set) for n, i in CAL_SET]
                receiver_cal = [_mask_to_set(_mitral_signal(receiver, n, i), write_set) for n, i in CAL_SET]

                transform = CalibrationTransform()
                transform.fit(sender_cal, receiver_cal)

                odor_names: List[str] = []
                natural_dp: List[Dict[str, float]] = []
                transferred_dp: List[Dict[str, float]] = []
                successes = 0

                for name, inten in HOLDOUT_SET:
                    odor_names.append(name)

                    sender_layers = sender._forward(name, inten)
                    sender_beh = sender._behavior_from_output(sender_layers["output"])

                    read_sig = _mask_to_set(_mitral_signal(sender, name, inten), read_set)
                    cal_sig = transform.transform(read_sig)
                    filtered = {k: v for k, v in cal_sig.items() if k in write_set}
                    result = receiver.simulate_from_mitral(filtered)
                    if result["behavior"] == sender_beh:
                        successes += 1

                    natural_layers = receiver._forward(name, inten)
                    nat_full = receiver._layers_to_dict(natural_layers)
                    xfer_full = result["neurons"]
                    natural_dp.append({k: v for k, v in nat_full.items() if k.startswith("Dp_")})
                    transferred_dp.append({k: v for k, v in xfer_full.items() if k.startswith("Dp_")})

                n = len(HOLDOUT_SET)
                all_metrics.append({
                    "transfer_rate": successes / max(1, n),
                    "correlation": correlation(transferred_dp, natural_dp),
                    "mse": mean_mse(transferred_dp, natural_dp),
                    "odor_id": odor_identification(odor_names, natural_dp, transferred_dp),
                })

            if not all_metrics:
                continue

            results.append({
                "n_read": n_read,
                "n_write": n_write,
                **{
                    key: float(np.mean([m[key] for m in all_metrics]))
                    for key in ("transfer_rate", "correlation", "mse", "odor_id")
                },
            })

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish mitral cell channel sweep")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_mitral_channel_sweep(seeds=args.seeds)

    print("=== Zebrafish: Mitral Cell Channel Sweep ===")
    print(f"{'read':>5}  {'write':>5}  {'rate':>6}  {'corr':>6}  {'odor_id':>6}")
    print("-" * 35)
    for r in results:
        print(f"{r['n_read']:5d}  {r['n_write']:5d}  {r['transfer_rate']:6.3f}  "
              f"{r['correlation']:6.3f}  {r['odor_id']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="zebrafish_mitral_channel_sweep",
            description="Zebrafish: 2D sweep of read/write mitral cell counts",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_pn_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
