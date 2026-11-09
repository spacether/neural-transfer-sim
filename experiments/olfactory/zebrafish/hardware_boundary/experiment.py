"""Hardware boundary experiment for the zebrafish stage.

Reads and writes at the mitral cell bottleneck with calibration, then
sweeps hardware impairment axes — read noise, write precision, channel
dropout, and transmit delay — to find where transfer breaks down.

Uses tau=300ms for the zebrafish larval response timescale.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.zebrafish import (
    ZebrafishModel, ODOR_PROFILES, ODOR_VALENCE,
)
from neural_transfer.models.zebrafish.model import MITRAL_NAMES
from neural_transfer.models.zebrafish.variants import ZebrafishVariantGenerator
from neural_transfer.hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import (
    correlation, mean_mse, representational_similarity,
    odor_identification, bootstrap_ci,
)
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_hardware_boundary

ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]

DEFAULT_JITTER = 1.0
DEFAULT_SEEDS = 5
FISH_TAU_MS = 300.0

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _get_mitral_signal(model: ZebrafishModel, odor_name: str, intensity: float) -> Dict[str, float]:
    orn = model._orn_type_rates(odor_name, intensity)
    mc = model._mitral_rates(orn)
    return {n: float(mc[i]) for i, n in enumerate(MITRAL_NAMES)}


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
    sender: ZebrafishModel,
    receiver: ZebrafishModel,
    transform: CalibrationTransform,
    hardware: HardwareChain,
    rng: np.random.Generator,
) -> Dict[str, float]:
    odor_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    transferred_states: List[Dict[str, float]] = []
    successes = 0

    for name, intensity in HOLDOUT_SET:
        odor_names.append(name)

        sender_mc = _get_mitral_signal(sender, name, intensity)
        sender_layers = sender._forward(name, intensity)
        sender_beh = sender._behavior_from_output(sender_layers["output"])

        cal_mc = transform.transform(sender_mc)
        degraded = hardware.apply(cal_mc, rng)
        result = receiver.simulate_from_mitral(degraded)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, intensity)
        natural_states.append(receiver._layers_to_dict(natural_layers))
        transferred_states.append(result["neurons"])

    natural_dp = [{k: v for k, v in s.items() if k.startswith("Dp_")} for s in natural_states]
    transferred_dp = [{k: v for k, v in s.items() if k.startswith("Dp_")} for s in transferred_states]
    return {
        "transfer_rate": successes / len(HOLDOUT_SET),
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_dp, transferred_dp),
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

    sender = ZebrafishModel()
    gen = ZebrafishVariantGenerator(jitter_scale=jitter)

    sender_cal_mc = [_get_mitral_signal(sender, name, intensity) for name, intensity in CAL_SET]

    receivers = {}
    for seed in range(seeds):
        receiver = gen.generate_variant(sender, seed=seed)
        if not gen.sanity_check(receiver):
            continue
        receiver_cal_mc = [_get_mitral_signal(receiver, name, intensity) for name, intensity in CAL_SET]
        transform = CalibrationTransform()
        transform.fit(sender_cal_mc, receiver_cal_mc)
        receivers[seed] = (receiver, transform)

    def _sweep_axis(make_chain, param_values, param_name):
        rows = []
        for val in param_values:
            chain = make_chain(val)
            seed_metrics = []
            for seed, (receiver, transform) in receivers.items():
                rng = np.random.default_rng(seed * 1000 + hash(str(val)) % 10000)
                m = _run_one_setting(sender, receiver, transform, chain, rng)
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
        lambda bits: HardwareChain(ReadNoise(), WritePrecision(bits=bits, clip_max=250.0), ChannelDropout()),
        precision_levels, "precision_bits",
    )
    results["channel_dropout"] = _sweep_axis(
        lambda frac: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(drop_fraction=frac)),
        dropout_levels, "dropout_fraction",
    )
    results["transmit_delay"] = _sweep_axis(
        lambda ms: HardwareChain(
            ReadNoise(), WritePrecision(), ChannelDropout(),
            TransmitDelay(delay_ms=ms, tau_ms=FISH_TAU_MS),
        ),
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
            WritePrecision(bits=params["precision_bits"], clip_max=250.0),
            ChannelDropout(drop_fraction=params["dropout_fraction"]),
            TransmitDelay(delay_ms=params["delay_ms"], tau_ms=FISH_TAU_MS),
        )
        seed_metrics = []
        for seed, (receiver, transform) in receivers.items():
            rng = np.random.default_rng(seed * 1000 + step_idx)
            m = _run_one_setting(sender, receiver, transform, chain, rng)
            seed_metrics.append(m)
        if seed_metrics:
            row: Dict = {"step": step_idx, **params}
            row.update(_agg_with_ci(seed_metrics))
            combined_rows.append(row)
    results["combined"] = combined_rows

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish hardware boundary experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_hardware_sweep(seeds=args.seeds)

    for axis, rows in results.items():
        print(f"\n  --- {axis} ---")
        if not rows:
            continue
        param_key = [k for k in rows[0] if k not in ("transfer_rate", "correlation", "mse", "rsa", "odor_id", "step")][0]
        print(f"  {param_key:>8}  {'rate':>6}  {'corr':>6}  {'mse':>9}  {'odor_id':>7}")
        print("  " + "-" * 50)
        for r in rows:
            print(f"  {r[param_key]:8.2f}  {r['transfer_rate']:6.3f}  "
                  f"{r['correlation']:6.3f}  {r['mse']:9.2f}  {r['odor_id']:7.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for axis, rows in results.items():
            json_path = write_results(
                experiment=f"zebrafish_hardware_{axis}",
                description=f"Zebrafish: hardware boundary — {axis}",
                parameters={"seeds": args.seeds},
                results=rows,
                output_dir=RESULTS_DIR,
            )
        plot_path = plot_hardware_boundary(results, RESULTS_DIR / "hardware_boundary_zebrafish.png")
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
