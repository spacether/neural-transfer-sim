"""Calibration experiment for adult Drosophila auditory transfer.

Fits a linear calibration transform on shared sounds at the AMMC
bottleneck, then compares calibrated transfer against direct broadcast
on held-out sounds.  Uses temporal encoding — each sound is a 500 ms
stimulus processed through the temporal pipeline before time-averaging.

Uses connectome-derived auditory model (FlyWire v783).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_adult import AuditoryAdultModel
from neural_transfer.models.drosophila_adult import AuditoryAdultVariantGenerator
from neural_transfer.models.drosophila_adult import (
    SOUND_PROFILES, SOUND_VALENCE,
    make_temporal_stimulus, temporal_avg_window,
)
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import (
    bootstrap_ci,
    correlation,
    mean_mse,
    representational_similarity,
    temporal_stimulus_identification,
)
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_calibration_comparison

ALL_SOUND_NAMES = list(SOUND_PROFILES.keys())

CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_SOUND_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_SOUND_NAMES for i in HOLDOUT_INTENSITIES]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "sound_id")


def _get_ammc_signal(
    model: AuditoryAdultModel, sound_name: str, intensity: float,
    avg_w: int,
) -> Dict[str, float]:
    stimulus = make_temporal_stimulus(sound_name, intensity)
    result = model.simulate_timesteps(stimulus)
    ammc_avg = result["ammc"][-avg_w:].mean(axis=0)
    return {n: float(ammc_avg[i]) for i, n in enumerate(model.ammc_names)}


def _apply_transform_timeseries(
    ammc_ts: np.ndarray, transform: CalibrationTransform,
    ammc_names: list,
) -> np.ndarray:
    """Apply calibration transform to full AMMC timeseries (vectorized)."""
    return transform.transform_array(ammc_ts, col_names=ammc_names)


def _wv_avg_from_temporal(result: Dict, avg_w: int) -> Dict[str, float]:
    wv_avg = result["wedvlp"][-avg_w:].mean(axis=0)
    return {f"WV_{i:03d}": float(wv_avg[i]) for i in range(len(wv_avg))}


def _run_one_pair(
    sender: AuditoryAdultModel,
    receiver: AuditoryAdultModel,
) -> Dict:
    avg_w = temporal_avg_window()
    ammc_names = sender.ammc_names

    sender_cal = [_get_ammc_signal(sender, name, intensity, avg_w) for name, intensity in CAL_SET]
    receiver_cal = [_get_ammc_signal(receiver, name, intensity, avg_w) for name, intensity in CAL_SET]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    sound_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    direct_states: List[Dict[str, float]] = []
    cal_states: List[Dict[str, float]] = []
    natural_wv_ts: List[np.ndarray] = []
    direct_wv_ts: List[np.ndarray] = []
    cal_wv_ts: List[np.ndarray] = []
    direct_ok, cal_ok = 0, 0

    for name, intensity in HOLDOUT_SET:
        sound_names.append(name)

        stimulus = make_temporal_stimulus(name, intensity)
        sender_result = sender.simulate_timesteps(stimulus)
        sender_beh = sender_result["behavior"]
        sender_ammc_ts = sender_result["ammc"]

        recv_result = receiver.simulate_timesteps(stimulus)
        natural_states.append(_wv_avg_from_temporal(recv_result, avg_w))
        natural_wv_ts.append(recv_result["wedvlp"])

        d_result = receiver.simulate_timesteps_from_ammc(sender_ammc_ts)
        direct_states.append(_wv_avg_from_temporal(d_result, avg_w))
        direct_wv_ts.append(d_result["wedvlp"])
        direct_ok += d_result["behavior"] == sender_beh

        cal_ammc_ts = _apply_transform_timeseries(sender_ammc_ts, transform, ammc_names)
        c_result = receiver.simulate_timesteps_from_ammc(cal_ammc_ts)
        cal_states.append(_wv_avg_from_temporal(c_result, avg_w))
        cal_wv_ts.append(c_result["wedvlp"])
        cal_ok += c_result["behavior"] == sender_beh

    n = len(HOLDOUT_SET)
    out: Dict = {"n_holdout": n, "n_calibration": len(CAL_SET)}
    for prefix, states, ok, wv_ts in [
        ("direct", direct_states, direct_ok, direct_wv_ts),
        ("calibrated", cal_states, cal_ok, cal_wv_ts),
    ]:
        out[f"{prefix}_transfer_rate"] = ok / max(1, n)
        out[f"{prefix}_correlation"] = correlation(states, natural_states)
        out[f"{prefix}_mse"] = mean_mse(states, natural_states)
        out[f"{prefix}_rsa"] = representational_similarity(natural_states, states)
        out[f"{prefix}_sound_id"] = temporal_stimulus_identification(sound_names, natural_wv_ts, wv_ts)
    return out


def _agg_with_ci(pairs: List[Dict]) -> Dict:
    row: Dict = {}
    for prefix in ("direct", "calibrated"):
        for k in METRIC_KEYS:
            key = f"{prefix}_{k}"
            vals = [p[key] for p in pairs]
            mean, lo, hi = bootstrap_ci(vals)
            row[key] = mean
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
    return row


def run_calibration_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = AuditoryAdultModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = AuditoryAdultVariantGenerator(jitter_scale=jitter)
        pairs: List[Dict] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            pairs.append(_run_one_pair(sender, receiver))

        if not pairs:
            continue

        row = _agg_with_ci(pairs)
        row["jitter_scale"] = jitter
        row["valid_seeds"] = len(pairs)
        results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly auditory calibration sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_sweep(seeds_per_level=args.seeds)

    print("=== Adult Drosophila: Auditory Calibration vs Broadcast ===")
    print(f"{'jitter':>6}  {'d_rate':>6}  {'c_rate':>6}  {'d_corr':>6}  {'c_corr':>6}  {'d_sid':>6}  {'c_sid':>6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['jitter_scale']:6.2f}  {r['direct_transfer_rate']:6.3f}  "
            f"{r['calibrated_transfer_rate']:6.3f}  "
            f"{r['direct_correlation']:6.3f}  {r['calibrated_correlation']:6.3f}  "
            f"{r['direct_sound_id']:6.3f}  {r['calibrated_sound_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="adult_auditory_calibration_sweep",
            description="Adult Drosophila auditory calibration vs broadcast across brain differences",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
