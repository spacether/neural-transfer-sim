"""Calibration experiment for mouse auditory transfer.

Fits a linear calibration transform on shared sounds at the cochlear
nucleus (CN) bottleneck, then compares calibrated transfer against direct
broadcast on held-out sounds.  Uses temporal encoding — each sound is a
500 ms stimulus processed through the temporal pipeline before
time-averaging.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from neural_transfer.models.mus_musculus import MouseAuditoryModel
from neural_transfer.models.mus_musculus import MouseAuditoryVariantGenerator
from neural_transfer.models.mus_musculus import (
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


def _get_cn_signal(
    model: MouseAuditoryModel, sound_name: str, intensity: float,
    avg_w: int,
) -> Dict[str, float]:
    stimulus = make_temporal_stimulus(sound_name, intensity)
    result = model.simulate_timesteps(stimulus)
    cn_avg = result["cn"][-avg_w:].mean(axis=0)
    return {n: float(cn_avg[i]) for i, n in enumerate(model.cn_names)}


def _apply_transform_timeseries(
    cn_ts: np.ndarray, transform: CalibrationTransform,
    cn_names: list,
) -> np.ndarray:
    """Apply calibration transform to full CN timeseries (vectorized)."""
    return transform.transform_array(cn_ts, col_names=cn_names)


def _ic_avg_from_temporal(result: Dict, avg_w: int) -> Dict[str, float]:
    ic_avg = result["ic"][-avg_w:].mean(axis=0)
    return {f"IC_{i:03d}": float(ic_avg[i]) for i in range(len(ic_avg))}


def _run_one_pair(
    sender: MouseAuditoryModel,
    receiver: MouseAuditoryModel,
) -> Dict:
    avg_w = temporal_avg_window()
    cn_names = sender.cn_names

    sender_cal = [_get_cn_signal(sender, name, intensity, avg_w) for name, intensity in CAL_SET]
    receiver_cal = [_get_cn_signal(receiver, name, intensity, avg_w) for name, intensity in CAL_SET]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    sound_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    direct_states: List[Dict[str, float]] = []
    cal_states: List[Dict[str, float]] = []
    natural_ic_ts: List[np.ndarray] = []
    direct_ic_ts: List[np.ndarray] = []
    cal_ic_ts: List[np.ndarray] = []
    direct_ok, cal_ok = 0, 0

    for name, intensity in HOLDOUT_SET:
        sound_names.append(name)

        stimulus = make_temporal_stimulus(name, intensity)
        sender_result = sender.simulate_timesteps(stimulus)
        sender_beh = sender_result["behavior"]
        sender_cn_ts = sender_result["cn"]

        recv_result = receiver.simulate_timesteps(stimulus)
        natural_states.append(_ic_avg_from_temporal(recv_result, avg_w))
        natural_ic_ts.append(recv_result["ic"])

        d_result = receiver.simulate_timesteps_from_cn(sender_cn_ts)
        direct_states.append(_ic_avg_from_temporal(d_result, avg_w))
        direct_ic_ts.append(d_result["ic"])
        direct_ok += d_result["behavior"] == sender_beh

        cal_cn_ts = _apply_transform_timeseries(sender_cn_ts, transform, cn_names)
        c_result = receiver.simulate_timesteps_from_cn(cal_cn_ts)
        cal_states.append(_ic_avg_from_temporal(c_result, avg_w))
        cal_ic_ts.append(c_result["ic"])
        cal_ok += c_result["behavior"] == sender_beh

    n = len(HOLDOUT_SET)
    out: Dict = {"n_holdout": n, "n_calibration": len(CAL_SET)}
    for prefix, states, ok, ic_ts in [
        ("direct", direct_states, direct_ok, direct_ic_ts),
        ("calibrated", cal_states, cal_ok, cal_ic_ts),
    ]:
        out[f"{prefix}_transfer_rate"] = ok / max(1, n)
        out[f"{prefix}_correlation"] = correlation(states, natural_states)
        out[f"{prefix}_mse"] = mean_mse(states, natural_states)
        out[f"{prefix}_rsa"] = representational_similarity(natural_states, states)
        out[f"{prefix}_sound_id"] = temporal_stimulus_identification(sound_names, natural_ic_ts, ic_ts)
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

    sender = MouseAuditoryModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = MouseAuditoryVariantGenerator(jitter_scale=jitter)
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
    parser = argparse.ArgumentParser(description="Mouse auditory calibration sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_sweep(seeds_per_level=args.seeds)

    print("=== Mouse: Auditory Calibration vs Broadcast ===")
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
            experiment="mouse_auditory_calibration_sweep",
            description="Mouse auditory calibration vs broadcast across brain differences",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
