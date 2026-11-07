"""Channel sweep experiment for adult Drosophila auditory transfer.

Sweeps the number of AMMC bottleneck channels connected between sender
and receiver, with calibration.  Reads N AMMC neurons from the sender,
fits a calibration transform, and writes to N AMMC neurons of the receiver.

With 150 AMMC neurons (the auditory bottleneck), this is directly
analogous to the olfactory channel sweeps that sweep PN or mitral counts.

Sender temporal simulations are precomputed once and reused across all
channel/seed combinations to avoid redundant work.

Uses the connectome-based auditory model (FlyWire v783, Dorkenwald et al. 2024).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np

from neural_transfer.models.drosophila_adult import AuditoryAdultModel
from neural_transfer.models.drosophila_adult import AuditoryAdultVariantGenerator
from neural_transfer.models.drosophila_adult import (
    SOUND_PROFILES, SOUND_VALENCE,
    make_temporal_stimulus, temporal_avg_window,
)
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    temporal_stimulus_identification,
    bootstrap_ci,
)
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_channel_sweep

ALL_SOUND_NAMES = list(SOUND_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_SOUND_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_SOUND_NAMES for i in HOLDOUT_INTENSITIES]

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


def _wv_avg_from_temporal(result: Dict, avg_w: int) -> Dict[str, float]:
    wv_avg = result["wedvlp"][-avg_w:].mean(axis=0)
    return {f"WV_{i:03d}": float(wv_avg[i]) for i in range(len(wv_avg))}


def _precompute_sender(
    sender: AuditoryAdultModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> List[Dict]:
    """Run sender temporal sims once, return AMMC timeseries + time-averaged AMMC + behavior."""
    cache = []
    ammc_names = sender.ammc_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = sender.simulate_timesteps(stimulus)
        ammc_avg = result["ammc"][-avg_w:].mean(axis=0)
        full_ammc = {n: float(ammc_avg[i]) for i, n in enumerate(ammc_names)}
        cache.append({
            "ammc": full_ammc,
            "ammc_ts": result["ammc"],
            "behavior": result["behavior"],
        })
    return cache


def _precompute_receiver(
    receiver: AuditoryAdultModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[np.ndarray]]:
    """Run receiver temporal sims once, return time-averaged AMMC, WV, and full WV timeseries."""
    ammc_list = []
    wv_list = []
    wv_ts_list: List[np.ndarray] = []
    r_ammc_names = receiver.ammc_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = receiver.simulate_timesteps(stimulus)
        ammc_avg = result["ammc"][-avg_w:].mean(axis=0)
        ammc_list.append({n: float(ammc_avg[i]) for i, n in enumerate(r_ammc_names)})
        wv_list.append(_wv_avg_from_temporal(result, avg_w))
        wv_ts_list.append(result["wedvlp"])
    return ammc_list, wv_list, wv_ts_list


def _apply_channel_cal_timeseries(
    ammc_ts: np.ndarray, ammc_names: List[str],
    active_idx: List[int], transform: CalibrationTransform,
    active_ammc: List[str], active_set: set,
    baseline: float,
) -> np.ndarray:
    """Apply channel selection + calibration to AMMC timeseries (vectorized)."""
    n_steps, n_ammc = ammc_ts.shape
    # Extract active channels, calibrate, write back
    sub_ts = ammc_ts[:, active_idx]
    cal_out = transform.transform_array(sub_ts, col_names=active_ammc)
    out = np.full((n_steps, n_ammc), baseline)
    out[:, active_idx] = cal_out
    return out


def run_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    if channel_counts is None:
        channel_counts = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150]
    if jitter_levels is None:
        jitter_levels = [0.0, 0.5, 1.0]

    sender = AuditoryAdultModel()
    n_ammc = sender.n_ammc
    ammc_names = sender.ammc_names
    avg_w = temporal_avg_window()

    sender_cal_cache = _precompute_sender(sender, CAL_SET, avg_w)
    sender_hold_cache = _precompute_sender(sender, HOLDOUT_SET, avg_w)

    recv_by_jitter: Dict[float, Dict[int, Tuple]] = {}
    for jitter in jitter_levels:
        gen = AuditoryAdultVariantGenerator(jitter_scale=jitter)
        recv_by_jitter[jitter] = {}
        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            recv_cal_ammc, _, _ = _precompute_receiver(receiver, CAL_SET, avg_w)
            _, recv_hold_wv, recv_hold_wv_ts = _precompute_receiver(receiver, HOLDOUT_SET, avg_w)
            recv_by_jitter[jitter][seed] = (receiver, recv_cal_ammc, recv_hold_wv, recv_hold_wv_ts)

    results: List[Dict] = []

    for n_channels in channel_counts:
        for jitter in jitter_levels:
            all_metrics: List[Dict[str, float]] = []

            for ch_seed in range(channel_seeds):
                ch_rng = np.random.default_rng(ch_seed)
                active_idx = sorted(ch_rng.choice(n_ammc, size=min(n_channels, n_ammc), replace=False).tolist())
                active_ammc = [ammc_names[i] for i in active_idx]
                active_set = set(active_ammc)

                sender_cal = [{n: sc["ammc"][n] for n in active_ammc} for sc in sender_cal_cache]

                for seed, (receiver, recv_cal_ammc, recv_hold_wv, recv_hold_wv_ts) in recv_by_jitter[jitter].items():
                    receiver_cal = [{n: ra[n] for n in active_ammc if n in ra}
                                    for ra in recv_cal_ammc]

                    transform = CalibrationTransform()
                    transform.neurons = tuple(sorted(active_ammc))
                    transform.fit(sender_cal, receiver_cal)

                    sound_names: List[str] = []
                    natural_states: List[Dict[str, float]] = []
                    transferred_states: List[Dict[str, float]] = []
                    natural_wv_ts: List[np.ndarray] = []
                    transferred_wv_ts: List[np.ndarray] = []
                    successes = 0

                    for idx, (name, inten) in enumerate(HOLDOUT_SET):
                        sound_names.append(name)
                        sender_beh = sender_hold_cache[idx]["behavior"]
                        sender_ammc_ts = sender_hold_cache[idx]["ammc_ts"]

                        cal_ts = _apply_channel_cal_timeseries(
                            sender_ammc_ts, ammc_names, active_idx,
                            transform, active_ammc, active_set,
                            sender.circuit.ammc_baseline,
                        )
                        trans_result = receiver.simulate_timesteps_from_ammc(cal_ts)
                        if trans_result["behavior"] == sender_beh:
                            successes += 1

                        natural_states.append(recv_hold_wv[idx])
                        natural_wv_ts.append(recv_hold_wv_ts[idx])
                        transferred_states.append(_wv_avg_from_temporal(trans_result, avg_w))
                        transferred_wv_ts.append(trans_result["wedvlp"])

                    all_metrics.append({
                        "transfer_rate": successes / len(HOLDOUT_SET),
                        "correlation": correlation(transferred_states, natural_states),
                        "mse": mean_mse(transferred_states, natural_states),
                        "rsa": representational_similarity(natural_states, transferred_states),
                        "sound_id": temporal_stimulus_identification(sound_names, natural_wv_ts, transferred_wv_ts),
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
    parser = argparse.ArgumentParser(description="Adult fly auditory JO channel sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_channel_sweep(seeds_per_level=args.seeds)

    print("=== Adult Drosophila: Auditory JO Channel Sweep ===")
    print(f"{'n_ch':>6}  {'jitter':>6}  {'rate':>6}  {'corr':>6}  {'snd_id':>6}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['n_channels']:6d}  {r['jitter_scale']:6.2f}  "
            f"{r['transfer_rate']:6.3f}  {r['correlation']:6.3f}  "
            f"{r['sound_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="adult_auditory_channel_sweep",
            description="Adult Drosophila auditory transfer quality vs JO channel count",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
