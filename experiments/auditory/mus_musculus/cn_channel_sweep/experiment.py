"""CN channel sweep: 2D sweep of read and write CN counts for mouse.

Reads from N_read CN neurons of the sender, writes to N_write CN neurons
of the receiver, with calibration mapping the read subset to the write subset.
Measures IC correlation -- does the receiver's downstream representation
match what it would produce naturally?

With 60 CN neurons (the auditory bottleneck), this tests whether writing
at the bottleneck outperforms writing at hair cells, and how many
bottleneck channels are needed.

Sender temporal simulations are precomputed once and reused across all
grid points to avoid redundant work.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from neural_transfer.models.mus_musculus import MouseAuditoryModel
from neural_transfer.models.mus_musculus import MouseAuditoryVariantGenerator
from neural_transfer.models.mus_musculus import (
    SOUND_PROFILES, SOUND_VALENCE,
    make_temporal_stimulus, temporal_avg_window,
)
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import (
    correlation,
    temporal_stimulus_identification,
    bootstrap_ci,
)
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_pn_channel_sweep


ALL_SOUND_NAMES = list(SOUND_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_SOUND_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_SOUND_NAMES for i in HOLDOUT_INTENSITIES]

METRIC_KEYS = ("transfer_rate", "correlation", "sound_id")


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def _ic_avg_from_temporal(result: Dict, avg_w: int) -> Dict[str, float]:
    ic_avg = result["ic"][-avg_w:].mean(axis=0)
    return {f"IC_{i:03d}": float(ic_avg[i]) for i in range(len(ic_avg))}


def _precompute_sender(
    sender: MouseAuditoryModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> List[Dict]:
    cache = []
    cn_names = sender.cn_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = sender.simulate_timesteps(stimulus)
        cn_avg = result["cn"][-avg_w:].mean(axis=0)
        full_cn = {n: float(cn_avg[i]) for i, n in enumerate(cn_names)}
        cache.append({
            "cn": full_cn,
            "cn_ts": result["cn"],
            "behavior": result["behavior"],
        })
    return cache


def _precompute_receiver(
    receiver: MouseAuditoryModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[np.ndarray]]:
    cn_list = []
    ic_list = []
    ic_ts_list: List[np.ndarray] = []
    r_cn_names = receiver.cn_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = receiver.simulate_timesteps(stimulus)
        cn_avg = result["cn"][-avg_w:].mean(axis=0)
        cn_list.append({n: float(cn_avg[i]) for i, n in enumerate(r_cn_names)})
        ic_list.append(_ic_avg_from_temporal(result, avg_w))
        ic_ts_list.append(result["ic"])
    return cn_list, ic_list, ic_ts_list


def _apply_2d_channel_cal_timeseries(
    cn_ts: np.ndarray, cn_names: List[str],
    read_idx: List[int], write_idx: List[int],
    transform: CalibrationTransform,
    write_cn: List[str], write_set: set,
    baseline: float,
) -> np.ndarray:
    """Apply read-channel selection, calibration, and write-channel injection (vectorized)."""
    n_steps, n_cn = cn_ts.shape
    cal_neurons = list(transform.neurons)
    read_cn = [cn_names[i] for i in read_idx]
    cal_in = np.zeros((n_steps, len(cal_neurons)))
    name_to_cn_idx = {n: i for i, n in enumerate(cn_names)}
    for ci, cn in enumerate(cal_neurons):
        if cn in set(read_cn):
            cal_in[:, ci] = cn_ts[:, name_to_cn_idx[cn]]
    cal_out = transform.transform_array(cal_in)
    out = np.full((n_steps, n_cn), baseline)
    for ci, cn in enumerate(cal_neurons):
        if cn in write_set:
            out[:, name_to_cn_idx[cn]] = cal_out[:, ci]
    return out


def run_cn_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter: float = 1.0,
    seeds: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    if channel_counts is None:
        channel_counts = [5, 15, 30, 45, 60]

    sender = MouseAuditoryModel()
    gen = MouseAuditoryVariantGenerator(jitter_scale=jitter)
    n_cn = sender.n_cn
    cn_names = sender.cn_names
    avg_w = temporal_avg_window()

    sender_cal_cache = _precompute_sender(sender, CAL_SET, avg_w)
    sender_hold_cache = _precompute_sender(sender, HOLDOUT_SET, avg_w)

    receiver_cache: Dict[int, Tuple] = {}
    for s in range(seeds):
        receiver = gen.generate_variant(sender, seed=s)
        if not gen.sanity_check(receiver):
            continue
        recv_cal_cn, _, _ = _precompute_receiver(receiver, CAL_SET, avg_w)
        _, recv_hold_ic, recv_hold_ic_ts = _precompute_receiver(receiver, HOLDOUT_SET, avg_w)
        receiver_cache[s] = (receiver, recv_cal_cn, recv_hold_ic, recv_hold_ic_ts)

    results: List[Dict] = []

    for n_read in channel_counts:
        for n_write in channel_counts:
            seed_metrics: List[Dict[str, float]] = []

            for cs in range(channel_seeds):
                rng = np.random.default_rng(cs + 1000)
                read_idx = sorted(rng.choice(n_cn, size=min(n_read, n_cn), replace=False).tolist())
                write_idx = sorted(rng.choice(n_cn, size=min(n_write, n_cn), replace=False).tolist())
                read_cn = [cn_names[i] for i in read_idx]
                write_cn = [cn_names[i] for i in write_idx]
                write_set = set(write_cn)

                sender_cal = [{n: sc["cn"][n] for n in read_cn} for sc in sender_cal_cache]

                for s, (receiver, recv_cal_cn, recv_hold_ic, recv_hold_ic_ts) in receiver_cache.items():
                    receiver_cal = [{n: ra[n] for n in write_cn if n in ra}
                                    for ra in recv_cal_cn]

                    transform = CalibrationTransform()
                    transform.neurons = tuple(sorted(set(read_cn) | set(write_cn)))
                    transform.fit(sender_cal, receiver_cal)

                    sound_names: List[str] = []
                    natural_ic: List[Dict[str, float]] = []
                    transferred_ic: List[Dict[str, float]] = []
                    natural_ic_ts_list: List[np.ndarray] = []
                    transferred_ic_ts_list: List[np.ndarray] = []
                    successes = 0

                    for idx, (name, inten) in enumerate(HOLDOUT_SET):
                        sound_names.append(name)
                        sender_beh = sender_hold_cache[idx]["behavior"]
                        sender_cn_ts = sender_hold_cache[idx]["cn_ts"]

                        cal_ts = _apply_2d_channel_cal_timeseries(
                            sender_cn_ts, cn_names,
                            read_idx, write_idx,
                            transform, write_cn, write_set,
                            sender.circuit.cn_baseline,
                        )
                        trans_result = receiver.simulate_timesteps_from_cn(cal_ts)
                        if trans_result["behavior"] == sender_beh:
                            successes += 1

                        natural_ic.append(recv_hold_ic[idx])
                        natural_ic_ts_list.append(recv_hold_ic_ts[idx])
                        transferred_ic.append(_ic_avg_from_temporal(trans_result, avg_w))
                        transferred_ic_ts_list.append(trans_result["ic"])

                    n = len(HOLDOUT_SET)
                    seed_metrics.append({
                        "transfer_rate": successes / max(1, n),
                        "correlation": correlation(transferred_ic, natural_ic),
                        "sound_id": temporal_stimulus_identification(sound_names, natural_ic_ts_list, transferred_ic_ts_list),
                    })

            if not seed_metrics:
                continue

            row: Dict = {
                "n_read": n_read,
                "n_write": n_write,
                "jitter_scale": jitter,
                "valid_trials": len(seed_metrics),
            }
            row.update(_agg_with_ci(seed_metrics))
            results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mouse auditory CN channel sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_cn_channel_sweep(seeds=args.seeds)

    print("=== Mouse: Auditory CN Channel Sweep ===")
    print(f"{'n_read':>6}  {'n_write':>7}  {'rate':>6}  {'corr':>6}  {'snd_id':>6}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['n_read']:6d}  {r['n_write']:7d}  "
            f"{r['transfer_rate']:6.3f}  {r['correlation']:6.3f}  "
            f"{r['sound_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="mouse_auditory_cn_channel_sweep",
            description="Mouse auditory 2D CN read/write channel sweep",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_pn_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
