"""AMMC channel sweep: 2D sweep of read and write AMMC counts for adult fly.

Reads from N_read AMMC neurons of the sender, writes to N_write AMMC neurons
of the receiver, with calibration mapping the read subset to the write subset.
Measures WED/VLP correlation -- does the receiver's downstream representation
match what it would produce naturally?

With 150 AMMC neurons (the auditory bottleneck), this tests whether writing
at the bottleneck outperforms writing at JO neurons, and how many bottleneck
channels are needed.

Sender temporal simulations are precomputed once and reused across all
grid points to avoid redundant work.

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
    odor_identification,
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


def _precompute_sender(
    sender: AuditoryAdultModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> List[Dict]:
    cache = []
    ammc_names = sender.ammc_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = sender.simulate_timesteps(stimulus)
        ammc_avg = result["ammc"][-avg_w:].mean(axis=0)
        full_ammc = {n: float(ammc_avg[i]) for i, n in enumerate(ammc_names)}
        cache.append({
            "ammc": full_ammc,
            "behavior": result["behavior"],
        })
    return cache


def _precompute_receiver(
    receiver: AuditoryAdultModel,
    stimuli: List[Tuple[str, float]],
    avg_w: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    ammc_list = []
    wv_list = []
    r_ammc_names = receiver.ammc_names
    for name, intensity in stimuli:
        stimulus = make_temporal_stimulus(name, intensity)
        result = receiver.simulate_timesteps(stimulus)
        ammc_avg = result["ammc"][-avg_w:].mean(axis=0)
        ammc_list.append({n: float(ammc_avg[i]) for i, n in enumerate(r_ammc_names)})
        wv_avg = result["wedvlp"][-avg_w:].mean(axis=0)
        wv_list.append({f"WV_{i:03d}": float(wv_avg[i]) for i in range(len(wv_avg))})
    return ammc_list, wv_list


def _extract_wv(state: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in state.items() if k.startswith("WV_")}


def run_ammc_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter: float = 0.5,
    seeds: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    if channel_counts is None:
        channel_counts = [5, 20, 50, 75, 100, 150]

    sender = AuditoryAdultModel()
    gen = AuditoryAdultVariantGenerator(jitter_scale=jitter)
    n_ammc = sender.n_ammc
    ammc_names = sender.ammc_names
    avg_w = temporal_avg_window()

    sender_cal_cache = _precompute_sender(sender, CAL_SET, avg_w)
    sender_hold_cache = _precompute_sender(sender, HOLDOUT_SET, avg_w)

    receiver_cache: Dict[int, Tuple] = {}
    for s in range(seeds):
        receiver = gen.generate_variant(sender, seed=s)
        if not gen.sanity_check(receiver):
            continue
        recv_cal_ammc, _ = _precompute_receiver(receiver, CAL_SET, avg_w)
        _, recv_hold_wv = _precompute_receiver(receiver, HOLDOUT_SET, avg_w)
        receiver_cache[s] = (receiver, recv_cal_ammc, recv_hold_wv)

    results: List[Dict] = []

    for n_read in channel_counts:
        for n_write in channel_counts:
            seed_metrics: List[Dict[str, float]] = []

            for cs in range(channel_seeds):
                rng = np.random.default_rng(cs + 1000)
                read_idx = sorted(rng.choice(n_ammc, size=min(n_read, n_ammc), replace=False).tolist())
                write_idx = sorted(rng.choice(n_ammc, size=min(n_write, n_ammc), replace=False).tolist())
                read_ammc = [ammc_names[i] for i in read_idx]
                write_ammc = [ammc_names[i] for i in write_idx]
                write_set = set(write_ammc)

                sender_cal = [{n: sc["ammc"][n] for n in read_ammc} for sc in sender_cal_cache]

                for s, (receiver, recv_cal_ammc, recv_hold_wv) in receiver_cache.items():
                    receiver_cal = [{n: ra[n] for n in write_ammc if n in ra}
                                    for ra in recv_cal_ammc]

                    transform = CalibrationTransform()
                    transform.neurons = tuple(sorted(set(read_ammc) | set(write_ammc)))
                    transform.fit(sender_cal, receiver_cal)

                    sound_names: List[str] = []
                    natural_wv: List[Dict[str, float]] = []
                    transferred_wv: List[Dict[str, float]] = []
                    successes = 0

                    for idx, (name, inten) in enumerate(HOLDOUT_SET):
                        sound_names.append(name)

                        sender_sig = {n: sender_hold_cache[idx]["ammc"][n] for n in read_ammc}
                        sender_beh = sender_hold_cache[idx]["behavior"]

                        cal_sig = transform.transform(sender_sig)
                        write_sig = {k: v for k, v in cal_sig.items() if k in write_set}
                        full_ammc_dict = {n: sender.circuit.ammc_baseline for n in ammc_names}
                        full_ammc_dict.update(write_sig)
                        result = receiver.simulate_from_ammc(full_ammc_dict)
                        if result["behavior"] == sender_beh:
                            successes += 1

                        natural_wv.append(recv_hold_wv[idx])
                        transferred_wv.append(_extract_wv(result["neurons"]))

                    n = len(HOLDOUT_SET)
                    seed_metrics.append({
                        "transfer_rate": successes / max(1, n),
                        "correlation": correlation(transferred_wv, natural_wv),
                        "sound_id": odor_identification(sound_names, natural_wv, transferred_wv),
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
    parser = argparse.ArgumentParser(description="Adult fly auditory AMMC channel sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_ammc_channel_sweep(seeds=args.seeds)

    print("=== Adult Drosophila: Auditory AMMC Channel Sweep ===")
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
            experiment="adult_auditory_ammc_channel_sweep",
            description="Adult Drosophila auditory 2D AMMC read/write channel sweep",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_pn_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
