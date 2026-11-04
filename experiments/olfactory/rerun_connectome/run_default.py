"""Default runner for full connectome rerun with bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.olfactory.rerun_connectome.experiment import (
    run_broadcast,
    run_channel_sweep,
    run_hardware_boundary,
    METRIC_KEYS,
)
from analysis.results_writer import write_results

RESULTS_DIR = Path(__file__).parent / "results"


def fmt_ci(row, key):
    v = row[key]
    lo = row.get(f"{key}_ci_lo", v)
    hi = row.get(f"{key}_ci_hi", v)
    if abs(hi - lo) < 0.005:
        return f"{v:.2f}"
    return f"{v:.2f} [{lo:.2f},{hi:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun all experiments on connectome models")
    parser.add_argument("organisms", nargs="*", default=["larva", "adult"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = {}

    for org in args.organisms:
        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — BROADCAST SWEEP")
        print(f"{'='*70}")
        bcast = run_broadcast(org, n_seeds=10)
        all_results[f"{org}_broadcast"] = bcast
        for r in bcast:
            print(f"  jitter={r['jitter_scale']:.2f}  transfer={fmt_ci(r,'transfer_rate')}  "
                  f"corr={fmt_ci(r,'correlation')}  rsa={fmt_ci(r,'rsa')}  "
                  f"odor_id={fmt_ci(r,'odor_id')}")

        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — CHANNEL SWEEP")
        print(f"{'='*70}")
        chsw = run_channel_sweep(org, n_seeds=5, channel_seeds=3)
        all_results[f"{org}_channel_sweep"] = chsw
        cur_jitter = None
        for r in chsw:
            if r["jitter_scale"] != cur_jitter:
                cur_jitter = r["jitter_scale"]
                print(f"\n  --- jitter = {cur_jitter} ---")
            print(f"    {r['n_channels']:3d} ch  transfer={fmt_ci(r,'transfer_rate')}  "
                  f"odor_id={fmt_ci(r,'odor_id')}")

        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — HARDWARE BOUNDARY")
        print(f"{'='*70}")
        hw = run_hardware_boundary(org, n_seeds=10)
        all_results[f"{org}_hardware"] = hw
        for axis_name, rows in hw.items():
            print(f"\n  --- {axis_name} ---")
            param_key = [k for k in rows[0] if k not in METRIC_KEYS and not k.endswith("_ci_lo") and not k.endswith("_ci_hi") and k != "n_seeds"][0]
            for r in rows:
                print(f"    {r[param_key]:>8}  transfer={fmt_ci(r,'transfer_rate')}  "
                      f"corr={fmt_ci(r,'correlation')}")

    if not args.no_save:
        def _clean(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return str(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, list):
                serializable[k] = [{kk: _clean(vv) for kk, vv in row.items()} for row in v]
            elif isinstance(v, dict):
                serializable[k] = {kk: [{kkk: _clean(vvv) for kkk, vvv in row.items()} for row in vv]
                                    for kk, vv in v.items()}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "connectome_results.json"
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
