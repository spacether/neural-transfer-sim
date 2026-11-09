"""Cross-species bottleneck transfer comparison.

Loads existing 2D bottleneck sweep results across modalities and organisms,
then plots stimulus identification vs fraction of bottleneck connected.
This isolates the effect of compression ratio (input → bottleneck) and
expansion ratio (bottleneck → downstream) on transfer quality.

Organisms and their bottleneck sweeps:
  Olfactory:
    - D. larva:       21 PN bottleneck  (21 ORN → 21 PN → 73 KC)
    - D. adult:       49 PN bottleneck  (1117 ORN → 49 PN → 2580 KC)
    - Zebrafish:      80 mitral bottleneck (300 ORN → 80 mitral → 500 Dp)
  Auditory:
    - D. adult:       150 AMMC bottleneck (1103 JO → 150 AMMC → 324 WED/VLP)
    - M. musculus:    60 CN bottleneck   (120 IHC → 60 CN → 150 IC)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_transfer.analysis.results_writer import write_results

EXPERIMENTS_ROOT = Path(__file__).parents[1]

ORGANISMS = [
    {
        "name": "D. larva (olf)",
        "modality": "olfactory",
        "path": EXPERIMENTS_ROOT / "olfactory" / "drosophila_larva" / "pn_channel_sweep" / "results" / "larva_pn_channel_sweep.json",
        "bottleneck_size": 21,
        "bottleneck_label": "PN",
        "input_size": 21,
        "expansion_size": 73,
        "id_key": "odor_id",
    },
    {
        "name": "D. adult (olf)",
        "modality": "olfactory",
        "path": EXPERIMENTS_ROOT / "olfactory" / "drosophila_adult" / "pn_channel_sweep" / "results" / "adult_pn_channel_sweep.json",
        "bottleneck_size": 49,
        "bottleneck_label": "PN",
        "input_size": 1117,
        "expansion_size": 2580,
        "id_key": "odor_id",
    },
    {
        "name": "Zebrafish (olf)",
        "modality": "olfactory",
        "path": EXPERIMENTS_ROOT / "olfactory" / "zebrafish" / "pn_channel_sweep" / "results" / "zebrafish_mitral_channel_sweep.json",
        "bottleneck_size": 80,
        "bottleneck_label": "mitral",
        "input_size": 300,
        "expansion_size": 500,
        "id_key": "odor_id",
    },
    {
        "name": "D. adult (aud)",
        "modality": "auditory",
        "path": EXPERIMENTS_ROOT / "auditory" / "drosophila_adult" / "ammc_channel_sweep" / "results" / "adult_auditory_ammc_channel_sweep.json",
        "bottleneck_size": 150,
        "bottleneck_label": "AMMC",
        "input_size": 1103,
        "expansion_size": 324,
        "id_key": "sound_id",
    },
    {
        "name": "M. musculus (aud)",
        "modality": "auditory",
        "path": EXPERIMENTS_ROOT / "auditory" / "mus_musculus" / "cn_channel_sweep" / "results" / "mouse_auditory_cn_channel_sweep.json",
        "bottleneck_size": 60,
        "bottleneck_label": "CN",
        "input_size": 120,
        "expansion_size": 150,
        "id_key": "sound_id",
    },
]

RESULTS_DIR = Path(__file__).parent / "results"


def _load_diagonal(path: Path, bottleneck_size: int, id_key: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)["results"]
    rows = []
    for r in data:
        if r["n_read"] == r["n_write"]:
            sid = r[id_key]
            rows.append({
                "n_channels": r["n_read"],
                "fraction": r["n_read"] / bottleneck_size,
                "input_id": sid,
                "input_id_ci_lo": r.get(f"{id_key}_ci_lo", sid),
                "input_id_ci_hi": r.get(f"{id_key}_ci_hi", sid),
                "transfer_rate": r["transfer_rate"],
            })
    return sorted(rows, key=lambda x: x["fraction"])


def run_comparison() -> List[Dict]:
    results = []
    for org in ORGANISMS:
        if not org["path"].exists():
            print(f"  Skipping {org['name']}: {org['path']} not found")
            continue
        diag = _load_diagonal(org["path"], org["bottleneck_size"], org["id_key"])
        compression = org["input_size"] / org["bottleneck_size"]
        expansion = org["expansion_size"] / org["bottleneck_size"]
        results.append({
            "organism": org["name"],
            "modality": org["modality"],
            "bottleneck": org["bottleneck_label"],
            "bottleneck_size": org["bottleneck_size"],
            "input_size": org["input_size"],
            "expansion_size": org["expansion_size"],
            "compression_ratio": round(compression, 1),
            "expansion_ratio": round(expansion, 1),
            "diagonal": diag,
        })
    return results


def plot_bottleneck_comparison(results: List[Dict], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    olf_colors = {"D. larva (olf)": "#2ca02c", "D. adult (olf)": "#ff7f0e", "Zebrafish (olf)": "#9467bd"}
    aud_colors = {"D. adult (aud)": "#d62728", "M. musculus (aud)": "#1f77b4"}
    all_colors = {**olf_colors, **aud_colors}
    olf_markers = {"D. larva (olf)": "^", "D. adult (olf)": "D", "Zebrafish (olf)": "v"}
    aud_markers = {"D. adult (aud)": "s", "M. musculus (aud)": "o"}
    all_markers = {**olf_markers, **aud_markers}

    for ax_idx, (ax, title) in enumerate(zip(axes, [
        "Input identification vs bottleneck fraction",
        "Input identification vs absolute channels",
    ])):
        for r in results:
            diag = r["diagonal"]
            sids = [d["input_id"] for d in diag]
            ci_lo = [d["input_id_ci_lo"] for d in diag]
            ci_hi = [d["input_id_ci_hi"] for d in diag]
            yerr_lo = [s - lo for s, lo in zip(sids, ci_lo)]
            yerr_hi = [hi - s for s, hi in zip(sids, ci_hi)]

            color = all_colors[r["organism"]]
            marker = all_markers[r["organism"]]
            linestyle = "-" if r["modality"] == "olfactory" else "--"

            if ax_idx == 0:
                xs = [d["fraction"] for d in diag]
                label = (
                    f"{r['organism']} {r['bottleneck']}={r['bottleneck_size']}"
                    f"  {r['compression_ratio']}:1→{r['expansion_ratio']}x"
                )
            else:
                xs = [d["n_channels"] for d in diag]
                label = f"{r['organism']} ({r['bottleneck']})"

            ax.errorbar(
                xs, sids, yerr=[yerr_lo, yerr_hi],
                fmt=f"{marker}", color=color, label=label,
                linestyle=linestyle, capsize=3, markersize=6,
            )

        ax.set_ylabel("Input identification")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel("Fraction of bottleneck connected (read = write)")
    axes[0].set_xlim(-0.02, 1.05)
    axes[1].set_xlabel("Absolute channel count (read = write)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-species bottleneck transfer comparison"
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_comparison()

    print("=== Cross-Species Bottleneck Scaling ===\n")
    for r in results:
        print(
            f"  {r['organism']:20s}  {r['modality']:9s}  "
            f"bottleneck={r['bottleneck']}({r['bottleneck_size']})"
            f"  {r['compression_ratio']}:1→{r['expansion_ratio']}x"
        )
        print(f"  {'frac':>6}  {'ch':>4}  {'inp_id':>6}  {'rate':>6}")
        for d in r["diagonal"]:
            print(
                f"  {d['fraction']:6.2f}  {d['n_channels']:4d}  "
                f"{d['input_id']:6.3f}  {d['transfer_rate']:6.3f}"
            )
        print()

    if not args.no_save:
        flat_results = []
        for r in results:
            for d in r["diagonal"]:
                flat_results.append({
                    "organism": r["organism"],
                    "modality": r["modality"],
                    "bottleneck": r["bottleneck"],
                    "bottleneck_size": r["bottleneck_size"],
                    "compression_ratio": r["compression_ratio"],
                    "expansion_ratio": r["expansion_ratio"],
                    **d,
                })

        json_path = write_results(
            experiment="bottleneck_scaling_comparison",
            description="Input identification vs bottleneck fraction across species and modalities",
            parameters={
                "organisms": [r["organism"] for r in results],
                "source": "loaded from existing 2D bottleneck sweep results",
            },
            results=flat_results,
            output_dir=RESULTS_DIR,
        )
        print(f"Results: {json_path}")

        plot_path = RESULTS_DIR / "bottleneck_scaling_comparison.png"
        plot_bottleneck_comparison(results, plot_path)
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
