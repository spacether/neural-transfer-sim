"""Shared plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_receptor_to_receptor_sweep(results: List[Dict], output_path: Path) -> Path:
    """Plot transfer metrics vs brain difference for the receptor-to-receptor baseline."""
    jitter = [r["jitter_scale"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Receptor-to-Receptor: Transfer Quality vs Brain Difference")

    ax = axes[0, 0]
    ax.plot(jitter, [r["correlation"] for r in results], "o-", color="#2563eb")
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Brain difference (× biological CV)")
    ax.set_title("Correlation")
    ax.set_ylim(None, 1.05)

    ax = axes[0, 1]
    ax.plot(jitter, [r["mse"] for r in results], "o-", color="#dc2626")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Brain difference (× biological CV)")
    ax.set_title("Reconstruction error")

    ax = axes[1, 0]
    ax.plot(jitter, [r["rsa"] for r in results], "o-", color="#059669")
    ax.set_ylabel("RSA (Spearman r)")
    ax.set_xlabel("Brain difference (× biological CV)")
    ax.set_title("Representational similarity")
    ax.set_ylim(None, 1.05)

    ax = axes[1, 1]
    ax.plot(jitter, [r["odor_id"] for r in results], "o-", color="#d97706")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Brain difference (× biological CV)")
    ax.set_title("Odor identification (N-way)")
    ax.set_ylim(None, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_comparison(results: List[Dict], output_path: Path) -> Path:
    """Plot receptor-to-receptor vs calibrated transfer across brain-difference levels."""
    jitter = [r["jitter_scale"] for r in results]

    metrics = [
        ("correlation", "Correlation (Pearson r)", True),
        ("mse", "MSE (reconstruction error)", False),
        ("rsa", "RSA (representational similarity)", True),
        ("odor_id", "Odor identification (N-way)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Calibrated vs Receptor-to-Receptor")

    for ax, (key, title, higher_better) in zip(axes.flat, metrics):
        d_vals = [r[f"direct_{key}"] for r in results]
        c_vals = [r[f"calibrated_{key}"] for r in results]

        ax.plot(jitter, d_vals, "o-", color="#2563eb", label="Receptor-to-receptor")
        ax.plot(jitter, c_vals, "s--", color="#dc2626", label="Calibrated")
        ax.set_xlabel("Brain difference (× biological CV)")
        ax.set_ylabel(key.upper() if key != "correlation" else "Pearson r")
        ax.set_title(title)
        ax.legend(fontsize=9)
        if higher_better:
            ax.set_ylim(None, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


TECH_COLORS = {
    "Patch":         "#6366f1",
    "VoltImg":       "#0891b2",
    "GCaMP-hi":      "#16a34a",
    "MEA":           "#ca8a04",
    "GCaMP-lo":      "#16a34a",
    "Light-sheet":   "#0891b2",
    "2P-targeted":   "#be185d",
    "4-ch probe":    "#ca8a04",
    "1 electrode":   "#6366f1",
    "2P-holo":       "#be185d",
    "Electrode":     "#6366f1",
    "1P-opto":       "#16a34a",
    "TrpA1":         "#ca8a04",
    "FPGA":          "#6366f1",
    "Intan+CereStim":"#0891b2",
    "RT-Linux":      "#be185d",
    "RPi":           "#ca8a04",
    "Python loop":   "#16a34a",
}

_FALLBACK_COLORS = ["#6366f1", "#0891b2", "#be185d", "#ca8a04", "#16a34a"]


def _annotate_techs(ax, techs, invert_x: bool = False) -> None:
    """Draw labeled vertical lines for real technology operating points.

    Places labels above the data region with staggered y-positions and
    alternating left/right alignment for techs whose x-values are close.
    """
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none",
                alpha=0.85)
    sorted_techs = sorted(techs, key=lambda t: t.value, reverse=invert_x)
    n = len(sorted_techs)
    y_top, y_bot = 0.98, 0.98 - 0.035 * min(n, 5)
    offsets = np.linspace(y_top, y_bot, n)

    vals = [t.value for t in sorted_techs]
    val_range = max(vals) - min(vals) if len(vals) > 1 else 1.0

    for i, tech in enumerate(sorted_techs):
        color = TECH_COLORS.get(tech.short, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        y_frac = offsets[i]
        ax.axvline(tech.value, color=color, linewidth=1, linestyle=":",
                   alpha=0.55, zorder=1)
        frac = (tech.value - min(vals)) / val_range if val_range > 0 else 0.5
        if invert_x:
            ha = "left" if frac > 0.85 else "right"
        else:
            ha = "right" if frac > 0.85 else "left"
        pad_l = " " if ha == "left" else ""
        pad_r = " " if ha == "right" else ""
        label = f"{pad_l}{tech.short}{pad_r}"
        ax.text(
            tech.value, y_frac, label,
            transform=ax.get_xaxis_transform(),
            fontsize=6.5, color=color, fontweight="bold",
            ha=ha, va="center", clip_on=True,
            bbox=bbox, zorder=5,
        )


def plot_hardware_boundary(all_results: Dict[str, list], output_path: Path) -> Path:
    """Plot transfer quality vs hardware degradation for each axis + combined.

    The first three panels are annotated with approximate operating points
    of real recording and stimulation technologies.
    """
    from neural_transfer.hardware.technologies import (
        READ_NOISE_TECHS,
        WRITE_PRECISION_TECHS,
        CHANNEL_DROPOUT_TECHS,
        TRANSMIT_DELAY_TECHS,
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle("Hardware Boundary: Where Transfer Breaks Down", fontsize=14)

    axis_configs = [
        ("read_noise", "noise_sigma", "Read noise (σ)", False, False),
        ("write_precision", "precision_bits", "Write precision (bits)", True, False),
        ("channel_dropout", "dropout_fraction", "Channel dropout fraction", False, False),
        ("transmit_delay", "delay_ms", "Transmit delay (ms)", False, True),
        ("combined", "step", "Combined degradation (step)", False, False),
    ]

    tech_annotations = {
        "read_noise": READ_NOISE_TECHS,
        "write_precision": WRITE_PRECISION_TECHS,
        "channel_dropout": CHANNEL_DROPOUT_TECHS,
        "transmit_delay": TRANSMIT_DELAY_TECHS,
    }

    metric_styles = [
        ("correlation", "Pearson r", "#2563eb", "o-"),
        ("odor_id", "Odor identification", "#d97706", "s--"),
        ("transfer_rate", "Behavioral match", "#059669", "^:"),
    ]

    flat_axes = axes.flat
    for i, (axis_key, param_key, xlabel, invert_x, log_x) in enumerate(axis_configs):
        ax = flat_axes[i]
        rows = all_results[axis_key]
        x = [r[param_key] for r in rows]

        for metric, label, color, style in metric_styles:
            y = [r[metric] for r in rows]
            ax.plot(x, y, style, color=color, label=label, markersize=5)

        if axis_key in tech_annotations:
            _annotate_techs(ax, tech_annotations[axis_key], invert_x)

        if log_x:
            ax.set_xscale("symlog", linthresh=0.05, linscale=0.3)
            ax.set_xlim(left=-0.02)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda v, _: f"{v:g}" if v >= 0 else ""
            ))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        has_techs = axis_key in tech_annotations
        ax.set_ylim(-0.05, 1.30 if has_techs else 1.1)
        ax.axhline(0.5, color="#999", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="lower left" if not invert_x else "lower right")
        if invert_x:
            ax.invert_xaxis()

    # Hide the unused 6th panel
    flat_axes[len(axis_configs)].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_channel_sweep(results: List[Dict], output_path: Path) -> Path:
    """Plot transfer quality vs number of connected channels at each jitter level."""
    jitter_levels = sorted(set(r["jitter_scale"] for r in results))
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("ORN Channel Sweep: Transfer Quality vs Connected ORNs (direct broadcast)")

    metrics = [
        ("correlation", "Pearson r"),
        ("transfer_rate", "Behavioral match rate"),
        ("odor_id", "Odor identification"),
    ]

    for ax, (key, ylabel) in zip(axes, metrics):
        for j, jitter in enumerate(jitter_levels):
            rows = [r for r in results if r["jitter_scale"] == jitter]
            x = [r["n_channels"] for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, "o-", color=colors[j % len(colors)],
                    label=f"jitter={jitter:.1f}", markersize=5)
        ax.set_xlabel("Number of connected ORNs (of 21)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0.5, color="#999", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_pn_channel_sweep(results: List[Dict], output_path: Path) -> Path:
    """2D heatmaps of transfer metrics vs read/write PN counts."""
    read_counts = sorted(set(r["n_read"] for r in results))
    write_counts = sorted(set(r["n_write"] for r in results))

    metrics = [
        ("correlation", "KC correlation (Pearson r)"),
        ("transfer_rate", "Behavioral match"),
        ("odor_id", "Odor identification"),
    ]

    n_m = len(metrics)
    fig, axes = plt.subplots(1, n_m + 1, figsize=(6 * n_m + 1.2, 5.5),
                             gridspec_kw={"width_ratios": [*([1] * n_m), 0.05],
                                          "wspace": 0.3})
    fig.suptitle("PN Channel Sweep: Transfer Quality vs Read/Write PN Count",
                 fontsize=13)

    for ax, (key, title) in zip(axes[:n_m], metrics):
        matrix = np.full((len(read_counts), len(write_counts)), np.nan)
        for r in results:
            ri = read_counts.index(r["n_read"])
            wi = write_counts.index(r["n_write"])
            matrix[ri, wi] = r[key]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(write_counts)))
        ax.set_xticklabels(write_counts)
        ax.set_yticks(range(len(read_counts)))
        ax.set_yticklabels(read_counts)
        n_total = max(write_counts)
        ax.set_xlabel(f"Write PNs (of {n_total})")
        ax.set_ylabel(f"Read PNs (of {n_total})")
        ax.set_title(title)

        for ri in range(len(read_counts)):
            for wi in range(len(write_counts)):
                val = matrix[ri, wi]
                if not np.isnan(val):
                    ax.text(wi, ri, f"{val:.2f}",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold")

    fig.colorbar(im, cax=axes[n_m], label="Score")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_efficiency(results: Dict[str, List[Dict]], output_path: Path) -> Path:
    """Plot calibration sample efficiency across organisms."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle("Calibration Sample Efficiency: Odor Identification vs Training Samples")

    colors = {"C. elegans": "#7c3aed", "D. larva": "#2563eb",
              "D. adult": "#059669", "Zebrafish": "#d97706"}
    for organism, rows in results.items():
        x = [r["n_samples"] for r in rows]
        y = [r["odor_id"] for r in rows]
        yerr = [r.get("odor_id_std", 0) for r in rows]
        color = colors.get(organism, "#666")
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, label=organism,
                    markersize=5, capsize=3)

    ax.set_xlabel("Number of calibration samples")
    ax.set_ylabel("Odor identification accuracy")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="#999", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scaling_comparison(results: List[Dict], output_path: Path) -> Path:
    """Plot cross-organism scaling: requirements vs organism complexity."""
    organisms = [r["organism"] for r in results]
    x_pos = list(range(len(organisms)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Cross-Organism Scaling: Transfer Requirements", fontsize=14)

    ax = axes[0, 0]
    ax.bar(x_pos, [r["receptor_types"] for r in results], color="#2563eb", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(organisms)
    ax.set_ylabel("Receptor types / channels")
    ax.set_title("Receptor repertoire size")
    ax.set_yscale("log")

    ax = axes[0, 1]
    if all("min_channels_90" in r for r in results):
        ax.bar(x_pos, [r["min_channels_90"] for r in results], color="#059669", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(organisms)
        ax.set_ylabel("Min channels for 90% behavioral match")
        ax.set_title("Channel requirement")

    ax = axes[1, 0]
    if all("calibration_samples" in r for r in results):
        ax.bar(x_pos, [r["calibration_samples"] for r in results], color="#d97706", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(organisms)
        ax.set_ylabel("Calibration samples needed")
        ax.set_title("Calibration cost")

    ax = axes[1, 1]
    if all("broadcast_behavioral_match" in r for r in results):
        ax.bar(x_pos, [r["broadcast_behavioral_match"] for r in results],
               color="#7c3aed", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(organisms)
        ax.set_ylabel("Behavioral match (broadcast)")
        ax.set_ylim(0, 1.1)
        ax.set_title("Broadcast transfer quality")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
