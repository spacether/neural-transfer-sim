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
    ax.plot(jitter, [r["decoding"] for r in results], "o-", color="#d97706")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Brain difference (× biological CV)")
    ax.set_title("Odor-type decoding (attractive vs repellent)")
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
        ("decoding", "Odor-type decoding (attractive vs repellent)", True),
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


def _annotate_techs(ax, techs, invert_x: bool = False) -> None:
    """Draw labeled vertical lines for real technology operating points."""
    from hardware.technologies import TechPoint

    colors = ["#6366f1", "#0891b2", "#be185d", "#ca8a04", "#16a34a"]
    offsets = [0.97, 0.90, 0.83, 0.76, 0.69]
    for i, tech in enumerate(techs):
        color = colors[i % len(colors)]
        y_frac = offsets[i % len(offsets)]
        ax.axvline(tech.value, color=color, linewidth=1, linestyle=":",
                   alpha=0.65, zorder=1)
        ax.text(
            tech.value, y_frac, f" {tech.short}",
            transform=ax.get_xaxis_transform(),
            fontsize=6.5, color=color, fontweight="bold",
            ha="left" if not invert_x else "right",
            va="center", clip_on=True,
        )


def plot_hardware_boundary(all_results: Dict[str, list], output_path: Path) -> Path:
    """Plot transfer quality vs hardware degradation for each axis + combined.

    The first three panels are annotated with approximate operating points
    of real recording and stimulation technologies.
    """
    from hardware.technologies import (
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
        ("decoding", "Odor-type decoding", "#d97706", "s--"),
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
        ax.set_ylim(-0.05, 1.1)
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
