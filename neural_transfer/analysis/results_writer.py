"""Write experiment results as self-describing JSON files."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_results(
    experiment: str,
    description: str,
    parameters: Dict[str, Any],
    results: List[Dict],
    output_dir: Path | str,
    filename: str | None = None,
) -> Path:
    """Write a results JSON file to the given directory.

    Returns the path of the written file.
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if filename is None:
        filename = f"{experiment}.json"

    payload = {
        "experiment": experiment,
        "description": description,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "parameters": parameters,
        "metrics": _extract_metric_keys(results),
        "results": results,
    }

    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def _extract_metric_keys(results: List[Dict]) -> List[str]:
    """Pull the metric column names from the first result row."""
    if not results:
        return []
    skip = {"jitter_scale", "seeds", "n_trials", "n_success", "n_holdout", "n_calibration"}
    return [k for k in results[0] if k not in skip]
