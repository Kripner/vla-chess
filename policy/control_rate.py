"""Control-rate sidecar: the checkpoint's control.json.

An action chunk has no intrinsic clock: training defines one action step as the demo pose
1/control_hz later, so the chunk must be executed at that rate or the policy plays back
time-scaled. The trainer writes it (save); the evaluator reads it (load).
"""
from __future__ import annotations

import json
from pathlib import Path

CONTROL_FILE = "control.json"


def save(pretrained_dir, control_hz: float, dataset_fps: float) -> None:
    """Write the sidecar next to the checkpoint's config.json; `dataset_fps` is provenance only."""
    (Path(pretrained_dir) / CONTROL_FILE).write_text(
        json.dumps({"control_hz": float(control_hz), "dataset_fps": float(dataset_fps)}) + "\n")


def load(pretrained_dir) -> float | None:
    """The checkpoint's control rate in Hz, or None (no sidecar, e.g. a hub repo id)."""
    p = Path(str(pretrained_dir)) / CONTROL_FILE
    if not p.is_file():
        return None
    return float(json.loads(p.read_text())["control_hz"])
