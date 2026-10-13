"""Checkpoint helpers shared by RheolNet training workflows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable short fingerprint for a JSON-compatible configuration."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a module state dictionary to CPU for portable checkpoints."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a torch checkpoint atomically so an interruption cannot corrupt it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a trusted project checkpoint with an explicit existence check."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
