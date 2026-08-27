"""Boundary-condition losses for the steady 2D planar-stenosis PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


def _require_xy(xy: torch.Tensor, name: str) -> None:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")


def no_slip_loss(model: nn.Module, wall_xy_m: torch.Tensor) -> torch.Tensor:
    """Mean-square no-slip loss, enforcing ``u=v=0`` on a wall."""
    _require_xy(wall_xy_m, "wall_xy_m")
    prediction = model(wall_xy_m)
    return torch.mean(prediction[:, :2] ** 2)


def inlet_velocity_loss(
    model: nn.Module,
    inlet_xy_m: torch.Tensor,
    inlet_uv_m_s: torch.Tensor,
) -> torch.Tensor:
    """Match the exact prescribed CFD inlet velocity profile."""
    _require_xy(inlet_xy_m, "inlet_xy_m")
    if inlet_uv_m_s.shape != (inlet_xy_m.shape[0], 2):
        raise ValueError("inlet_uv_m_s must have shape (n_inlet, 2).")
    prediction = model(inlet_xy_m)
    return torch.mean((prediction[:, :2] - inlet_uv_m_s) ** 2)


def outlet_pressure_loss(
    model: nn.Module,
    outlet_xy_m: torch.Tensor,
    pressure_pa: float = 0.0,
) -> torch.Tensor:
    """Enforce Kiera's outlet pressure reference, normally ``p=0 Pa``."""
    _require_xy(outlet_xy_m, "outlet_xy_m")
    prediction = model(outlet_xy_m)
    target = torch.as_tensor(pressure_pa, dtype=prediction.dtype, device=prediction.device)
    return torch.mean((prediction[:, 2] - target) ** 2)


@dataclass(frozen=True)
class PlanarStenosisLossWeights:
    physics: float = 1.0
    wall: float = 10.0
    inlet: float = 10.0
    outlet_pressure: float = 5.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("Loss weights must be non-negative.")


def total_planar_stenosis_loss(
    terms: Mapping[str, torch.Tensor],
    weights: PlanarStenosisLossWeights = PlanarStenosisLossWeights(),
) -> torch.Tensor:
    """Combine named losses while rejecting silently omitted constraints."""
    required = {"physics", "wall", "inlet", "outlet_pressure"}
    missing = required - set(terms)
    unexpected = set(terms) - required
    if missing or unexpected:
        raise ValueError(f"Expected loss terms {sorted(required)}; missing={sorted(missing)}, unexpected={sorted(unexpected)}.")
    return (
        weights.physics * terms["physics"]
        + weights.wall * terms["wall"]
        + weights.inlet * terms["inlet"]
        + weights.outlet_pressure * terms["outlet_pressure"]
    )
