"""Torch samplers for the steady 2D planar-stenosis PINN domain."""

from __future__ import annotations

from typing import Optional

import torch

from .planar_stenosis import PlanarStenosisGeometry


def _half_height_torch(
    geometry: PlanarStenosisGeometry,
    x_m: torch.Tensor,
) -> torch.Tensor:
    h0 = geometry.healthy_half_height_m
    ht = geometry.throat_half_height_m
    h = torch.full_like(x_m, h0)

    converging = (x_m >= geometry.converging_start_m) & (x_m < geometry.throat_start_m)
    throat = (x_m >= geometry.throat_start_m) & (x_m <= geometry.throat_end_m)
    diverging = (x_m > geometry.throat_end_m) & (x_m <= geometry.diverging_end_m)

    h = torch.where(
        converging,
        h0 + (ht - h0) * (x_m - geometry.converging_start_m)
        / (geometry.throat_start_m - geometry.converging_start_m),
        h,
    )
    h = torch.where(throat, torch.full_like(h, ht), h)
    h = torch.where(
        diverging,
        ht + (h0 - ht) * (x_m - geometry.throat_end_m)
        / (geometry.diverging_end_m - geometry.throat_end_m),
        h,
    )
    return h


def sample_interior_points(
    geometry: PlanarStenosisGeometry,
    n_points: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Uniformly sample physical ``(x, y)`` points inside the fluid domain."""
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    x = torch.rand(n_points, 1, device=device, dtype=dtype, generator=generator)
    x = x * geometry.total_length_m
    h = _half_height_torch(geometry, x)
    y = (2.0 * torch.rand(n_points, 1, device=device, dtype=dtype, generator=generator) - 1.0) * h
    return torch.cat((x, y), dim=1)


def sample_wall_points(
    geometry: PlanarStenosisGeometry,
    n_points_per_wall: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points on the bottom and top no-slip walls respectively."""
    if n_points_per_wall <= 0:
        raise ValueError("n_points_per_wall must be positive.")
    x = torch.rand(n_points_per_wall, 1, device=device, dtype=dtype, generator=generator)
    x = x * geometry.total_length_m
    h = _half_height_torch(geometry, x)
    return torch.cat((x, -h), dim=1), torch.cat((x, h), dim=1)


def sample_inlet_points(
    geometry: PlanarStenosisGeometry,
    n_points: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return evenly distributed inlet points, including both wall endpoints."""
    if n_points < 2:
        raise ValueError("At least two inlet points are needed to include both walls.")
    y = torch.linspace(
        -geometry.healthy_half_height_m,
        geometry.healthy_half_height_m,
        n_points,
        device=device,
        dtype=dtype,
    ).reshape(-1, 1)
    return torch.cat((torch.zeros_like(y), y), dim=1)


def sample_outlet_points(
    geometry: PlanarStenosisGeometry,
    n_points: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return evenly distributed outlet points, including both wall endpoints."""
    inlet = sample_inlet_points(geometry, n_points, device=device, dtype=dtype)
    inlet[:, 0] = geometry.total_length_m
    return inlet
