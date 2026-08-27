"""Physical-coordinate scaling for PINNs trained on SI planar CFD cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DimensionalScales:
    """Scales that normalise network variables without changing PDE units."""

    x_scale_m: float
    y_scale_m: float
    velocity_scale_m_s: float
    pressure_scale_pa: float
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    pressure_offset_pa: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_scale_m", "y_scale_m", "velocity_scale_m_s", "pressure_scale_pa"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    @classmethod
    def from_case_metadata(cls, metadata: dict[str, Any]) -> "DimensionalScales":
        geometry = metadata["geometry"]
        rho = float(metadata["density_kg_m3"])
        mean_velocity = float(metadata["mean_inlet_velocity_m_s"])
        return cls(
            x_scale_m=float(geometry["total_length_m"]),
            y_scale_m=0.5 * float(geometry["healthy_channel_height_m"]),
            velocity_scale_m_s=mean_velocity,
            pressure_scale_pa=rho * mean_velocity**2,
        )


class DimensionalPINN(nn.Module):
    """Wrap a dimensionless network so physics remains in physical SI units.

    ``compute_ns_residuals_nonnewtonian`` differentiates this wrapper with
    respect to physical coordinates. Autograd therefore applies the required
    chain rule automatically.
    """

    def __init__(self, network: nn.Module, scales: DimensionalScales) -> None:
        super().__init__()
        self.network = network
        self.register_buffer(
            "coordinate_scale",
            torch.tensor([scales.x_scale_m, scales.y_scale_m], dtype=torch.float32),
        )
        self.register_buffer(
            "coordinate_offset",
            torch.tensor([scales.x_offset_m, scales.y_offset_m], dtype=torch.float32),
        )
        self.register_buffer(
            "output_scale",
            torch.tensor(
                [scales.velocity_scale_m_s, scales.velocity_scale_m_s, scales.pressure_scale_pa],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "output_offset",
            torch.tensor([0.0, 0.0, scales.pressure_offset_pa], dtype=torch.float32),
        )

    def normalise_coordinates(self, xy_m: torch.Tensor) -> torch.Tensor:
        return (xy_m - self.coordinate_offset) / self.coordinate_scale

    def forward(self, xy_m: torch.Tensor) -> torch.Tensor:
        unit_prediction = self.network(self.normalise_coordinates(xy_m))
        return unit_prediction * self.output_scale + self.output_offset
