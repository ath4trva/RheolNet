"""Physical-coordinate scaling for PINNs trained on SI planar CFD cases.

Revision notes (v2)
-------------------
The first version used a single ``velocity_scale_m_s`` for both ``u`` and
``v``, normalised ``x`` to ``[0, 1]`` and normalised the continuity residual
by ``U / H``. All three choices are wrong for a long, thin channel:

* In a channel of length ``L`` and half-height ``H`` with ``H << L``,
  continuity ``u_x + v_y = 0`` forces ``V ~ U H / L``. Scaling ``v`` by ``U``
  asks the network to output values around ``1e-2`` for its second head,
  which it can only do by suppressing that head entirely.
* Inputs on ``[0, 1]`` are not centred, so a ``tanh`` first layer starts far
  from its symmetric, well-conditioned region.
* ``U / H`` is not the continuity scale. Both terms of the continuity
  equation are of size ``U / L = V / H``, so ``U / L`` is the correct
  normalisation and ``U / H`` inflates the residual by ``L / H`` (about 30x
  for the ST40 geometry), which drowns the momentum and boundary terms.

This version therefore stores ``L``, ``H``, ``U`` and ``P`` explicitly,
derives ``V = U H / L``, exposes centred coordinate scales, and exposes the
three residual normalisations as properties so the notebook and the library
cannot disagree about them.

``DimensionalPINN`` buffers now follow the module dtype instead of being
pinned to ``float32``, so ``.to(dtype=torch.float64)`` produces a genuinely
double-precision model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class DimensionalScales:
    """Scales that normalise network variables without changing PDE units.

    Attributes
    ----------
    length_scale_m:
        Streamwise channel length ``L`` in metres.
    height_scale_m:
        Transverse half-height ``H`` of the healthy channel in metres.
    velocity_scale_m_s:
        Streamwise velocity scale ``U`` (the mean inlet velocity).
    pressure_scale_pa:
        Pressure scale ``P``, normally ``rho * U**2``.
    v_scale_m_s:
        Transverse velocity scale ``V``. Defaults to ``U * H / L``, which is
        the value implied by incompressibility.
    """

    length_scale_m: float
    height_scale_m: float
    velocity_scale_m_s: float
    pressure_scale_pa: float
    v_scale_m_s: Optional[float] = None
    pressure_offset_pa: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "length_scale_m",
            "height_scale_m",
            "velocity_scale_m_s",
            "pressure_scale_pa",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.v_scale_m_s is None:
            object.__setattr__(
                self,
                "v_scale_m_s",
                self.velocity_scale_m_s * self.height_scale_m / self.length_scale_m,
            )
        if self.v_scale_m_s <= 0:
            raise ValueError("v_scale_m_s must be positive.")

    # ------------------------------------------------------------------
    # Coordinate normalisation: x -> [-1, 1], y -> [-1, 1]
    # ------------------------------------------------------------------
    @property
    def x_scale_m(self) -> float:
        return 0.5 * self.length_scale_m

    @property
    def x_offset_m(self) -> float:
        return 0.5 * self.length_scale_m

    @property
    def y_scale_m(self) -> float:
        return self.height_scale_m

    @property
    def y_offset_m(self) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Residual normalisation
    # ------------------------------------------------------------------
    @property
    def continuity_scale(self) -> float:
        """``U / L``, which equals ``V / H`` by construction."""
        return self.velocity_scale_m_s / self.length_scale_m

    @property
    def momentum_x_scale(self) -> float:
        return self.pressure_scale_pa / self.length_scale_m

    @property
    def momentum_y_scale(self) -> float:
        return self.pressure_scale_pa / self.height_scale_m

    @property
    def output_scales(self) -> tuple[float, float, float]:
        return (self.velocity_scale_m_s, float(self.v_scale_m_s), self.pressure_scale_pa)

    @classmethod
    def from_case_metadata(cls, metadata: dict[str, Any]) -> "DimensionalScales":
        geometry = metadata["geometry"]
        rho = float(metadata["density_kg_m3"])
        mean_velocity = float(metadata["mean_inlet_velocity_m_s"])
        return cls(
            length_scale_m=float(geometry["total_length_m"]),
            height_scale_m=0.5 * float(geometry["healthy_channel_height_m"]),
            velocity_scale_m_s=mean_velocity,
            pressure_scale_pa=rho * mean_velocity**2,
        )


class DimensionalPINN(nn.Module):
    """Wrap a dimensionless network so physics remains in physical SI units.

    ``compute_ns_residuals_nonnewtonian`` differentiates this wrapper with
    respect to physical coordinates. Autograd therefore applies the required
    chain rule automatically.

    All registered buffers are created with ``dtype`` (defaulting to the
    current torch default dtype) so that ``.to(dtype=torch.float64)`` gives a
    consistently double-precision model rather than silently downcasting the
    scaling constants.
    """

    def __init__(
        self,
        network: nn.Module,
        scales: DimensionalScales,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.network = network
        self.register_buffer(
            "coordinate_scale",
            torch.tensor([scales.x_scale_m, scales.y_scale_m], dtype=dtype),
        )
        self.register_buffer(
            "coordinate_offset",
            torch.tensor([scales.x_offset_m, scales.y_offset_m], dtype=dtype),
        )
        self.register_buffer(
            "output_scale",
            torch.tensor(list(scales.output_scales), dtype=dtype),
        )
        self.register_buffer(
            "output_offset",
            torch.tensor([0.0, 0.0, scales.pressure_offset_pa], dtype=dtype),
        )

    def normalise_coordinates(self, xy_m: torch.Tensor) -> torch.Tensor:
        return (xy_m - self.coordinate_offset) / self.coordinate_scale

    def forward(self, xy_m: torch.Tensor) -> torch.Tensor:
        unit_prediction = self.network(self.normalise_coordinates(xy_m))
        return unit_prediction * self.output_scale + self.output_offset