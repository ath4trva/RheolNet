"""Boundary-condition and conservation losses for the steady 2D planar-stenosis PINN.

Revision notes (v2)
-------------------
Two constraints were missing from the first version and both were failure
modes of the previous training run:

* **Cross-sectional flow conservation.** Continuity is a *local* constraint.
  A network can drive ``u_x + v_y`` towards zero pointwise while still
  losing almost all of the mass flux, because the pointwise residual is
  scale-free and a nearly-zero velocity field satisfies it trivially. The
  previous run reported a 96% inlet/outlet flow mismatch with a small
  normalised loss for exactly this reason. ``CrossSectionalFlow`` integrates
  ``Q(x) = int u dy`` at many streamwise stations and
  ``flow_conservation_loss`` penalises deviation from the exact prescribed
  inlet flow, giving the optimiser a *global* signal that a collapsed field
  cannot satisfy.

* **Outlet zero-gradient velocity.** Kiera's OpenFOAM outlet uses
  ``zeroGradient`` on ``U`` and a fixed pressure. Only the pressure part was
  enforced, so the network was free to distort the velocity field at the
  exit. ``outlet_velocity_gradient_loss`` adds ``du/dx = dv/dx = 0``.

The loss registry now also keeps ``continuity``, ``momentum_x`` and
``momentum_y`` separate instead of collapsing them into one ``physics``
number, so their individual magnitudes and gradients can be monitored.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn


def _require_xy(xy: torch.Tensor, name: str) -> None:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")


# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------
def _velocity_normaliser(
    prediction: torch.Tensor,
    u_scale_m_s: Optional[float],
    v_scale_m_s: Optional[float],
) -> torch.Tensor:
    """Return per-component divisors for a velocity residual.

    With ``V << U`` a single shared scale makes the transverse constraint
    numerically invisible: an error of order ``V`` in ``v`` contributes
    ``(V/U)**2 ~ 1e-3`` of what the same relative error in ``u`` contributes.
    Passing both scales normalises each component by its own magnitude.
    """
    if u_scale_m_s is None and v_scale_m_s is None:
        return torch.ones(2, dtype=prediction.dtype, device=prediction.device)
    if u_scale_m_s is None or v_scale_m_s is None:
        raise ValueError("Provide both u_scale_m_s and v_scale_m_s, or neither.")
    if u_scale_m_s <= 0 or v_scale_m_s <= 0:
        raise ValueError("Velocity scales must be positive.")
    return torch.tensor(
        [u_scale_m_s, v_scale_m_s], dtype=prediction.dtype, device=prediction.device
    )


def no_slip_loss(
    model: nn.Module,
    wall_xy_m: torch.Tensor,
    u_scale_m_s: Optional[float] = None,
    v_scale_m_s: Optional[float] = None,
) -> torch.Tensor:
    """Mean-square no-slip loss, enforcing ``u=v=0`` on a wall."""
    _require_xy(wall_xy_m, "wall_xy_m")
    prediction = model(wall_xy_m)
    divisor = _velocity_normaliser(prediction, u_scale_m_s, v_scale_m_s)
    return torch.mean((prediction[:, :2] / divisor) ** 2)


def inlet_velocity_loss(
    model: nn.Module,
    inlet_xy_m: torch.Tensor,
    inlet_uv_m_s: torch.Tensor,
    u_scale_m_s: Optional[float] = None,
    v_scale_m_s: Optional[float] = None,
) -> torch.Tensor:
    """Match the exact prescribed CFD inlet velocity profile."""
    _require_xy(inlet_xy_m, "inlet_xy_m")
    if inlet_uv_m_s.shape != (inlet_xy_m.shape[0], 2):
        raise ValueError("inlet_uv_m_s must have shape (n_inlet, 2).")
    prediction = model(inlet_xy_m)
    divisor = _velocity_normaliser(prediction, u_scale_m_s, v_scale_m_s)
    return torch.mean(((prediction[:, :2] - inlet_uv_m_s) / divisor) ** 2)


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


def outlet_velocity_gradient_loss(
    model: nn.Module,
    outlet_xy_m: torch.Tensor,
    u_scale_m_s: float,
    v_scale_m_s: float,
    length_scale_m: float,
) -> torch.Tensor:
    """Enforce the OpenFOAM ``zeroGradient`` outlet condition on velocity.

    Penalises ``du/dx`` and ``dv/dx`` at the outlet, each normalised by its
    own characteristic gradient (``U/L`` and ``V/L``) so the two components
    contribute comparably.
    """
    _require_xy(outlet_xy_m, "outlet_xy_m")
    if u_scale_m_s <= 0 or v_scale_m_s <= 0 or length_scale_m <= 0:
        raise ValueError("Scales must be positive.")

    xy = outlet_xy_m.detach().clone().requires_grad_(True)
    prediction = model(xy)
    u = prediction[:, 0:1]
    v = prediction[:, 1:2]

    def _d_dx(output: torch.Tensor) -> torch.Tensor:
        # An exactly-constant field carries no autograd graph back to xy at
        # all; its x-derivative is zero by definition rather than an error.
        if not output.requires_grad:
            return torch.zeros_like(output)
        gradient = torch.autograd.grad(
            output,
            xy,
            torch.ones_like(output),
            create_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            return torch.zeros_like(output)
        return gradient[:, 0:1]

    u_x = _d_dx(u)
    v_x = _d_dx(v)

    u_gradient_scale = u_scale_m_s / length_scale_m
    v_gradient_scale = v_scale_m_s / length_scale_m
    return torch.mean((u_x / u_gradient_scale) ** 2) + torch.mean(
        (v_x / v_gradient_scale) ** 2
    )


# ----------------------------------------------------------------------
# Global mass conservation
# ----------------------------------------------------------------------
class CrossSectionalFlow:
    """Gauss-Legendre integrator for ``Q(x) = int_{-h(x)}^{h(x)} u(x, y) dy``.

    The station coordinates and quadrature weights are fixed at construction,
    so evaluating the flow during training costs one extra forward pass over
    ``n_stations * n_quadrature_points`` points and no derivatives.
    """

    def __init__(
        self,
        geometry,
        n_stations: int = 21,
        n_quadrature_points: int = 121,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        x_stations_m: Optional[np.ndarray] = None,
    ) -> None:
        if n_stations < 2:
            raise ValueError("At least two flow stations are required.")
        if n_quadrature_points < 2:
            raise ValueError("At least two quadrature points are required.")

        if x_stations_m is None:
            x_stations_m = np.linspace(0.0, geometry.total_length_m, n_stations)
        x_stations_m = np.asarray(x_stations_m, dtype=np.float64).reshape(-1)

        nodes, weights = np.polynomial.legendre.leggauss(n_quadrature_points)
        half_heights = np.asarray(geometry.half_height(x_stations_m), dtype=np.float64)

        # y_ij = h(x_i) * xi_j  ->  dy = h(x_i) dxi
        y = half_heights[:, None] * nodes[None, :]
        x = np.repeat(x_stations_m[:, None], nodes.size, axis=1)
        xy = np.column_stack((x.reshape(-1), y.reshape(-1)))

        self.n_stations = x_stations_m.size
        self.n_quadrature_points = int(nodes.size)
        self.x_stations_m = x_stations_m
        self.half_heights_m = half_heights
        self.xy = torch.tensor(xy, dtype=dtype, device=device)
        self._weights = torch.tensor(weights, dtype=dtype, device=device)
        self._half_heights = torch.tensor(half_heights, dtype=dtype, device=device)

    def flow(self, model: nn.Module) -> torch.Tensor:
        """Return ``Q(x)`` at every station, shape ``(n_stations,)``."""
        prediction = model(self.xy)
        u = prediction[:, 0].reshape(self.n_stations, self.n_quadrature_points)
        return self._half_heights * torch.sum(u * self._weights, dim=1)

    __call__ = flow


def exact_inlet_flow(u_inlet_m_s: np.ndarray, half_height_m: float) -> float:
    """Exact prescribed inlet flow from the equal-area CFD inlet faces.

    The 120 OpenFOAM inlet faces are equal-area, so the midpoint rule is
    exact for the prescribed profile: ``Q = mean(u) * 2H``.
    """
    u = np.asarray(u_inlet_m_s, dtype=np.float64).reshape(-1)
    if u.size == 0:
        raise ValueError("u_inlet_m_s must not be empty.")
    if half_height_m <= 0:
        raise ValueError("half_height_m must be positive.")
    return float(np.mean(u) * 2.0 * half_height_m)


def flow_conservation_loss(
    flow_m2_s: torch.Tensor,
    target_flow_m2_s: float,
) -> torch.Tensor:
    """Mean squared *relative* deviation of ``Q(x)`` from the target flow."""
    if target_flow_m2_s == 0.0:
        raise ValueError("target_flow_m2_s must be non-zero.")
    target = torch.as_tensor(
        target_flow_m2_s, dtype=flow_m2_s.dtype, device=flow_m2_s.device
    )
    return torch.mean(((flow_m2_s - target) / target) ** 2)


# ----------------------------------------------------------------------
# Weighting and total
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PlanarStenosisLossWeights:
    continuity: float = 1.0
    momentum_x: float = 1.0
    momentum_y: float = 1.0
    wall: float = 20.0
    inlet: float = 20.0
    outlet_pressure: float = 10.0
    outlet_velocity_gradient: float = 1.0
    flow: float = 10.0

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) < 0 for item in fields(self)):
            raise ValueError("Loss weights must be non-negative.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in fields(self))


def total_planar_stenosis_loss(
    terms: Mapping[str, torch.Tensor],
    weights: PlanarStenosisLossWeights = PlanarStenosisLossWeights(),
) -> torch.Tensor:
    """Combine named losses while rejecting silently omitted constraints."""
    required = set(weights.names)
    missing = required - set(terms)
    unexpected = set(terms) - required
    if missing or unexpected:
        raise ValueError(
            f"Expected loss terms {sorted(required)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    total = None
    for name in weights.names:
        contribution = getattr(weights, name) * terms[name]
        total = contribution if total is None else total + contribution
    return total