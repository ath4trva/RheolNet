"""Tests for the flow-conservation and outlet zero-gradient velocity losses."""

import pytest
import torch
from torch import nn

from geometry.planar_stenosis import PlanarStenosisGeometry
from geometry.planar_stenosis_losses import (
    CrossSectionalFlow,
    PlanarStenosisLossWeights,
    exact_inlet_flow,
    flow_conservation_loss,
    outlet_velocity_gradient_loss,
    total_planar_stenosis_loss,
)

DTYPE = torch.float64


class UniformFlowModel(nn.Module):
    """``u = u0``, ``v = 0``, ``p = 0`` everywhere."""

    def __init__(self, u0=0.27516):
        super().__init__()
        self.u0 = u0

    def forward(self, xy):
        u = torch.full_like(xy[:, 0:1], self.u0)
        zeros = torch.zeros_like(u)
        return torch.cat((u, zeros, zeros), dim=1)


class LeakyFlowModel(nn.Module):
    """``u`` grows linearly with ``x``, so ``Q(x)`` is not conserved."""

    def __init__(self, u0=0.27516, length=0.092):
        super().__init__()
        self.u0 = u0
        self.length = length

    def forward(self, xy):
        x = xy[:, 0:1]
        u = self.u0 * (1.0 + x / self.length)
        zeros = torch.zeros_like(u)
        return torch.cat((u, zeros, zeros), dim=1)


class StreamwiseVaryingModel(nn.Module):
    """Velocity that genuinely varies with ``x`` at the outlet."""

    def forward(self, xy):
        x = xy[:, 0:1]
        u = 0.3 + 2.0 * x
        v = 0.01 - 0.5 * x
        return torch.cat((u, v, torch.zeros_like(u)), dim=1)


def straight_channel():
    """A channel with no stenosis, so ``h(x)`` is constant."""
    return PlanarStenosisGeometry(throat_height_m=0.006)


def stenosed_channel():
    return PlanarStenosisGeometry()


# ----------------------------------------------------------------------
# Flow conservation
# ----------------------------------------------------------------------
def test_exact_inlet_flow_matches_mean_velocity_times_height():
    u = torch.linspace(0.0, 1.0, 120).numpy()
    q = exact_inlet_flow(u, 0.003)
    assert q == pytest.approx(u.mean() * 0.006, rel=1e-12)


def test_constant_flow_gives_zero_flow_loss():
    geometry = straight_channel()
    integrator = CrossSectionalFlow(geometry, n_stations=21, n_quadrature_points=121, dtype=DTYPE)
    model = UniformFlowModel(u0=0.27516)
    q = integrator.flow(model)
    q_target = 0.27516 * 2.0 * geometry.healthy_half_height_m
    assert torch.allclose(q, torch.full_like(q, q_target), rtol=1e-10)
    assert float(flow_conservation_loss(q, q_target)) == pytest.approx(0.0, abs=1e-20)


def test_quadrature_is_exact_for_a_parabolic_profile_through_the_throat():
    """Gauss-Legendre must integrate the analytic conserved profile exactly."""
    geometry = stenosed_channel()
    integrator = CrossSectionalFlow(geometry, n_stations=11, n_quadrature_points=41, dtype=DTYPE)
    q_target = 0.27516 * 2.0 * geometry.healthy_half_height_m

    class ParabolicConservedModel(nn.Module):
        def forward(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            h = torch.tensor(
                geometry.half_height(x.detach().cpu().numpy().reshape(-1)),
                dtype=xy.dtype,
                device=xy.device,
            ).reshape(-1, 1)
            # 1.5 * Q / (2h) * (1 - (y/h)^2) integrates to exactly Q.
            u = 1.5 * q_target / (2.0 * h) * (1.0 - (y / h) ** 2)
            zeros = torch.zeros_like(u)
            return torch.cat((u, zeros, zeros), dim=1)

    q = integrator.flow(ParabolicConservedModel())
    assert torch.allclose(q, torch.full_like(q, q_target), rtol=1e-10)
    assert float(flow_conservation_loss(q, q_target)) < 1e-18


def test_incorrect_flow_gives_positive_loss():
    geometry = straight_channel()
    integrator = CrossSectionalFlow(geometry, n_stations=21, n_quadrature_points=121, dtype=DTYPE)
    q = integrator.flow(LeakyFlowModel(u0=0.27516, length=geometry.total_length_m))
    q_target = 0.27516 * 2.0 * geometry.healthy_half_height_m
    assert float(flow_conservation_loss(q, q_target)) > 1e-3


def test_collapsed_field_is_heavily_penalised():
    """A near-zero velocity field satisfies pointwise continuity but not flow."""
    geometry = stenosed_channel()
    integrator = CrossSectionalFlow(geometry, n_stations=21, n_quadrature_points=121, dtype=DTYPE)
    q = integrator.flow(UniformFlowModel(u0=1e-6))
    q_target = 0.27516 * 2.0 * geometry.healthy_half_height_m
    assert float(flow_conservation_loss(q, q_target)) == pytest.approx(1.0, rel=1e-3)


def test_flow_loss_is_differentiable():
    geometry = straight_channel()
    integrator = CrossSectionalFlow(geometry, n_stations=5, n_quadrature_points=21, dtype=DTYPE)

    class TrainableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))

        def forward(self, xy):
            u = self.scale * torch.ones_like(xy[:, 0:1])
            zeros = torch.zeros_like(u)
            return torch.cat((u, zeros, zeros), dim=1)

    model = TrainableModel()
    loss = flow_conservation_loss(integrator.flow(model), 0.27516 * 0.006)
    loss.backward()
    assert model.scale.grad is not None
    assert float(model.scale.grad) != 0.0


# ----------------------------------------------------------------------
# Outlet zero-gradient velocity
# ----------------------------------------------------------------------
def test_zero_outlet_gradients_give_zero_loss():
    outlet = torch.stack(
        (
            torch.full((17,), 0.092, dtype=DTYPE),
            torch.linspace(-0.003, 0.003, 17, dtype=DTYPE),
        ),
        dim=1,
    )
    loss = outlet_velocity_gradient_loss(
        UniformFlowModel(), outlet, u_scale_m_s=0.27516, v_scale_m_s=0.009, length_scale_m=0.092
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-20)


def test_nonzero_outlet_gradients_give_positive_loss():
    outlet = torch.stack(
        (
            torch.full((17,), 0.092, dtype=DTYPE),
            torch.linspace(-0.003, 0.003, 17, dtype=DTYPE),
        ),
        dim=1,
    )
    loss = outlet_velocity_gradient_loss(
        StreamwiseVaryingModel(),
        outlet,
        u_scale_m_s=0.27516,
        v_scale_m_s=0.009,
        length_scale_m=0.092,
    )
    assert float(loss) > 0.0


def test_outlet_gradient_loss_rejects_bad_scales():
    outlet = torch.zeros(4, 2, dtype=DTYPE)
    with pytest.raises(ValueError):
        outlet_velocity_gradient_loss(UniformFlowModel(), outlet, 0.0, 1.0, 1.0)


# ----------------------------------------------------------------------
# Weight registry
# ----------------------------------------------------------------------
def test_total_loss_requires_every_named_term():
    weights = PlanarStenosisLossWeights()
    terms = {name: torch.tensor(1.0, dtype=DTYPE) for name in weights.names}
    total = total_planar_stenosis_loss(terms, weights)
    expected = sum(getattr(weights, name) for name in weights.names)
    assert float(total) == pytest.approx(expected)

    terms.pop("flow")
    with pytest.raises(ValueError):
        total_planar_stenosis_loss(terms, weights)


def test_total_loss_rejects_unexpected_terms():
    weights = PlanarStenosisLossWeights()
    terms = {name: torch.tensor(1.0, dtype=DTYPE) for name in weights.names}
    terms["physics"] = torch.tensor(1.0, dtype=DTYPE)
    with pytest.raises(ValueError):
        total_planar_stenosis_loss(terms, weights)


def test_pde_terms_are_separate_not_combined():
    weights = PlanarStenosisLossWeights()
    assert "continuity" in weights.names
    assert "momentum_x" in weights.names
    assert "momentum_y" in weights.names
    assert "physics" not in weights.names


# ----------------------------------------------------------------------
# Anisotropic boundary normalisation
# ----------------------------------------------------------------------
def test_no_slip_loss_scales_each_velocity_component_separately():
    from geometry.planar_stenosis_losses import no_slip_loss

    class WallLeakModel(nn.Module):
        def forward(self, xy):
            u = torch.zeros_like(xy[:, 0:1])
            v = torch.full_like(u, 1e-3)
            return torch.cat((u, v, torch.zeros_like(u)), dim=1)

    wall = torch.zeros(8, 2, dtype=DTYPE)
    unscaled = float(no_slip_loss(WallLeakModel(), wall))
    scaled = float(no_slip_loss(WallLeakModel(), wall, 0.27516, 0.27516 * 0.003 / 0.092))
    # The same transverse leak must register far more strongly once v is
    # measured against V rather than U.
    assert scaled > 100.0 * unscaled


def test_velocity_normaliser_requires_both_scales():
    from geometry.planar_stenosis_losses import no_slip_loss

    wall = torch.zeros(4, 2, dtype=DTYPE)
    with pytest.raises(ValueError):
        no_slip_loss(UniformFlowModel(), wall, u_scale_m_s=1.0)