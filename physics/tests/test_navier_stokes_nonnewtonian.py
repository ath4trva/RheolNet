import torch
import torch.nn as nn

from physics.navier_stokes import (
    compute_ns_residuals_nonnewtonian,
)


class ManufacturedPoiseuille(nn.Module):
    """
    Exact Newtonian Poiseuille solution:

        u = 4 y (1-y)
        v = 0
        p = -8 x

    With eta0 = eta_inf = 1,
    Carreau-Yasuda reduces to constant viscosity.
    """

    def forward(self, xy):

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        u = 4.0 * y * (1.0 - y)

        # Written this way so v remains connected
        # to the autograd graph.
        v = (
            0.0 * x
            + 0.0 * y
        )

        p = -8.0 * x

        return torch.cat(
            [u, v, p],
            dim=1,
        )


def test_residual_shapes():

    model = ManufacturedPoiseuille()

    xy = torch.rand(
        32,
        2,
        requires_grad=True,
    )

    (
        continuity,
        momentum_x,
        momentum_y,
        eta,
        gamma_dot,
    ) = compute_ns_residuals_nonnewtonian(
        model,
        xy,
        rho=1.0,
        eta0=1.0,
        eta_inf=1.0,
    )

    assert continuity.shape == (32, 1)
    assert momentum_x.shape == (32, 1)
    assert momentum_y.shape == (32, 1)
    assert eta.shape == (32, 1)
    assert gamma_dot.shape == (32, 1)


def test_newtonian_limit_poisseuille():

    model = ManufacturedPoiseuille()

    xy = torch.tensor(
        [
            [0.2, 0.10],
            [0.4, 0.25],
            [0.6, 0.75],
            [0.8, 0.90],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    (
        continuity,
        momentum_x,
        momentum_y,
        eta,
        gamma_dot,
    ) = compute_ns_residuals_nonnewtonian(
        model,
        xy,
        rho=1.0,

        # eta0 == eta_inf means
        # viscosity is exactly constant.
        eta0=1.0,
        eta_inf=1.0,
    )

    assert torch.allclose(
        continuity,
        torch.zeros_like(continuity),
        atol=1e-5,
    )

    assert torch.allclose(
        momentum_x,
        torch.zeros_like(momentum_x),
        atol=1e-4,
    )

    assert torch.allclose(
        momentum_y,
        torch.zeros_like(momentum_y),
        atol=1e-5,
    )


def test_viscosity_is_finite():

    model = ManufacturedPoiseuille()

    xy = torch.rand(
        32,
        2,
        requires_grad=True,
    )

    (
        _,
        _,
        _,
        eta,
        gamma_dot,
    ) = compute_ns_residuals_nonnewtonian(
        model,
        xy,
    )

    assert torch.isfinite(eta).all()

    assert torch.isfinite(
        gamma_dot
    ).all()

    assert (eta > 0).all()

    assert (gamma_dot >= 0).all()