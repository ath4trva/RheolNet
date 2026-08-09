import torch

from .carreau_yasuda import carreau_yasuda_viscosity


def gradients(output, inputs):
    """
    Compute first derivative of output with respect to inputs.
    """

    return torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def compute_ns_residuals_nonnewtonian(
    model,
    xy,
    rho=1.0,
    eta0=0.056,
    eta_inf=0.0035,
    lam=3.313,
    n=0.3568,
    a=2.0,
    eps=1e-12,
):
    """
    Steady 2D incompressible Navier-Stokes residuals
    with Carreau-Yasuda viscosity.

    Model input:
        xy -> [x, y]

    Model output:
        [u, v, p]

    Returns:
        continuity
        momentum_x
        momentum_y
        eta
        gamma_dot
    """

    # ----------------------------------------------------------
    # Make coordinates differentiable
    # ----------------------------------------------------------

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    # ----------------------------------------------------------
    # Neural-network prediction
    # ----------------------------------------------------------

    pred = model(xy)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    # ----------------------------------------------------------
    # First derivatives
    # ----------------------------------------------------------

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    grad_p = gradients(p, xy)

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]

    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]

    # ----------------------------------------------------------
    # Strain-rate tensor
    #
    # D = 1/2 (grad(u) + grad(u)^T)
    # ----------------------------------------------------------

    D_xx = u_x

    D_yy = v_y

    D_xy = 0.5 * (
        u_y + v_x
    )

    # ----------------------------------------------------------
    # Effective shear rate
    #
    # gamma_dot = sqrt(2 * D:D)
    #
    # In simple Poiseuille flow this reduces to:
    #
    # gamma_dot = |du/dy|
    # ----------------------------------------------------------

    gamma_dot = torch.sqrt(
        2.0 * D_xx**2
        + 2.0 * D_yy**2
        + 4.0 * D_xy**2
        + eps
    )

    # ----------------------------------------------------------
    # Carreau-Yasuda viscosity
    # ----------------------------------------------------------

    eta = carreau_yasuda_viscosity(
        gamma_dot=gamma_dot,
        eta0=eta0,
        eta_inf=eta_inf,
        lam=lam,
        n=n,
        a=a,
    )

    # ----------------------------------------------------------
    # Non-Newtonian viscous stress
    #
    # tau = 2 eta D
    # ----------------------------------------------------------

    tau_xx = (
        2.0
        * eta
        * D_xx
    )

    tau_yy = (
        2.0
        * eta
        * D_yy
    )

    tau_xy = (
        2.0
        * eta
        * D_xy
    )

    # ----------------------------------------------------------
    # Divergence of viscous stress
    # ----------------------------------------------------------

    grad_tau_xx = gradients(
        tau_xx,
        xy,
    )

    grad_tau_xy = gradients(
        tau_xy,
        xy,
    )

    grad_tau_yy = gradients(
        tau_yy,
        xy,
    )

    tau_xx_x = grad_tau_xx[:, 0:1]

    tau_xy_y = grad_tau_xy[:, 1:2]

    tau_xy_x = grad_tau_xy[:, 0:1]

    tau_yy_y = grad_tau_yy[:, 1:2]

    # ----------------------------------------------------------
    # Continuity
    #
    # du/dx + dv/dy = 0
    # ----------------------------------------------------------

    continuity = (
        u_x
        + v_y
    )

    # ----------------------------------------------------------
    # X momentum
    #
    # rho(u ux + v uy)
    # + px
    # - div(tau)_x
    # = 0
    # ----------------------------------------------------------

    momentum_x = (
        rho
        * (
            u * u_x
            + v * u_y
        )
        + p_x
        - (
            tau_xx_x
            + tau_xy_y
        )
    )

    # ----------------------------------------------------------
    # Y momentum
    # ----------------------------------------------------------

    momentum_y = (
        rho
        * (
            u * v_x
            + v * v_y
        )
        + p_y
        - (
            tau_xy_x
            + tau_yy_y
        )
    )

    return (
        continuity,
        momentum_x,
        momentum_y,
        eta,
        gamma_dot,
    )