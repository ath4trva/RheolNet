
import torch


def compute_ns_residuals_nonnewtonian(xy, eta0=0.056, eta_inf=0.0035, lam=3.313, n=0.3568, a=2.0):
    
    xy = xy.detach().requires_grad_(True)
    
    pred = model(xy)
    u = pred[:, 0:1]   # x-velocity, shape [N, 1]
    v = pred[:, 1:2]   # y-velocity, shape [N, 1]
    p = pred[:, 2:3]   # pressure,   shape [N, 1]

    def grad(f, wrt):
        return torch.autograd.grad(
            f, wrt,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True
        )[0]

    # First derivatives
    grad_u = grad(u, xy)          # [N, 2]: columns are du/dx, du/dy
    grad_p = grad(p, xy)          # [N, 2]: columns are dp/dx, dp/dy

    du_dx = grad_u[:, 0:1]        # [N, 1]
    du_dy = grad_u[:, 1:2]        # [N, 1]  ← this is what drives shear rate
    dp_dx = grad_p[:, 0:1]        # [N, 1]

    # Shear rate: how fast velocity changes across the pipe
    # Add 1e-8 to avoid exactly zero (causes numerical issues in CY formula)
    gamma_dot = torch.abs(du_dy) + 1e-8    # [N, 1]

    # Local viscosity at every point via Carreau-Yasuda
    eta = carreau_yasuda_viscosity(gamma_dot, eta0, eta_inf, lam, n, a)  # [N, 1]

    # Viscous stress = eta * du/dy  (force one fluid layer exerts on neighbour)
    stress = eta * du_dy                   # [N, 1]

    # d/dy(eta * du/dy) — net viscous force per unit volume
    # autograd applies product rule automatically here
    d_stress_dy = grad(stress, xy)[:, 1:2] # [N, 1]

    # Momentum residual: pressure force + viscous force = 0
    R_momentum = -dp_dx + d_stress_dy      # [N, 1]

    # Continuity: fully-developed flow → du/dx = 0
    R_continuity = du_dx                   # [N, 1]

    # Return flat [N] tensors so they're easy to work with
    return (R_momentum.squeeze(),
            R_continuity.squeeze(),
            eta.squeeze(),
            gamma_dot.squeeze())