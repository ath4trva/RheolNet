import torch


def navier_stokes_physics_loss(
    continuity,
    momentum_x,
    momentum_y,
):
    """
    Mean-squared Navier-Stokes residual loss.
    """

    continuity_loss = torch.mean(
        continuity**2
    )

    momentum_x_loss = torch.mean(
        momentum_x**2
    )

    momentum_y_loss = torch.mean(
        momentum_y**2
    )

    total_physics_loss = (
        continuity_loss
        + momentum_x_loss
        + momentum_y_loss
    )

    return total_physics_loss