import torch


def carreau_yasuda_viscosity(
    gamma_dot,
    eta0=0.056,
    eta_inf=0.0035,
    lam=3.313,
    n=0.3568,
    a=2.0,
):
    """
    Carreau-Yasuda model for shear-dependent blood viscosity.

    Parameters
    ----------
    gamma_dot : torch.Tensor
        Local shear rate.

    eta0 : float
        Zero-shear viscosity [Pa.s].

    eta_inf : float
        Infinite-shear viscosity [Pa.s].

    lam : float
        Time constant [s].

    n : float
        Power-law index.

    a : float
        Yasuda parameter.

    Returns
    -------
    torch.Tensor
        Local dynamic viscosity.
    """

    gamma_dot = torch.clamp(
        gamma_dot,
        min=0.0,
    )

    eta = (
        eta_inf
        + (eta0 - eta_inf)
        * (
            1.0
            + (lam * gamma_dot) ** a
        ) ** ((n - 1.0) / a)
    )

    return eta