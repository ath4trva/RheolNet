import torch

from physics.carreau_yasuda import (
    carreau_yasuda_viscosity,
)


# ===============================================================
# BASIC SHAPE TEST
# ===============================================================

def test_output_shape_matches_input():

    gamma_dot = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
            [100.0],
        ],
        dtype=torch.float32,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    assert eta.shape == gamma_dot.shape


# ===============================================================
# ZERO-SHEAR LIMIT
# ===============================================================

def test_zero_shear_viscosity_equals_eta0():

    eta0 = 0.056
    eta_inf = 0.0035

    gamma_dot = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot,
        eta0=eta0,
        eta_inf=eta_inf,
    )

    expected = torch.tensor(
        [[eta0]],
        dtype=torch.float32,
    )

    assert torch.allclose(
        eta,
        expected,
        atol=1e-6,
    )


# ===============================================================
# HIGH-SHEAR LIMIT
# ===============================================================

def test_high_shear_viscosity_approaches_eta_inf():

    eta0 = 0.056
    eta_inf = 0.0035

    gamma_dot = torch.tensor(
        [[1e10]],
        dtype=torch.float32,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot,
        eta0=eta0,
        eta_inf=eta_inf,
        lam=3.313,
        n=0.3568,
        a=2.0,
    )

    # At extremely high shear, viscosity should
    # approach eta_inf.

    assert torch.allclose(
        eta,
        torch.tensor([[eta_inf]]),
        atol=1e-4,
    )


# ===============================================================
# SHEAR-THINNING TEST
# ===============================================================

def test_viscosity_decreases_with_shear_rate():

    gamma_dot = torch.tensor(
        [
            [0.0],
            [0.1],
            [1.0],
            [10.0],
            [100.0],
        ],
        dtype=torch.float32,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    # Carreau-Yasuda blood model should be
    # shear-thinning:
    #
    # increasing shear rate
    #       ↓
    # decreasing viscosity

    assert torch.all(
        eta[:-1] >= eta[1:]
    )


# ===============================================================
# VISCOSITY BOUNDS
# ===============================================================

def test_viscosity_stays_between_eta_limits():

    eta0 = 0.056
    eta_inf = 0.0035

    gamma_dot = torch.logspace(
        -6,
        6,
        200,
    ).reshape(-1, 1)

    eta = carreau_yasuda_viscosity(
        gamma_dot,
        eta0=eta0,
        eta_inf=eta_inf,
    )

    assert torch.all(
        eta <= eta0 + 1e-6
    )

    assert torch.all(
        eta >= eta_inf - 1e-6
    )


# ===============================================================
# FINITE VALUES TEST
# ===============================================================

def test_viscosity_contains_no_nan_or_inf():

    gamma_dot = torch.logspace(
        -8,
        8,
        300,
    ).reshape(-1, 1)

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    assert torch.isfinite(
        eta
    ).all()


# ===============================================================
# POSITIVE VISCOSITY TEST
# ===============================================================

def test_viscosity_is_positive():

    gamma_dot = torch.logspace(
        -6,
        6,
        200,
    ).reshape(-1, 1)

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    assert torch.all(
        eta > 0
    )


# ===============================================================
# AUTOGRAD TEST
# ===============================================================

def test_carreau_yasuda_supports_autograd():

    gamma_dot = torch.tensor(
        [
            [0.1],
            [1.0],
            [10.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    total_eta = eta.sum()

    total_eta.backward()

    # Gradient must exist because Navier-Stokes
    # needs to differentiate viscosity through
    # the velocity gradients.

    assert gamma_dot.grad is not None

    assert torch.isfinite(
        gamma_dot.grad
    ).all()


# ===============================================================
# SHEAR-THINNING GRADIENT SIGN
# ===============================================================

def test_viscosity_gradient_is_negative_for_positive_shear():

    gamma_dot = torch.tensor(
        [
            [0.1],
            [1.0],
            [10.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot
    )

    eta.sum().backward()

    # For a shear-thinning model (n < 1),
    # viscosity should decrease as shear increases.
    #
    # Therefore:
    #
    # d eta / d gamma <= 0

    assert torch.all(
        gamma_dot.grad <= 0
    )


# ===============================================================
# NEWTONIAN LIMIT
# ===============================================================

def test_eta0_equal_eta_inf_gives_constant_viscosity():

    constant_eta = 0.01

    gamma_dot = torch.tensor(
        [
            [0.0],
            [0.1],
            [1.0],
            [10.0],
            [1000.0],
        ],
        dtype=torch.float32,
    )

    eta = carreau_yasuda_viscosity(
        gamma_dot,
        eta0=constant_eta,
        eta_inf=constant_eta,
    )

    expected = torch.full_like(
        gamma_dot,
        constant_eta,
    )

    assert torch.allclose(
        eta,
        expected,
        atol=1e-7,
    )
