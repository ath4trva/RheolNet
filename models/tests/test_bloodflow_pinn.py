import torch

from models.bloodflow_pinn import BloodFlowPINN


def test_documented_four_output_configuration():
    model = BloodFlowPINN(out_dim=4, fourier_enabled=True)
    assert model(torch.randn(13, 2)).shape == (13, 4)


def test_mixed_six_output_configuration():
    model = BloodFlowPINN(out_dim=6, fourier_enabled=False)
    assert model(torch.randn(13, 2)).shape == (13, 6)


def test_on_off_models_use_requested_depth_and_width():
    for enabled in (False, True):
        model = BloodFlowPINN(
            out_dim=6,
            hidden_layers=5,
            hidden_width=128,
            fourier_enabled=enabled,
        )
        linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        assert len(linear_layers) == 6
        assert all(layer.out_features == 128 for layer in linear_layers[:-1])
        assert linear_layers[-1].out_features == 6
