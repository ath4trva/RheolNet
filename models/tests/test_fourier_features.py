import torch

from models.fourier_features import FourierFeatureEncoding


def test_projection_is_fixed_non_trainable_and_reproducible():
    first = FourierFeatureEncoding(2, num_frequencies=16, seed=7)
    second = FourierFeatureEncoding(2, num_frequencies=16, seed=7)
    assert "projection" not in dict(first.named_parameters())
    assert "projection" in dict(first.named_buffers())
    assert torch.equal(first.projection, second.projection)


def test_enabled_shape_and_disabled_identity():
    coordinates = torch.randn(11, 2)
    enabled = FourierFeatureEncoding(2, num_frequencies=20, enabled=True, include_input=True)
    disabled = FourierFeatureEncoding(2, num_frequencies=20, enabled=False)
    assert enabled(coordinates).shape == (11, 42)
    assert torch.equal(disabled(coordinates), coordinates)


def test_coordinate_gradient_is_finite():
    coordinates = torch.randn(8, 2, requires_grad=True)
    encoded = FourierFeatureEncoding(2, num_frequencies=12)(coordinates)
    gradient = torch.autograd.grad(encoded.square().sum(), coordinates)[0]
    assert torch.isfinite(gradient).all()
