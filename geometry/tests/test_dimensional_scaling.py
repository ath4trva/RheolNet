"""Tests for the anisotropic, centred, dtype-safe dimensional scaling."""

import torch

from geometry.dimensional_scaling import DimensionalPINN, DimensionalScales
from models.pinn_core import PINN

METADATA = {
    "density_kg_m3": 1060.0,
    "mean_inlet_velocity_m_s": 0.27516,
    "geometry": {"total_length_m": 0.092, "healthy_channel_height_m": 0.006},
}


def _scales():
    return DimensionalScales.from_case_metadata(METADATA)


def test_transverse_velocity_scale_satisfies_continuity_invariant():
    """U / L must equal V / H, otherwise continuity cannot be normalised."""
    scales = _scales()
    u_over_l = scales.velocity_scale_m_s / scales.length_scale_m
    v_over_h = scales.v_scale_m_s / scales.height_scale_m
    assert abs(u_over_l - v_over_h) <= 1e-12 * u_over_l


def test_continuity_scale_is_u_over_l_not_u_over_h():
    scales = _scales()
    assert abs(
        scales.continuity_scale
        - scales.velocity_scale_m_s / scales.length_scale_m
    ) <= 1e-15
    # The old (wrong) normalisation differs by the aspect ratio L/H ~ 30.
    wrong = scales.velocity_scale_m_s / scales.height_scale_m
    assert wrong / scales.continuity_scale > 10.0


def test_u_and_v_output_scales_differ():
    scales = _scales()
    u_scale, v_scale, _ = scales.output_scales
    assert v_scale < 0.1 * u_scale


def test_coordinates_are_centred_on_minus_one_to_one():
    scales = _scales()
    model = DimensionalPINN(PINN(hidden_layers=2, hidden_width=8), scales)
    length = scales.length_scale_m
    height = scales.height_scale_m
    xy = torch.tensor(
        [[0.0, -height], [length, height], [0.5 * length, 0.0]],
        dtype=torch.get_default_dtype(),
    )
    normalised = model.normalise_coordinates(xy)
    assert torch.allclose(
        normalised,
        torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]], dtype=normalised.dtype),
        atol=1e-12,
    )


def test_buffers_follow_requested_dtype():
    scales = _scales()
    model = DimensionalPINN(
        PINN(hidden_layers=2, hidden_width=8), scales, dtype=torch.float64
    )
    assert model.coordinate_scale.dtype == torch.float64
    assert model.output_scale.dtype == torch.float64


def test_model_to_float64_converts_buffers_and_parameters():
    scales = _scales()
    model = DimensionalPINN(PINN(hidden_layers=2, hidden_width=8), scales)
    model = model.to(dtype=torch.float64)
    assert model.output_scale.dtype == torch.float64
    xy = torch.zeros(4, 2, dtype=torch.float64)
    assert model(xy).dtype == torch.float64