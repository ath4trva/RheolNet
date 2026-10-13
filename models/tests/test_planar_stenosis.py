import numpy as np
import torch

from geometry.planar_stenosis import PlanarStenosisGeometry
from geometry.planar_stenosis_sampling import (
    sample_inlet_points,
    sample_interior_points,
    sample_outlet_points,
    sample_wall_points,
)


def test_st40_geometry_matches_openfoam_breakpoints():
    geometry = PlanarStenosisGeometry.from_severity_percent(40.0)

    assert np.isclose(geometry.top_wall(0.0), 0.003)
    assert np.isclose(geometry.top_wall(0.042), 0.0018)
    assert np.isclose(geometry.top_wall(0.046), 0.0018)
    assert np.isclose(geometry.top_wall(0.062), 0.003)
    assert np.isclose(geometry.bottom_wall(0.046), -0.0018)
    assert np.isclose(geometry.severity_percent, 40.0)


def test_geometry_rejects_points_outside_the_fluid():
    geometry = PlanarStenosisGeometry.from_severity_percent(60.0)
    points = np.array([[0.046, 0.0012], [0.046, 0.00121], [0.093, 0.0]])

    assert geometry.contains(points).tolist() == [True, False, False]


def test_samplers_return_physical_points_on_the_correct_boundaries():
    geometry = PlanarStenosisGeometry.from_severity_percent(40.0)
    generator = torch.Generator().manual_seed(7)
    interior = sample_interior_points(geometry, 100, generator=generator)
    bottom, top = sample_wall_points(geometry, 32, generator=generator)
    inlet = sample_inlet_points(geometry, 11)
    outlet = sample_outlet_points(geometry, 11)

    assert interior.shape == (100, 2)
    assert geometry.contains(interior.numpy()).all()
    assert np.allclose(bottom[:, 1].numpy(), geometry.bottom_wall(bottom[:, 0].numpy()))
    assert np.allclose(top[:, 1].numpy(), geometry.top_wall(top[:, 0].numpy()))
    assert np.allclose(inlet[:, 0].numpy(), 0.0)
    assert np.allclose(outlet[:, 0].numpy(), geometry.total_length_m)
