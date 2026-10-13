"""Load and validate Kiera's Task 3.2 planar-stenosis CFD deliverables."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geometry.planar_stenosis import PlanarStenosisGeometry

FIELD_COLUMNS = ("x", "y", "u", "v", "p", "viscosity")


@dataclass(frozen=True)
class CFDFieldData:
    """Cell-centre CFD values in physical SI units."""

    x_m: np.ndarray
    y_m: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    pressure_pa: np.ndarray
    viscosity_pa_s: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.x_m,
            self.y_m,
            self.u_m_s,
            self.v_m_s,
            self.pressure_pa,
            self.viscosity_pa_s,
        )
        if not arrays[0].size:
            raise ValueError("CFD field cannot be empty.")
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("Every CFD field column must be one-dimensional.")
        if len({np.asarray(value).size for value in arrays}) != 1:
            raise ValueError("CFD field columns must have equal lengths.")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("CFD field contains NaN or Inf.")

    @property
    def n_points(self) -> int:
        return int(self.x_m.size)

    def coordinates(self) -> np.ndarray:
        return np.column_stack((self.x_m, self.y_m))

    def velocity_pressure(self) -> np.ndarray:
        return np.column_stack((self.u_m_s, self.v_m_s, self.pressure_pa))


@dataclass(frozen=True)
class InletProfileData:
    y_m: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray

    def __post_init__(self) -> None:
        if self.y_m.ndim != 1 or self.u_m_s.ndim != 1 or self.v_m_s.ndim != 1:
            raise ValueError("Inlet profile arrays must be one-dimensional.")
        if not (len(self.y_m) == len(self.u_m_s) == len(self.v_m_s)):
            raise ValueError("Inlet profile arrays must have the same length.")
        if len(self.y_m) < 2 or not np.isfinite(np.column_stack((self.y_m, self.u_m_s, self.v_m_s))).all():
            raise ValueError("Inlet profile is incomplete or non-finite.")

    def interpolate(self, y_query_m: np.ndarray) -> np.ndarray:
        """Interpolate the supplied CFD inlet profile as ``[u, v]`` values."""
        y_query = np.asarray(y_query_m, dtype=np.float64)
        if np.any(y_query < self.y_m[0]) or np.any(y_query > self.y_m[-1]):
            raise ValueError("Requested inlet points lie outside the supplied inlet profile.")
        return np.column_stack((
            np.interp(y_query, self.y_m, self.u_m_s),
            np.interp(y_query, self.y_m, self.v_m_s),
        ))


@dataclass(frozen=True)
class PlanarStenosisCFDCase:
    case_directory: Path
    metadata: dict[str, Any]
    geometry: PlanarStenosisGeometry
    field: CFDFieldData
    inlet_profile: InletProfileData


@dataclass(frozen=True)
class CFDValidationSummary:
    n_field_points: int
    n_unique_coordinates: int
    n_streamwise_locations: int
    min_points_per_streamwise_location: int
    max_points_per_streamwise_location: int
    all_points_inside_geometry: bool
    inlet_profile_points: int


def load_full_field_csv(path: str | Path) -> CFDFieldData:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELD_COLUMNS:
            raise ValueError(f"{path} must have columns {FIELD_COLUMNS}; got {reader.fieldnames}.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no CFD records.")

    def column(name: str) -> np.ndarray:
        try:
            return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path} has an invalid {name!r} column.") from exc

    return CFDFieldData(
        x_m=column("x"),
        y_m=column("y"),
        u_m_s=column("u"),
        v_m_s=column("v"),
        pressure_pa=column("p"),
        viscosity_pa_s=column("viscosity"),
    )


def geometry_from_block_mesh_dict(path: str | Path) -> PlanarStenosisGeometry:
    """Parse the two-dimensional outline encoded in Kiera's blockMeshDict."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"vertices\s*\((.*?)\);", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("blockMeshDict does not contain a vertices section.")
    points = np.asarray(
        [
            (float(x), float(y), float(z))
            for x, y, z in re.findall(
                r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                match.group(1),
            )
        ],
        dtype=np.float64,
    )
    if points.shape[0] < 12:
        raise ValueError("blockMeshDict has too few vertices for a planar stenosis.")

    outline: dict[float, float] = {}
    for x in np.unique(points[:, 0]):
        y_values = np.abs(points[np.isclose(points[:, 0], x), 1])
        outline[float(x)] = float(np.max(y_values))
    x_values = sorted(outline)
    if len(x_values) != 6:
        raise ValueError("Expected six streamwise geometry breakpoints in blockMeshDict.")

    return PlanarStenosisGeometry(
        total_length_m=x_values[-1],
        healthy_height_m=2.0 * outline[x_values[0]],
        throat_height_m=2.0 * outline[x_values[2]],
        converging_start_m=x_values[1],
        throat_start_m=x_values[2],
        throat_end_m=x_values[3],
        diverging_end_m=x_values[4],
    )


def load_openfoam_inlet_profile(
    path: str | Path,
    healthy_half_height_m: float,
) -> InletProfileData:
    """Read the prescribed 120-face inlet velocity list from OpenFOAM ``0/U``.

    The planar mesh has uniformly spaced inlet faces, so OpenFOAM's ordered
    values map to their face-centre y coordinates.  This is the exact imposed
    profile, not a profile reconstructed from interior CFD cells.
    """
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(
        r"value\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find a nonuniform inlet velocity list in {path}.")
    declared_count = int(match.group(1))
    values = np.asarray(
        [
            (float(u), float(v))
            for u, v, _ in re.findall(
                r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                match.group(2),
            )
        ],
        dtype=np.float64,
    )
    if values.shape != (declared_count, 2):
        raise ValueError("OpenFOAM inlet profile count does not match its vector list.")
    dy = 2.0 * healthy_half_height_m / declared_count
    y = np.linspace(
        -healthy_half_height_m + 0.5 * dy,
        healthy_half_height_m - 0.5 * dy,
        declared_count,
        dtype=np.float64,
    )
    return InletProfileData(y_m=y, u_m_s=values[:, 0], v_m_s=values[:, 1])


def validate_planar_stenosis_case(case: PlanarStenosisCFDCase) -> CFDValidationSummary:
    """Check CFD schema, geometry membership, coordinate uniqueness and inlet data."""
    field = case.field
    coordinates = field.coordinates()
    unique_coordinates = np.unique(coordinates, axis=0)
    x_values, counts = np.unique(field.x_m, return_counts=True)
    if unique_coordinates.shape[0] != field.n_points:
        raise ValueError("full_field.csv contains duplicate (x, y) coordinates.")
    if np.any(field.viscosity_pa_s <= 0):
        raise ValueError("CFD viscosity must be positive everywhere.")
    inside = case.geometry.contains(coordinates, tolerance_m=1e-8)
    if not inside.all():
        raise ValueError("full_field.csv contains points outside the stenosis geometry.")

    return CFDValidationSummary(
        n_field_points=field.n_points,
        n_unique_coordinates=int(unique_coordinates.shape[0]),
        n_streamwise_locations=int(x_values.size),
        min_points_per_streamwise_location=int(counts.min()),
        max_points_per_streamwise_location=int(counts.max()),
        all_points_inside_geometry=bool(inside.all()),
        inlet_profile_points=int(case.inlet_profile.y_m.size),
    )


def load_planar_stenosis_case(case_directory: str | Path) -> PlanarStenosisCFDCase:
    """Load one extracted ``P2D_ST*_Re500`` directory from Kiera's archive."""
    case_directory = Path(case_directory)
    metadata_path = case_directory / "case_metadata.json"
    field_path = case_directory / "full_field.csv"
    block_mesh_path = case_directory / "OpenFOAM_case" / "system" / "blockMeshDict"
    inlet_path = case_directory / "OpenFOAM_case" / "0" / "U"
    for path in (metadata_path, field_path, block_mesh_path, inlet_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    geometry = geometry_from_block_mesh_dict(block_mesh_path)
    metadata_geometry = PlanarStenosisGeometry.from_case_metadata(metadata)
    if not np.isclose(geometry.throat_height_m, metadata_geometry.throat_height_m):
        raise ValueError("Metadata and blockMeshDict disagree on throat height.")
    if not np.isclose(geometry.healthy_height_m, metadata_geometry.healthy_height_m):
        raise ValueError("Metadata and blockMeshDict disagree on healthy channel height.")

    case = PlanarStenosisCFDCase(
        case_directory=case_directory,
        metadata=metadata,
        geometry=geometry,
        field=load_full_field_csv(field_path),
        inlet_profile=load_openfoam_inlet_profile(inlet_path, geometry.healthy_half_height_m),
    )
    validate_planar_stenosis_case(case)
    return case
