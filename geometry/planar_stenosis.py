"""Analytic geometry for the Task 1.2 planar, centred stenosis cases.

The geometry is deliberately separate from the 3D patient-geometry pipeline.
Coordinates are physical SI coordinates in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class PlanarStenosisGeometry:
    """A symmetric, piecewise-linear planar stenosis.

    The default breakpoints reproduce Kiera's OpenFOAM cases:
    healthy channel -> converging section -> constant throat -> diffuser ->
    healthy channel.
    """

    total_length_m: float = 0.092
    healthy_height_m: float = 0.006
    throat_height_m: float = 0.0036
    converging_start_m: float = 0.030
    throat_start_m: float = 0.042
    throat_end_m: float = 0.050
    diverging_end_m: float = 0.062

    def __post_init__(self) -> None:
        if self.total_length_m <= 0 or self.healthy_height_m <= 0:
            raise ValueError("Channel length and healthy height must be positive.")
        if not 0 < self.throat_height_m <= self.healthy_height_m:
            raise ValueError("Throat height must be in (0, healthy_height].")
        if not (
            0 <= self.converging_start_m < self.throat_start_m
            <= self.throat_end_m < self.diverging_end_m <= self.total_length_m
        ):
            raise ValueError("Stenosis breakpoints must be ordered inside the channel.")

    @property
    def healthy_half_height_m(self) -> float:
        return 0.5 * self.healthy_height_m

    @property
    def throat_half_height_m(self) -> float:
        return 0.5 * self.throat_height_m

    @property
    def severity_percent(self) -> float:
        return 100.0 * (1.0 - self.throat_height_m / self.healthy_height_m)

    @classmethod
    def from_severity_percent(
        cls,
        severity_percent: float,
        **kwargs: Any,
    ) -> "PlanarStenosisGeometry":
        healthy_height_m = float(kwargs.get("healthy_height_m", 0.006))
        if not 0 <= severity_percent < 100:
            raise ValueError("Stenosis severity must be in [0, 100).")
        kwargs["throat_height_m"] = healthy_height_m * (1.0 - severity_percent / 100.0)
        return cls(**kwargs)

    @classmethod
    def from_case_metadata(cls, metadata: dict[str, Any]) -> "PlanarStenosisGeometry":
        """Build geometry from Kiera's metadata and the documented breakpoints."""
        geometry = metadata["geometry"]
        return cls(
            total_length_m=float(geometry["total_length_m"]),
            healthy_height_m=float(geometry["healthy_channel_height_m"]),
            throat_height_m=float(metadata["throat_height_m"]),
        )

    def half_height(self, x_m: ArrayLike) -> ArrayLike:
        """Return the positive wall coordinate ``y_top(x)`` in metres."""
        x = np.asarray(x_m, dtype=np.float64)
        h0 = self.healthy_half_height_m
        ht = self.throat_half_height_m

        h = np.full_like(x, h0, dtype=np.float64)
        converging = (x >= self.converging_start_m) & (x < self.throat_start_m)
        throat = (x >= self.throat_start_m) & (x <= self.throat_end_m)
        diverging = (x > self.throat_end_m) & (x <= self.diverging_end_m)

        h[converging] = h0 + (ht - h0) * (
            (x[converging] - self.converging_start_m)
            / (self.throat_start_m - self.converging_start_m)
        )
        h[throat] = ht
        h[diverging] = ht + (h0 - ht) * (
            (x[diverging] - self.throat_end_m)
            / (self.diverging_end_m - self.throat_end_m)
        )

        if np.isscalar(x_m):
            return float(h.item())
        return h

    def top_wall(self, x_m: ArrayLike) -> ArrayLike:
        return self.half_height(x_m)

    def bottom_wall(self, x_m: ArrayLike) -> ArrayLike:
        return -np.asarray(self.half_height(x_m))

    def contains(self, xy_m: np.ndarray, tolerance_m: float = 1e-12) -> np.ndarray:
        """Return a Boolean mask for points in the fluid domain, including walls."""
        xy = np.asarray(xy_m, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy_m must have shape (n, 2).")
        x, y = xy[:, 0], xy[:, 1]
        h = self.half_height(x)
        return (
            (x >= -tolerance_m)
            & (x <= self.total_length_m + tolerance_m)
            & (np.abs(y) <= h + tolerance_m)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "total_length_m": self.total_length_m,
            "healthy_height_m": self.healthy_height_m,
            "throat_height_m": self.throat_height_m,
            "severity_percent": self.severity_percent,
            "converging_start_m": self.converging_start_m,
            "throat_start_m": self.throat_start_m,
            "throat_end_m": self.throat_end_m,
            "diverging_end_m": self.diverging_end_m,
        }
