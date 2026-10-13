"""CFD comparison metrics for Task 1.2 planar-stenosis predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


def _one_dimensional(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values.")
    return values


def relative_l2_error(predicted: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> float:
    predicted = _one_dimensional(predicted, "predicted")
    reference = _one_dimensional(reference, "reference")
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")
    return float(np.linalg.norm(predicted - reference) / max(np.linalg.norm(reference), eps))


def rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    predicted = _one_dimensional(predicted, "predicted")
    reference = _one_dimensional(reference, "reference")
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


@dataclass(frozen=True)
class FieldErrorMetrics:
    relative_l2: float
    rmse: float
    max_absolute_error: float


@dataclass(frozen=True)
class PlanarStenosisValidationResult:
    u: FieldErrorMetrics
    v: FieldErrorMetrics
    pressure: FieldErrorMetrics
    pressure_offset_pa: float

    def as_dict(self) -> dict[str, float]:
        flattened: dict[str, float] = {"pressure_offset_pa": self.pressure_offset_pa}
        for field_name in ("u", "v", "pressure"):
            for metric_name, value in asdict(getattr(self, field_name)).items():
                flattened[f"{field_name}_{metric_name}"] = value
        return flattened


def _field_metrics(predicted: np.ndarray, reference: np.ndarray) -> FieldErrorMetrics:
    predicted = _one_dimensional(predicted, "predicted")
    reference = _one_dimensional(reference, "reference")
    return FieldErrorMetrics(
        relative_l2=relative_l2_error(predicted, reference),
        rmse=rmse(predicted, reference),
        max_absolute_error=float(np.max(np.abs(predicted - reference))),
    )


def align_pressure_gauge(
    predicted_pressure_pa: np.ndarray,
    reference_pressure_pa: np.ndarray,
    outlet_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Align predicted pressure to CFD pressure using the outlet reference.

    Pressure is defined only up to an additive constant.  If cell-centre outlet
    indices are supplied, their means determine the alignment; otherwise all
    supplied points are used.
    """
    predicted = _one_dimensional(predicted_pressure_pa, "predicted_pressure_pa")
    reference = _one_dimensional(reference_pressure_pa, "reference_pressure_pa")
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference pressure arrays must have the same shape.")
    if outlet_mask is None:
        outlet_mask = np.ones(predicted.shape, dtype=bool)
    outlet_mask = np.asarray(outlet_mask, dtype=bool).reshape(-1)
    if outlet_mask.shape != predicted.shape or not outlet_mask.any():
        raise ValueError("outlet_mask must select at least one pressure value.")
    offset = float(np.mean(reference[outlet_mask]) - np.mean(predicted[outlet_mask]))
    return predicted + offset, offset


def evaluate_planar_stenosis_field(
    predicted_uvp: np.ndarray,
    reference_uvp: np.ndarray,
    *,
    outlet_mask: np.ndarray | None = None,
    pressure_gauge: Literal["outlet_mean", "none"] = "outlet_mean",
) -> PlanarStenosisValidationResult:
    """Compute velocity and pressure errors at exactly the CFD coordinates."""
    predicted = np.asarray(predicted_uvp, dtype=np.float64)
    reference = np.asarray(reference_uvp, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or predicted.shape != reference.shape:
        raise ValueError("predicted_uvp and reference_uvp must both have shape (n, 3).")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("Predictions and reference values must be finite.")
    if pressure_gauge == "outlet_mean":
        aligned_pressure, offset = align_pressure_gauge(predicted[:, 2], reference[:, 2], outlet_mask)
    elif pressure_gauge == "none":
        aligned_pressure, offset = predicted[:, 2], 0.0
    else:
        raise ValueError(f"Unsupported pressure_gauge: {pressure_gauge}.")
    return PlanarStenosisValidationResult(
        u=_field_metrics(predicted[:, 0], reference[:, 0]),
        v=_field_metrics(predicted[:, 1], reference[:, 1]),
        pressure=_field_metrics(aligned_pressure, reference[:, 2]),
        pressure_offset_pa=offset,
    )
