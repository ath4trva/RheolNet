import numpy as np


class VesselGeometry:
    """
    Unified container for vessel geometry data across different sources
    (Aneumo, VMR, AneuRisk65). Downstream code (collocation sampler,
    dataset class) should only ever touch this interface, never the
    raw source-specific files directly.

    All coordinate data is stored and validated as float64 for
    clinical-grade numerical precision.
    """

    def __init__(self, points, faces=None, normals=None, source=None,
                 case_id=None, label=None, metadata=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces) if faces is not None else None
        self.normals = np.asarray(normals, dtype=np.float64) if normals is not None else None
        self.source = source
        self.case_id = case_id
        self.label = label
        self.metadata = metadata or {}

        self._validate()
        self.bounds = self._compute_bounds()

    def _validate(self):
        if self.points.size == 0:
            raise ValueError(f"[{self.source}/{self.case_id}] Empty point set")
        if not np.isfinite(self.points).all():
            n_bad = np.sum(~np.isfinite(self.points).all(axis=1))
            raise ValueError(
                f"[{self.source}/{self.case_id}] {n_bad} points contain NaN/Inf — "
                f"geometry is corrupted and must not be used downstream"
            )
        if self.points.shape[1] != 3:
            raise ValueError(f"[{self.source}/{self.case_id}] Points must be Nx3, got {self.points.shape}")

    def _compute_bounds(self):
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return {"min": mins, "max": maxs}

    def n_points(self):
        return self.points.shape[0]

    def is_approximated(self):
        """True if this geometry is a reconstruction/approximation rather
        than a direct surface scan (e.g. AneuRisk65's swept-tube reconstruction).
        Downstream code MUST check this before treating geometry as ground truth."""
        return self.metadata.get("reconstructed_tube", False)

    def summary(self):
        print(f"VesselGeometry [{self.source}] case={self.case_id}")
        print(f"  points: {self.n_points()}")
        print(f"  bounds: {self.bounds}")
        print(f"  label: {self.label}")
        if self.is_approximated():
            print(f"  ⚠ APPROXIMATED GEOMETRY — not a direct surface scan (see metadata)")
