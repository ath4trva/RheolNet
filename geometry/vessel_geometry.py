import numpy as np


class VesselGeometry:
    """
    Unified container for vessel geometry data across different sources
    (Aneumo, VMR, AneuRisk65). Downstream code (collocation sampler,
    dataset class) should only ever touch this interface, never the
    raw source-specific files directly.
    """

    def __init__(self, points, faces=None, normals=None, source=None,
                 case_id=None, label=None, metadata=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces) if faces is not None else None
        self.normals = np.asarray(normals) if normals is not None else None
        self.source = source
        self.case_id = case_id
        self.label = label
        self.metadata = metadata or {}

        self.bounds = self._compute_bounds()

    def _compute_bounds(self):
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return {"min": mins, "max": maxs}

    def n_points(self):
        return self.points.shape[0]

    def summary(self):
        print(f"VesselGeometry [{self.source}] case={self.case_id}")
        print(f"  points: {self.n_points()}")
        print(f"  bounds: {self.bounds}")
        print(f"  label: {self.label}")
