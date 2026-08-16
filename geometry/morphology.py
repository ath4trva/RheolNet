import numpy as np


class MorphologyExtractor:
    """
    Extracts vessel/aneurysm morphological features from a VesselGeometry.

    METHOD (explicitly documented for precision traceability):
    Centreline is computed via PCA-based cross-sectional slicing:
      1. Compute the principal axis of the point cloud (PCA).
      2. Slice the geometry into N bins perpendicular to that axis.
      3. Each slice's centroid = one centreline point.
      4. Each slice's point spread = local cross-sectional radius/shape.
    This is an approximation, not full VMTK skeletonization - valid for
    roughly tube-like or single-bulge geometries, but will be inaccurate
    on strongly curved or bifurcating vessels where the principal axis is
    not a good proxy for local flow direction. This limitation is
    reported in every result via metadata["method_limitation"].
    """

    def __init__(self, vessel_geometry, n_slices=150):
        self.vg = vessel_geometry
        self.points = vessel_geometry.points
        self.n_slices = n_slices
        self._compute_centreline()

    def _compute_centreline(self):
        centred = self.points - self.points.mean(axis=0)
        cov = np.cov(centred.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, np.argmax(eigvals)]

        proj = centred @ principal_axis
        bin_edges = np.linspace(proj.min(), proj.max(), self.n_slices + 1)

        centreline_pts = []
        slice_radii = []
        slice_point_sets = []

        for i in range(self.n_slices):
            mask = (proj >= bin_edges[i]) & (proj < bin_edges[i + 1])
            slice_pts = self.points[mask]
            if len(slice_pts) < 4:
                continue
            centroid = slice_pts.mean(axis=0)
            dists = np.linalg.norm(slice_pts - centroid, axis=1)
            centreline_pts.append(centroid)
            slice_radii.append(dists.mean())
            slice_point_sets.append(slice_pts)

        self.centreline = np.array(centreline_pts)
        self.radii = np.array(slice_radii)
        self.slice_point_sets = slice_point_sets

        if len(self.centreline) < 3:
            raise ValueError(
                f"[{self.vg.case_id}] Centreline extraction failed - "
                f"fewer than 3 valid slices. Geometry may be too small "
                f"or degenerate for this slicing method."
            )

    def diameters(self):
        return 2 * self.radii

    def parent_vessel_diameter(self):
        """Median diameter away from any bulge - robust to a single
        aneurysm sac skewing the mean."""
        return np.median(self.diameters())

    def detect_sac(self, threshold_ratio=1.5):
        """
        Flags slices where local diameter exceeds threshold_ratio times
        the parent vessel diameter - a simple bulge detector, not a
        clinically validated segmentation.
        """
        parent_d = self.parent_vessel_diameter()
        diam = self.diameters()
        sac_mask = diam > (threshold_ratio * parent_d)
        return sac_mask, parent_d

    def sac_height(self):
        """
        Sac height = distance from the true anatomical neck point to the
        dome (point of MAXIMUM diameter within the sac) - the standard
        clinical definition.
        """
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        neck_idx = self._find_neck_index()
        if neck_idx is None:
            return None

        sac_indices = np.where(sac_mask)[0]
        dome_idx = sac_indices[np.argmax(self.diameters()[sac_indices])]

        return np.linalg.norm(self.centreline[dome_idx] - self.centreline[neck_idx])

    def _find_neck_index(self, baseline_tolerance=1.15):
        """
        Walks backward from the sac onset until the diameter returns to
        within baseline_tolerance x parent_vessel_diameter - i.e. finds
        where the vessel is actually back to its normal caliber, rather
        than checking a fixed number of slices (which is guesswork and
        can land inside a still-widening shoulder region on aneurysms
        with a gradual, non-abrupt onset).
        """
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        first_sac_idx = np.argmax(sac_mask)
        parent_d = self.parent_vessel_diameter()
        diam = self.diameters()

        idx = first_sac_idx
        while idx > 0 and diam[idx - 1] > baseline_tolerance * parent_d:
            idx -= 1
        return max(0, idx - 1)

    def neck_width(self):
        """Diameter at the true anatomical neck (see _find_neck_index)."""
        neck_idx = self._find_neck_index()
        if neck_idx is None:
            return None
        return self.diameters()[neck_idx]

    def max_sac_diameter(self):
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        return self.diameters()[sac_mask].max()

    def aspect_ratio(self):
        """AR = sac_height / neck_width. Elevated risk reported at AR > 1.6
        in the literature this project cites."""
        h, n = self.sac_height(), self.neck_width()
        if h is None or n is None or n == 0:
            return None
        return h / n

    def size_ratio(self):
        """SR = max_sac_diameter / parent_vessel_diameter."""
        max_d = self.max_sac_diameter()
        parent_d = self.parent_vessel_diameter()
        if max_d is None or parent_d == 0:
            return None
        return max_d / parent_d

    def bottleneck_factor(self):
        """max_width / neck_width."""
        max_d = self.max_sac_diameter()
        neck = self.neck_width()
        if max_d is None or neck is None or neck == 0:
            return None
        return max_d / neck

    def undulation_index(self):
        """Ratio of actual centreline path length to straight-line
        distance between its endpoints - measures how 'wavy' the vessel is."""
        path_len = np.sum(np.linalg.norm(np.diff(self.centreline, axis=0), axis=1))
        straight_dist = np.linalg.norm(self.centreline[-1] - self.centreline[0])
        if straight_dist < 1e-10:
            return None
        return path_len / straight_dist

    def ellipticity_index(self, slice_points):
        """For one cross-sectional slice: ratio describing deviation from
        a perfect circle, via 2D PCA of the slice's in-plane spread."""
        centred = slice_points - slice_points.mean(axis=0)
        cov = np.cov(centred.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[0] < 1e-10:
            return None
        return 1.0 - (eigvals[-1] / eigvals[0])

    def nonsphericity_index(self):
        """1 - (18*pi)^(1/3) * V^(2/3) / A, computed on the sac point subset
        via convex hull, comparing to a perfect sphere (NSI=0)."""
        from scipy.spatial import ConvexHull
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        sac_slice_pts = np.vstack([self.slice_point_sets[i] for i in np.where(sac_mask)[0]])
        if len(sac_slice_pts) < 4:
            return None
        hull = ConvexHull(sac_slice_pts)
        volume = hull.volume
        area = hull.area
        if area < 1e-10:
            return None
        nsi = 1.0 - (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / area
        return nsi

    def curvature_profile(self):
        """Discrete curvature at each centreline point via finite
        differences of the (already-smoothed-by-slicing) centreline."""
        d1 = np.gradient(self.centreline, axis=0)
        d2 = np.gradient(d1, axis=0)
        cross = np.cross(d1, d2)
        num = np.linalg.norm(cross, axis=1)
        denom = np.linalg.norm(d1, axis=1) ** 3
        denom[denom < 1e-10] = 1e-10
        return num / denom

    def extract_all(self):
        """Returns the full feature set for this vessel geometry."""
        sac_mask, parent_d = self.detect_sac()
        ellipticities = [self.ellipticity_index(sp) for sp in self.slice_point_sets]
        ellipticities = [e for e in ellipticities if e is not None]

        return {
            "case_id": self.vg.case_id,
            "source": self.vg.source,
            "parent_vessel_diameter": parent_d,
            "has_detected_sac": bool(sac_mask.any()),
            "sac_height": self.sac_height(),
            "neck_width": self.neck_width(),
            "max_sac_diameter": self.max_sac_diameter(),
            "aspect_ratio": self.aspect_ratio(),
            "size_ratio": self.size_ratio(),
            "bottleneck_factor": self.bottleneck_factor(),
            "undulation_index": self.undulation_index(),
            "mean_ellipticity": float(np.mean(ellipticities)) if ellipticities else None,
            "nonsphericity_index": self.nonsphericity_index(),
            "curvature_mean": float(np.mean(self.curvature_profile())),
            "curvature_max": float(np.max(self.curvature_profile())),
            "method": "PCA_slicing_approximation",
            "method_limitation": (
                "Centreline via PCA cross-sectional slicing - approximation, "
                "not full VMTK skeletonization. May be inaccurate on strongly "
                "curved or bifurcating geometries."
            )
        }
