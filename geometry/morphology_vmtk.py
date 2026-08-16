import numpy as np
import pyvista as pv


class VMTKMorphologyExtractor:
    """
    Extracts vessel/aneurysm morphology using VMTK's dense centerline
    extraction (vmtkCenterlines + resampling) - the correct tool for a
    fine-grained radius profile, as opposed to vmtkNetworkExtraction
    which only returns sparse topological branch-graph nodes (unsuitable
    for detecting a localized bulge between two distant nodes).
    """

    def __init__(self, vessel_geometry, resampling_step=0.2):
        self.vg = vessel_geometry
        self._extract_centerline(resampling_step)

    def _extract_centerline(self, resampling_step):
        from vmtk import vmtkscripts

        if self.vg.faces is None or len(self.vg.faces) == 0:
            raise ValueError(
                f"[{self.vg.case_id}] VMTK centerline extraction requires a "
                f"surface mesh with faces."
            )

        n_faces = len(self.vg.faces)
        face_array = np.hstack([
            np.full((n_faces, 1), 3), self.vg.faces
        ]).astype(np.int64)
        surface = pv.PolyData(self.vg.points, face_array)

        # Step 1: find automatic seed endpoints via network extraction
        # (this IS the correct use of network extraction - just for
        # endpoint discovery, not for the radius profile itself)
        network = vmtkscripts.vmtkNetworkExtraction()
        network.Surface = surface
        network.Execute()
        net_output = pv.wrap(network.Network)
        endpoints = np.array(net_output.points, dtype=np.float64)

        if len(endpoints) < 2:
            raise ValueError(f"[{self.vg.case_id}] Could not find at least 2 endpoints for centerline seeding")

        # IMPORTANT: using only the two points furthest apart as
        # source/target misses lateral outpouchings (e.g. an aneurysm
        # sac branching off the side of the main vessel) - the
        # centerline would simply route past it. Instead, compute
        # centerlines from ONE source to EVERY other endpoint, so every
        # branch (including a side-branch sac) is captured.
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(endpoints))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        source_pt = endpoints[i]

        target_pts = []
        for k in range(len(endpoints)):
            if k != i:
                target_pts.extend(list(endpoints[k]))

        # Step 2: compute dense centerlines with real MISR, to ALL branches
        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = surface
        centerlines.SeedSelectorName = "pointlist"
        centerlines.SourcePoints = list(source_pt)
        centerlines.TargetPoints = target_pts
        centerlines.Execute()

        # Step 3: resample to a fine, even spacing
        resampler = vmtkscripts.vmtkCenterlineResampling()
        resampler.Centerlines = centerlines.Centerlines
        resampler.Length = resampling_step
        resampler.Execute()

        cl_output = pv.wrap(resampler.Centerlines)
        self.centreline = np.array(cl_output.points, dtype=np.float64)

        if "MaximumInscribedSphereRadius" in cl_output.point_data:
            self.radii = np.array(cl_output.point_data["MaximumInscribedSphereRadius"])
        else:
            raise ValueError(
                f"[{self.vg.case_id}] No MISR field on resampled centerline - "
                f"available fields: {list(cl_output.point_data.keys())}"
            )

        # Drop any degenerate points (zero/negative radius - junction artifacts)
        valid = self.radii > 1e-6
        self.centreline = self.centreline[valid]
        self.radii = self.radii[valid]

        if len(self.centreline) < 10:
            raise ValueError(
                f"[{self.vg.case_id}] Only {len(self.centreline)} valid centerline "
                f"points after filtering - too sparse for reliable morphology"
            )

    def diameters(self):
        return 2 * self.radii

    def parent_vessel_diameter(self, _exclude_mask=None):
        """
        Median diameter of the vessel. When _exclude_mask is given
        (centreline indices belonging to the detected sac), those are
        excluded so the baseline reflects only NORMAL vessel calibre -
        avoiding circularity where the bulge itself inflates the median
        used to detect it.
        """
        diam = self.diameters()
        if _exclude_mask is not None and (~_exclude_mask).sum() >= 5:
            return np.median(diam[~_exclude_mask])
        return np.median(diam)

    def _surface_distance_to_centreline(self):
        """
        For each surface point, distance to the NEAREST centreline point
        and that point's local radius. A saccular aneurysm is a blind
        pouch with no second open end, so it can never appear in the
        centreline's own radius profile (vmtkCenterlines only traces
        paths between open orifices) - it can only be detected by
        surface points sitting much farther out than the local tube
        radius predicts. This is the standard VMTK approach for
        aneurysm sac isolation.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(self.centreline)
        dist, idx = tree.query(self.vg.points)
        local_radius = self.radii[idx]
        return dist, local_radius, idx

    def detect_sac(self, threshold_ratio=1.4):
        """
        Two-pass detection to avoid baseline circularity:
        Pass 1 - use the naive (all-points) median to get a first-guess
                 sac mask.
        Pass 2 - recompute the parent diameter EXCLUDING that first-guess
                 sac region, then re-flag the sac against this cleaner
                 baseline. This prevents the bulge from inflating the
                 very threshold used to detect it.
        """
        dist, local_radius, idx = self._surface_distance_to_centreline()

        # Pass 1: naive baseline
        parent_d_pass1 = self.parent_vessel_diameter()
        bulge_mask_surface = dist > (threshold_ratio * local_radius)
        sac_mask_pass1 = np.zeros(len(self.centreline), dtype=bool)
        if bulge_mask_surface.any():
            sac_mask_pass1[np.unique(idx[bulge_mask_surface])] = True

        # Pass 2: refine baseline excluding pass-1 sac region
        parent_d = self.parent_vessel_diameter(_exclude_mask=sac_mask_pass1)
        bulge_mask_surface = dist > (threshold_ratio * local_radius / parent_d_pass1 * parent_d) \
            if parent_d_pass1 > 0 else bulge_mask_surface
        # Re-flag using the corrected local radius scale
        bulge_mask_surface = dist > (threshold_ratio * (parent_d / 2.0))

        sac_mask = np.zeros(len(self.centreline), dtype=bool)
        if bulge_mask_surface.any():
            sac_mask[np.unique(idx[bulge_mask_surface])] = True

        self._surface_dist_cache = dist
        self._surface_idx_cache = idx
        self._surface_bulge_mask_cache = bulge_mask_surface
        self._sac_mask_for_baseline = sac_mask

        return sac_mask, parent_d

    def _find_neck_index(self, baseline_tolerance=1.1):
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        sac_indices = np.where(sac_mask)[0]
        first_sac_idx = sac_indices[0]
        parent_d = self.parent_vessel_diameter()
        diam = self.diameters()
        idx = first_sac_idx
        while idx > 0 and diam[idx - 1] > baseline_tolerance * parent_d:
            idx -= 1
        return max(0, idx - 1)

    def neck_width(self):
        idx = self._find_neck_index()
        return self.diameters()[idx] if idx is not None else None

    def sac_height(self, percentile=98):
        """
        Height = distance from the neck centreline point to the dome.
        The dome is located using the given PERCENTILE of bulge
        distances (default 98th), not a raw argmax - a single noisy or
        degenerate mesh vertex should not be able to single-handedly
        define a clinical measurement. The dome point used is the one
        closest to that percentile value, for a physically real point.
        """
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        neck_idx = self._find_neck_index()
        if neck_idx is None:
            return None

        dist = self._surface_dist_cache
        bulge_mask = self._surface_bulge_mask_cache
        if not bulge_mask.any():
            return None
        bulge_dists = dist[bulge_mask]
        bulge_indices = np.where(bulge_mask)[0]
        target_val = np.percentile(bulge_dists, percentile)
        closest = bulge_indices[np.argmin(np.abs(bulge_dists - target_val))]
        dome_point = self.vg.points[closest]

        return np.linalg.norm(dome_point - self.centreline[neck_idx])

    def max_sac_diameter(self, percentile=98):
        """
        Max sac diameter derived from the given PERCENTILE (default 98th)
        of surface bulge distances, not the raw maximum - robust to a
        single outlier/degenerate mesh vertex inflating the result.
        """
        sac_mask, _ = self.detect_sac()
        if not sac_mask.any():
            return None
        dist = self._surface_dist_cache
        bulge_mask = self._surface_bulge_mask_cache
        if not bulge_mask.any():
            return None
        return 2 * np.percentile(dist[bulge_mask], percentile)

    def aspect_ratio(self):
        h, n = self.sac_height(), self.neck_width()
        return h / n if (h is not None and n) else None

    def size_ratio(self):
        m, p = self.max_sac_diameter(), self.parent_vessel_diameter()
        return m / p if (m is not None and p) else None

    def bottleneck_factor(self):
        m, n = self.max_sac_diameter(), self.neck_width()
        return m / n if (m is not None and n) else None

    def extract_all(self):
        sac_mask, parent_d = self.detect_sac()
        return {
            "case_id": self.vg.case_id,
            "source": self.vg.source,
            "n_centreline_points": len(self.centreline),
            "parent_vessel_diameter": float(parent_d),
            "has_detected_sac": bool(sac_mask.any()),
            "sac_height": self.sac_height(),
            "neck_width": self.neck_width(),
            "max_sac_diameter": self.max_sac_diameter(),
            "aspect_ratio": self.aspect_ratio(),
            "size_ratio": self.size_ratio(),
            "bottleneck_factor": self.bottleneck_factor(),
            "method": "VMTK_dense_centerline_extraction",
            "method_note": "Real MISR-based radius profile via vmtkCenterlines + resampling."
        }
