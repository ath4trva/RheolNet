import numpy as np
import pyvista as pv
import h5py


def sample_collocation_points(vessel_geometry, n_interior=50000, n_wall=10000,
                                n_inlet=500, n_outlet=500, n_time=10, t_max=1.0):
    """
    Samples interior, wall, inlet, outlet points from a VesselGeometry,
    replicated across n_time cardiac-cycle time samples.
    """
    points = vessel_geometry.points
    bounds_min = vessel_geometry.bounds["min"]
    bounds_max = vessel_geometry.bounds["max"]

    # Wall points: sample directly from surface/point cloud
    wall_idx = np.random.choice(len(points), min(n_wall, len(points)), replace=False)
    wall_pts = points[wall_idx]

    # Interior points: use PyVista delaunay if faces exist, else random-in-bbox fallback
    if vessel_geometry.faces is not None and len(vessel_geometry.faces) > 0:
        mesh = pv.PolyData(points, np.hstack([
            np.full((len(vessel_geometry.faces), 1), 3), vessel_geometry.faces
        ]).astype(np.int64))
        volume = mesh.delaunay_3d()
        interior_candidates = volume.extract_cells(
            np.arange(volume.n_cells)
        ).cell_centers().points
    else:
        # AneuRisk65-style point cloud (no faces = no enclosed volume to
        # sample). Bounding-box random sampling would place most points
        # outside the thin tube, so instead we jitter INWARD from the
        # already-correct tube surface points along their inward normal
        # (toward the local centreline), which stays physically inside
        # the reconstructed vessel.
        centre = points.mean(axis=0)
        directions = centre - points
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions = directions / norms
        jitter_fracs = np.random.uniform(0.05, 0.95, size=(len(points), 1))
        interior_candidates = points + directions * jitter_fracs * norms

    idx = np.random.choice(len(interior_candidates),
                            min(n_interior, len(interior_candidates)),
                            replace=len(interior_candidates) < n_interior)
    interior_pts = interior_candidates[idx]

    # Inlet/outlet: use the LONGEST bounding-box axis as the flow-direction
    # proxy, not a hardcoded x-axis assumption - a vessel can be oriented
    # along any axis.
    extent = bounds_max - bounds_min
    flow_axis = np.argmax(extent)
    axis_sorted = np.argsort(wall_pts[:, flow_axis])
    inlet_pts = wall_pts[axis_sorted[:n_inlet]]
    outlet_pts = wall_pts[axis_sorted[-n_outlet:]]

    # Expand across time samples
    time_samples = np.linspace(0, t_max, n_time)

    def add_time(pts):
        expanded = []
        for t in time_samples:
            t_col = np.full((len(pts), 1), t)
            expanded.append(np.hstack([pts, t_col]))
        return np.vstack(expanded)

    return {
        "interior": add_time(interior_pts),
        "wall": add_time(wall_pts),
        "inlet": add_time(inlet_pts),
        "outlet": add_time(outlet_pts),
        "source": vessel_geometry.source,
        "case_id": vessel_geometry.case_id,
    }


def save_collocation_to_hdf5(coll_dict, save_path):
    with h5py.File(save_path, 'w') as f:
        for key in ["interior", "wall", "inlet", "outlet"]:
            f.create_dataset(key, data=coll_dict[key])
        f.attrs["source"] = coll_dict["source"]
        f.attrs["case_id"] = str(coll_dict["case_id"])
    print(f"Saved collocation set to {save_path}")
