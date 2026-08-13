import pyvista as pv
import numpy as np
import h5py


def load_and_sample_geometry(stl_path,
                              n_interior=50000,
                              n_surface=10000,
                              n_inlet=500):
    """
    Load artery surface mesh and sample collocation points.
    Returns dict of (x,y,z) arrays for PINN training.
    """
    mesh = pv.read(stl_path)
    print(f"Mesh loaded: {mesh.n_points} points, "
          f"{mesh.n_cells} cells")

    surface_pts = mesh.points[
        np.random.choice(mesh.n_points, n_surface, replace=False)
    ]

    volume = mesh.delaunay_3d()
    interior = volume.extract_cells(
        np.arange(volume.n_cells)
    ).cell_centers().points

    idx = np.random.choice(len(interior),
                            min(n_interior, len(interior)),
                            replace=False)
    interior_pts = interior[idx]

    bounds = mesh.bounds
    inlet_mask = surface_pts[:, 0] < (bounds[0] + 0.5)
    inlet_pts = surface_pts[inlet_mask][:n_inlet]

    return {
        "interior": interior_pts,
        "wall": surface_pts,
        "inlet": inlet_pts
    }


def save_to_hdf5(pts_dict, save_path):
    """Save collocation points to HDF5 for training"""
    with h5py.File(save_path, 'w') as f:
        for key, pts in pts_dict.items():
            f.create_dataset(key, data=pts)
    print(f"Saved to {save_path}")


import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_aneumo_cfd(vtu_path):
    """
    Load Aneumo precomputed CFD solution.
    Returns points, velocity, pressure as numpy arrays.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    data = reader.GetOutput()

    points = vtk_to_numpy(data.GetPoints().GetData())

    velocity = vtk_to_numpy(
        data.GetPointData().GetArray('U')
    )
    pressure = vtk_to_numpy(
        data.GetPointData().GetArray('p')
    )

    print(f"Points: {points.shape}")
    print(f"Velocity range: {velocity.min():.4f} "
          f"to {velocity.max():.4f} m/s")
    print(f"Pressure range: {pressure.min():.2f} "
          f"to {pressure.max():.2f} Pa")

    return points, velocity, pressure


from vessel_geometry import VesselGeometry


def load_aneumo_as_vessel_geometry(stl_path, case_id=None):
    """
    Wraps STL loading into the unified VesselGeometry container.
    Points and normals are explicitly cast to float64.
    """
    mesh = pv.read(stl_path)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    try:
        normals = mesh.point_normals.astype(np.float64)
    except Exception:
        normals = None

    return VesselGeometry(
        points=mesh.points.astype(np.float64),
        faces=faces,
        normals=normals,
        source="aneumo",
        case_id=case_id or stl_path,
        metadata={"n_cells": mesh.n_cells}
    )


def load_vmr_as_vessel_geometry(vtp_path, case_id=None):
    """
    Loads a VMR surface mesh (.vtp) into the unified VesselGeometry container.
    Points are explicitly cast to float64 since VTK internals may store
    single-precision floats, which is not acceptable for clinical geometry.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    data = reader.GetOutput()

    points = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64)

    polys = data.GetPolys()
    polys.InitTraversal()
    faces = []
    n_non_triangle = 0
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            faces.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
        else:
            n_non_triangle += 1
    faces = np.array(faces)

    if n_non_triangle > 0:
        print(f"WARNING [{case_id or vtp_path}]: {n_non_triangle} non-triangular "
              f"cells were present and excluded from the face list")

    return VesselGeometry(
        points=points,
        faces=faces,
        normals=None,
        source="vmr",
        case_id=case_id or vtp_path,
        metadata={"n_cells": data.GetNumberOfCells(), "n_non_triangle_dropped": n_non_triangle}
    )


def load_aneurisk_as_vessel_geometry(txt_path, case_id=None, n_circle_pts=12):
    """
    Loads an AneuRisk65 centreline+radius file and reconstructs an
    approximate tubular surface as a VesselGeometry.

    IMPORTANT — this is a reconstruction, not a real vessel surface scan:
    AneuRisk65 provides only a 1D centreline + radius profile (no true
    3D surface). A circular cross-section is swept along the centreline
    as the closest reasonable approximation. This is flagged explicitly
    via metadata["reconstructed_tube"]=True and geometry.is_approximated().

    Precision notes (verified against Sangalli et al., AneuRisk65 papers):
    - Uses the FKS free-knot-spline SMOOTHED centreline (X0_FKS etc.),
      not the raw noisy digitized points (X0_obs etc.), since the FKS
      version has measurement noise removed via validated spline fitting.
    - Uses the FKS first-derivative columns (X1_FKS etc.) directly as the
      tangent vector, rather than re-differentiating with finite
      differences, which would reintroduce noise and lose precision
      already resolved by the dataset's own spline fitting.
    - Uses the FKS second-derivative columns (X2_FKS etc.) to build the
      Frenet normal, producing a continuous, non-twisting frame along
      the whole centreline — avoiding the artifact of picking an
      arbitrary reference vector, which can cause the swept circle to
      flip or twist wherever the tangent direction is near-parallel to
      that arbitrary choice.
    - MISR (the radius) is documented as RAW/unsmoothed in the source
      data. This is a genuine dataset limitation, not something this
      loader can correct — it is preserved as-is and should be treated
      with lower confidence than the smoothed centreline.
    """
    data = np.loadtxt(txt_path, skiprows=1)

    radius = data[:, 1]                  # MISR (raw, unsmoothed - dataset limitation)
    centreline = data[:, 5:8]            # X0_FKS, Y0_FKS, Z0_FKS (smoothed)
    tangent_raw = data[:, 9:12]          # X1_FKS, Y1_FKS, Z1_FKS (1st derivative)
    normal_raw = data[:, 13:16]          # X2_FKS, Y2_FKS, Z2_FKS (2nd derivative)

    n_pts = centreline.shape[0]
    surface_points = []

    for i in range(n_pts):
        tangent = tangent_raw[i]
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-10:
            # Degenerate tangent at this point - skip building a ring here
            continue
        tangent = tangent / t_norm

        # Build normal1 from the 2nd-derivative (curvature) vector, projected
        # orthogonal to the tangent, for a continuous Frenet-style frame.
        n_raw = normal_raw[i]
        n_raw = n_raw - np.dot(n_raw, tangent) * tangent  # orthogonalize
        n_norm = np.linalg.norm(n_raw)

        if n_norm < 1e-10:
            # Curvature ~0 here (near-straight segment) - fall back to a
            # stable arbitrary perpendicular, only for this local point.
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(arbitrary, tangent)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0])
            normal1 = np.cross(tangent, arbitrary)
            normal1 = normal1 / np.linalg.norm(normal1)
        else:
            normal1 = n_raw / n_norm

        normal2 = np.cross(tangent, normal1)

        for theta in np.linspace(0, 2*np.pi, n_circle_pts, endpoint=False):
            offset = radius[i] * (np.cos(theta) * normal1 + np.sin(theta) * normal2)
            surface_points.append(centreline[i] + offset)

    surface_points = np.array(surface_points, dtype=np.float64)

    return VesselGeometry(
        points=surface_points,
        faces=None,
        normals=None,
        source="aneurisk65",
        case_id=case_id or txt_path,
        metadata={
            "n_centreline_pts": n_pts,
            "reconstructed_tube": True,
            "centreline_source": "FKS_smoothed",
            "radius_source": "MISR_raw_unsmoothed",
            "warning": "Synthetic circular cross-section - not a true surface scan"
        }
    )
