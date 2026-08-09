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
    """
    mesh = pv.read(stl_path)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    try:
        normals = mesh.point_normals
    except Exception:
        normals = None

    return VesselGeometry(
        points=mesh.points,
        faces=faces,
        normals=normals,
        source="aneumo",
        case_id=case_id or stl_path,
        metadata={"n_cells": mesh.n_cells}
    )


def load_vmr_as_vessel_geometry(vtp_path, case_id=None):
    """
    Loads a VMR surface mesh (.vtp) into the unified VesselGeometry container.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    data = reader.GetOutput()

    points = vtk_to_numpy(data.GetPoints().GetData())

    polys = data.GetPolys()
    polys.InitTraversal()
    faces = []
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            faces.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    faces = np.array(faces)

    return VesselGeometry(
        points=points,
        faces=faces,
        normals=None,
        source="vmr",
        case_id=case_id or vtp_path,
        metadata={"n_cells": data.GetNumberOfCells()}
    )
