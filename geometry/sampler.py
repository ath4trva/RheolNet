import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyvista as pv
from loader import load_and_sample_geometry, save_to_hdf5


def visualise_points(pts_dict):
    """Quick 3D visualisation to verify sampling"""
    plotter = pv.Plotter()

    colors = {"interior": "blue",
              "wall": "red",
              "inlet": "green"}

    for key, pts in pts_dict.items():
        cloud = pv.PolyData(pts)
        plotter.add_points(cloud, color=colors[key],
                            point_size=2, label=key)

    plotter.add_legend()
    plotter.show(screenshot='geometry_day1.png')


if __name__ == "__main__":
    pts = load_and_sample_geometry("data/raw/aneumo/10651.stl")
    save_to_hdf5(pts, "case_001_collocation.h5")
    visualise_points(pts)
