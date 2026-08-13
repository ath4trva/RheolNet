import numpy as np
import torch
from torch.utils.data import Dataset


class VesselFlowDataset(Dataset):
    """
    Wraps a collection of VesselGeometry + collocation point sets into a
    PyTorch-compatible dataset for PINN training and meta-learning (Reptile).
    """

    def __init__(self, geometries, collocation_sets, cfd_data=None):
        """
        geometries       : list of VesselGeometry objects
        collocation_sets : list of dicts from sample_collocation_points(),
                            same order/index as geometries
        cfd_data         : optional list of dicts with CFD supervision
                            (u, v, w, p, viscosity tensors), same order,
                            or None per-entry if unavailable
        """
        assert len(geometries) == len(collocation_sets)
        self.geometries = geometries
        self.collocation_sets = collocation_sets
        self.cfd_data = cfd_data or [None] * len(geometries)

    def __len__(self):
        return len(self.geometries)

    def __getitem__(self, idx):
        return self.sample_geometry(idx)

    def sample_geometry(self, idx):
        """
        Returns one geometry's full data package: collocation points as
        tensors, plus CFD supervision if available.
        """
        coll = self.collocation_sets[idx]
        item = {
            "case_id": self.geometries[idx].case_id,
            "source": self.geometries[idx].source,
            "interior": torch.tensor(coll["interior"], dtype=torch.float32),
            "wall": torch.tensor(coll["wall"], dtype=torch.float32),
            "inlet": torch.tensor(coll["inlet"], dtype=torch.float32),
            "outlet": torch.tensor(coll["outlet"], dtype=torch.float32),
        }
        if self.cfd_data[idx] is not None:
            item["cfd"] = {k: torch.tensor(v, dtype=torch.float32)
                            for k, v in self.cfd_data[idx].items()}
        return item

    def get_batch(self, batch_size, point_type="interior"):
        """
        Returns a batch of point_type points, drawn from randomly selected
        geometries in the dataset. Used for Reptile-style meta-training
        across multiple vessel shapes.
        """
        idx = np.random.choice(len(self.geometries), size=min(batch_size, len(self.geometries)),
                                replace=len(self.geometries) < batch_size)
        batch = [self.sample_geometry(i) for i in idx]
        return batch
