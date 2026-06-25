import numpy as np
import torch
import torch.nn as nn


class ClusteringLayer(nn.Module):
    """Student's t-distribution clustering layer used by DEC."""

    def __init__(self, hidden_dim, alpha=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.centers = None

    def set_centers(self, centers):
        if isinstance(centers, np.ndarray):
            centers = torch.tensor(centers, dtype=torch.float32)
        self.centers = nn.Parameter(centers.to(next(self.parameters(), centers).device))

    def forward(self, z):
        if self.centers is None:
            raise RuntimeError("Cluster centers have not been initialized.")
        dist = torch.cdist(z, self.centers)
        q = 1.0 / (1.0 + dist.square() / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        return q / (q.sum(dim=1, keepdim=True) + 1e-10)
