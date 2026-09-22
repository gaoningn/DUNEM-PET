"""Test-case loading and intensity normalization.

Each test file holds consecutive slices of one volume:
    sinogram   (P, n_radial, n_views)  measured counts
    x0         (P, H, W)               OSEM initial image
    reference  (P, H, W)               reference image
    mu         (P, H, W)               attenuation map [1/mm]
    norm       (n_radial, n_views)     normalization sinogram
    geometry   JSON                    scanner geometry of the 2-D system model
    voxel_size_mm_xyz (3,)
Images and sinogram of a slice are divided by the same per-slice scale (the maximum of
the clipped reference slice), so the linear model sinogram ~ G(image) is unchanged.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from .metrics import CLIP_PCT, despike_plane


class TestCase:
    def __init__(self, path: str):
        d = np.load(path)
        self.geometry = json.loads(str(d["geometry"]))
        self.voxel_xyz = tuple(float(v) for v in d["voxel_size_mm_xyz"])
        x = d["reference"].astype(np.float32)
        x0 = d["x0"].astype(np.float32)
        y = d["sinogram"].astype(np.float32)
        self.mu = d["mu"].astype(np.float32)
        self.norm = d["norm"].astype(np.float32)
        P, H, W = x.shape
        self.n, self.img_hw = P, (H, W)

        x = x.copy()
        x0 = x0.copy()
        for z in range(P):                                  # clip hot pixels per slice
            x[z] = despike_plane(x[z], CLIP_PCT)
            x0[z] = despike_plane(x0[z], CLIP_PCT)
        plane_max = x.reshape(P, -1).max(1)
        floor = 1e-3 * max(float(plane_max.max()), 1e-8)
        self.scale = np.maximum(plane_max, floor).astype(np.float32)   # per-slice scale
        sp = self.scale[:, None, None]
        self.x = x / sp
        self.x0 = x0 / sp
        self.y = y / sp

    def vectors(self, i: int, device: str):
        """(x0, s) of slice i as (1, n, 1) / (1, m, 1) tensors."""
        x0 = torch.from_numpy(self.x0[i]).reshape(-1, 1).to(device)[None]
        s = torch.from_numpy(self.y[i]).reshape(-1, 1).to(device)[None]
        return x0, s
