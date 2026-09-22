"""2-D plane-wise PET system operator built with parallelproj.

    image plane (H, W)  --G-->  sinogram plane (n_radial, n_views)
    fwd(x) = N (X) G(x),   adj(v) = G^T(N (X) v),   G = att(mu) . proj . psf

N is the normalization sinogram, att = exp(-proj(mu)) the attenuation factors of the
plane and psf an image-domain Gaussian resolution model. The projector and PSF are built
once per geometry; only the attenuation is rebuilt per plane.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import array_api_compat.torch as xp_torch  # noqa: E402
    import parallelproj as pp                  # noqa: E402


def build_projector(geom: dict, img_shape_xyz, voxel_xyz, device: str):
    sg = geom["scanner_geometry"]
    nz = img_shape_xyz[2]
    vz = voxel_xyz[2]
    xp = xp_torch
    ring_positions = (xp.arange(nz, dtype=xp.float32) - (nz - 1) / 2.0) * vz
    scanner = pp.RegularPolygonPETScannerGeometry(
        xp, device,
        radius=float(sg["ring_radius_mm"]),
        num_sides=int(sg["num_sides"]),
        num_lor_endpoints_per_side=int(sg["num_lor_endpoints_per_side"]),
        lor_spacing=float(sg["lor_spacing_mm"]),
        ring_positions=ring_positions,
        symmetry_axis=2,
    )
    lor_desc = pp.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=int(geom["radial_trim"]),
        max_ring_difference=int(geom["max_ring_difference"]),
        sinogram_order=pp.SinogramSpatialAxisOrder.RVP,
    )
    origin = (-(np.asarray(img_shape_xyz) - 1) / 2.0 * np.asarray(voxel_xyz)).astype(np.float32)
    return pp.RegularPolygonPETProjector(
        lor_desc,
        img_shape=tuple(int(s) for s in img_shape_xyz),
        voxel_size=tuple(float(v) for v in voxel_xyz),
        img_origin=xp.asarray(origin),
    )


def build_psf(img_shape_xyz, voxel_xyz, fwhm_mm: float):
    sigma_vox = [fwhm_mm / (2.355 * v) for v in voxel_xyz]
    return pp.GaussianFilterOperator(tuple(int(s) for s in img_shape_xyz), sigma=sigma_vox)


class _FwdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return op(x.detach().contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        return ctx.op.adjoint(grad_out.detach().contiguous()), None


class _AdjFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, op):
        ctx.op = op
        return op.adjoint(v.detach().contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        return ctx.op(grad_out.detach().contiguous()), None


class SharedGeom:
    """Projector + PSF for one 2-D plane (nz = 1), shared by all planes."""

    def __init__(self, geom: dict, psf_fwhm_mm: float, img_hw, voxel_xyz, device: str = "cuda"):
        self.device = device
        H, W = img_hw
        self.img_shape_xyz = (W, H, 1)             # stored plane (y, x) -> (x, y, 1)
        self.voxel_xyz = tuple(float(v) for v in voxel_xyz)
        self.projector = build_projector(geom, self.img_shape_xyz, self.voxel_xyz, device)
        self.psf_op = (build_psf(self.img_shape_xyz, self.voxel_xyz, psf_fwhm_mm)
                       if psf_fwhm_mm > 0 else None)
        self.sino_shape = tuple(int(s) for s in self.projector.out_shape)

    def make_G(self, mu_plane_hw: torch.Tensor):
        mu_xyz = mu_plane_hw.t().unsqueeze(-1).to(self.device).float().contiguous()
        with torch.no_grad():
            mu_sino = self.projector(mu_xyz)
            att = torch.exp(-mu_sino).float().contiguous()
        ops = [pp.ElementwiseMultiplicationOperator(att), self.projector]
        if self.psf_op is not None:
            ops.append(self.psf_op)
        return pp.CompositeLinearOperator(ops)


class PlaneOp:
    """Per-plane forward / adjoint on flattened vectors:
       image (1, H*W, 1), view(H, W) = stored plane (y, x)
       sinogram (1, R*V, 1), view(R, V) = (n_radial, n_views)"""

    def __init__(self, geom: SharedGeom, mu_plane_hw: torch.Tensor, norm_rv: torch.Tensor):
        self.geom = geom
        self.H, self.W = int(mu_plane_hw.shape[0]), int(mu_plane_hw.shape[1])
        self.R, self.V, _ = geom.sino_shape
        self.n = self.H * self.W
        self.m = self.R * self.V
        self.G = geom.make_G(mu_plane_hw)
        self.norm2d = norm_rv.to(geom.device).contiguous()

    def fwd(self, x_vec):
        x_img = x_vec.reshape(self.H, self.W).t().unsqueeze(-1).contiguous()
        y_sino = _FwdFn.apply(x_img, self.G)
        y_sino = y_sino * self.norm2d.unsqueeze(-1)
        like = x_vec.new_empty(x_vec.shape[:-2] + (self.m, 1))
        return y_sino.reshape_as(like).contiguous()

    def adj(self, y_vec):
        y_sino = y_vec.reshape(self.R, self.V).unsqueeze(-1).contiguous() * self.norm2d.unsqueeze(-1)
        x_img = _AdjFn.apply(y_sino, self.G)
        like = y_vec.new_empty(y_vec.shape[:-2] + (self.n, 1))
        return x_img.squeeze(-1).t().reshape_as(like).contiguous()
