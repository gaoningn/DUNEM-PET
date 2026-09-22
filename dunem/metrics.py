"""Window metrics: SSIM, PSNR and RMSE inside a fixed-size window centred on the object.

The window is centred on the bounding box of the object support (mu > 0) of each slice.
Inside the window both images are clipped at the 99.9th percentile and mapped onto the
reference's own [min, max] range, then scored (SSIM: 11x11 Gaussian window, sigma 1.5,
data range 1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

WINDOW = {"head": (80, 80), "thorax": (120, 210)}
CLIP_PCT = 99.9


def despike_plane(v: np.ndarray, pct: float) -> np.ndarray:
    """Clip a plane to its pct-th percentile over all voxels (pct >= 100 = off)."""
    if pct >= 100 or float(v.max()) <= 0:
        return v
    hi = float(np.percentile(v, pct))
    return np.clip(v, 0.0, hi) if hi > 0 else v


def shared_minmax(a: np.ndarray, truth: np.ndarray, pct: float):
    """Map `a` and `truth` onto [0, 1] using the (clipped) reference's min / max."""
    t = despike_plane(truth, pct)
    lo, hi = float(t.min()), float(t.max())
    scale = hi - lo + 1e-8
    a = despike_plane(a, pct)
    return (a - lo) / scale, (t - lo) / scale


def fixed_crop(mask2d: np.ndarray, size_hw, img_shape=None):
    """(y0, y1, x0, x1) of a `size_hw` window centred on the True pixels of `mask2d`."""
    H, W = int(size_hw[0]), int(size_hw[1])
    sh = img_shape if img_shape is not None else mask2d.shape
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        cy, cx = sh[0] // 2, sh[1] // 2
    else:
        cy = (int(ys.min()) + int(ys.max()) + 1) // 2
        cx = (int(xs.min()) + int(xs.max()) + 1) // 2
    y0 = int(np.clip(cy - H // 2, 0, max(sh[0] - H, 0)))
    x0 = int(np.clip(cx - W // 2, 0, max(sh[1] - W, 0)))
    return y0, y0 + H, x0, x0 + W


def _gauss(ws: int, sigma: float) -> torch.Tensor:
    g = torch.exp(-((torch.arange(ws).float() - ws // 2) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).unsqueeze(1)


class SSIMMap:
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0,
                 device: str = "cuda"):
        w1 = _gauss(window_size, sigma)
        self.window = (w1 @ w1.t())[None, None].to(device)
        self.ws, self.L, self.device = window_size, float(data_range), device

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ta = torch.as_tensor(a, device=self.device, dtype=torch.float32)[None, None]
        tb = torch.as_tensor(b, device=self.device, dtype=torch.float32)[None, None]
        w, p = self.window, self.ws // 2
        mu1, mu2 = F.conv2d(ta, w, padding=p), F.conv2d(tb, w, padding=p)
        m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = F.conv2d(ta * ta, w, padding=p) - m1s
        s2 = F.conv2d(tb * tb, w, padding=p) - m2s
        s12 = F.conv2d(ta * tb, w, padding=p) - m12
        C1, C2 = (0.01 * self.L) ** 2, (0.03 * self.L) ** 2
        m = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2))
        return m[0, 0].cpu().numpy()


class WindowMetrics:
    def __init__(self, region: str, device: str = "cuda"):
        self.size = WINDOW[region]
        self.ssim = SSIMMap(device=device)

    def window(self, mask: np.ndarray, shape):
        return fixed_crop(mask, self.size, shape)

    def __call__(self, rec: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
        y0, y1, x0, x1 = self.window(mask, truth.shape)
        a, t = shared_minmax(rec[y0:y1, x0:x1], truth[y0:y1, x0:x1], CLIP_PCT)
        mse = float(((a - t) ** 2).mean())
        return {"ssim": float(self.ssim(a, t).mean()),
                "psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                "rmse": float(np.sqrt(mse))}
