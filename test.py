#!/usr/bin/env python
"""Run DUNEM-PET on the released test data.

    python test.py --region head
    python test.py --region thorax

Prints the window SSIM / PSNR / RMSE of DUNEM-PET and of its OSEM initial image for every
slice and their mean, and saves the 20 reconstructed window images as one 4 x 5 figure
in results/.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIONS = {
    "head": dict(checkpoint="checkpoints/dunem_pet_head.pt", data="data/head_test.npz"),
    "thorax": dict(checkpoint="checkpoints/dunem_pet_thorax.pt", data="data/thorax_test.npz"),
}
PSF_FWHM_MM = 4.5          # resolution model of the system operator G
MONTAGE_ROWS, MONTAGE_COLS = 4, 5


def main() -> int:
    ap = argparse.ArgumentParser(description="DUNEM-PET test")
    ap.add_argument("--region", choices=tuple(REGIONS), required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("DUNEM-PET test: a CUDA-capable GPU is required.")
    dev = "cuda"

    from dunem.data import TestCase
    from dunem.metrics import WindowMetrics, shared_minmax, CLIP_PCT
    from dunem.model import load_model
    from dunem.operators import PlaneOp, SharedGeom

    cfg = REGIONS[args.region]
    net = load_model(os.path.join(HERE, cfg["checkpoint"]), dev)
    case = TestCase(os.path.join(HERE, cfg["data"]))
    geom = SharedGeom(case.geometry, PSF_FWHM_MM, case.img_hw, case.voxel_xyz, dev)
    norm = torch.from_numpy(case.norm)
    metric = WindowMetrics(args.region, dev)
    H, W = case.img_hw

    head = (f"{'Slice':>5}   {'DUNEM-PET SSIM':>14} {'PSNR(dB)':>9} {'RMSE':>8}"
            f"   {'OSEM SSIM':>9} {'PSNR(dB)':>9} {'RMSE':>8}")
    print(head)
    rows, panels = [], []
    for i in range(case.n):
        x0, s = case.vectors(i, dev)
        net.phi.set_plane(PlaneOp(geom, torch.from_numpy(case.mu[i]), norm), s)
        xk, _ = net(x0, net.init_y(s))

        sp = case.scale[i]
        mask = case.mu[i] > 0
        rec = xk.reshape(H, W).cpu().numpy() * sp
        rec = rec * mask
        ref = case.x[i] * sp
        osem = case.x0[i] * sp
        m_net, m_osem = metric(rec, ref, mask), metric(osem, ref, mask)
        rows.append((m_net, m_osem))
        print(f"{i + 1:>5}   {m_net['ssim']:>14.4f} {m_net['psnr']:>9.3f} {m_net['rmse']:>8.4f}"
              f"   {m_osem['ssim']:>9.4f} {m_osem['psnr']:>9.3f} {m_osem['rmse']:>8.4f}")

        y0, y1, xa, xb = metric.window(mask, ref.shape)
        panels.append(shared_minmax(rec[y0:y1, xa:xb], ref[y0:y1, xa:xb], CLIP_PCT)[0])

    mean = {k: [float(np.mean([r[j][k] for r in rows])) for j in (0, 1)]
            for k in ("ssim", "psnr", "rmse")}
    print(f"{'Mean':>5}   {mean['ssim'][0]:>14.4f} {mean['psnr'][0]:>9.3f} {mean['rmse'][0]:>8.4f}"
          f"   {mean['ssim'][1]:>9.4f} {mean['psnr'][1]:>9.3f} {mean['rmse'][1]:>8.4f}")

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    h, w = panels[0].shape
    fig, axes = plt.subplots(MONTAGE_ROWS, MONTAGE_COLS,
                             figsize=(MONTAGE_COLS * 2.0 * w / max(h, w),
                                      MONTAGE_ROWS * 2.0 * h / max(h, w) + 0.3))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < len(panels):
            ax.imshow(np.clip(panels[i], 0.0, 1.0), cmap="jet", vmin=0.0, vmax=1.0,
                      interpolation="nearest")
            ax.set_title(f"slice {i + 1}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{args.region}_dunem_pet.png"), dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
