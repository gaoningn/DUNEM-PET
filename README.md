# DUNEM-PET: Learned Alternating Minimization with Dual-domain Priors and Residual Coupling for PET Reconstruction

Test code and trained weights of DUNEM-PET, with head and thorax test data (20 consecutive
slices each), so that the reconstruction results can be reproduced with a single command.

## Environment

- Linux with an NVIDIA GPU (CUDA 12.x)
- Python 3.10
- PyTorch 2.12.0
- CuPy 14.1.1
- NumPy 2.2.6
- array-api-compat 1.15.0
- Matplotlib 3.10.9

## Directory structure

```
DUNEM-PET/
├── test.py                  # test entry point
├── dunem/                   # network, system model, metrics and data loading
├── checkpoints/
│   ├── dunem_pet_head.pt
│   └── dunem_pet_thorax.pt
└── data/
    ├── head_test.npz        # head test data (20 slices)
    └── thorax_test.npz      # thorax test data (20 slices)
```

## Usage

```bash
python test.py --region head
python test.py --region thorax
```

The region option selects the corresponding weights, test data and settings automatically.

## Output

The terminal shows, for every slice and as a mean over the 20 slices, the SSIM, PSNR (dB) and
RMSE of the DUNEM-PET reconstruction and of its OSEM initial image, both computed against the
reference image inside a fixed window around the object.

The 20 reconstructed window images are saved as one 4 × 5 figure (jet colormap):

```
results/head_dunem_pet.png
results/thorax_dunem_pet.png
```

## Reference results

Mean over the 20 test slices:

| Region | Method    | SSIM   | PSNR (dB) | RMSE   |
|--------|-----------|--------|-----------|--------|
| Head   | OSEM      | 0.9630 | 28.313    | 0.0384 |
| Head   | DUNEM-PET | 0.9732 | 30.652    | 0.0294 |
| Thorax | OSEM      | 0.8528 | 26.176    | 0.0492 |
| Thorax | DUNEM-PET | 0.9148 | 29.704    | 0.0328 |

GPU back-projection is not bit-wise deterministic, so repeated runs may differ slightly in the
last printed digit.
