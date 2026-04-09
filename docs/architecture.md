# Architecture Notes

## 1. Input data

The project uses LLFF-style scene folders with:

- multi-view RGB images in `images/`
- camera poses and near/far bounds in `poses_bounds.npy`

The loader in `datasets/llff.py` reads:

- `images`: shape `(N, H, W, 3)`
- `poses`: shape `(N, 3, 4)`
- `bounds`: shape `(N, 2)`
- `hwf`: shape `(N, 3)` containing height, width, and focal length

## 2. Ray generation

For each pixel, `rendering/rays.py` computes:

- ray origin `o`
- ray direction `d`

using the focal length and camera-to-world matrix.

## 3. Point sampling

`rendering/sampler.py` samples `n_samples` depth values between near and far bounds.
Each 3D query point is computed as:

`x(t) = o + t d`

This produces a tensor of sampled points with shape `(N_rays, N_samples, 3)`.

## 4. Positional encoding

`models/embedder.py` transforms 3D coordinates and view directions into a higher-dimensional feature space using sinusoidal encoding.

For each scalar input `x`, the encoder outputs:

`[x, sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)]`

This helps the MLP represent high-frequency details.

## 5. NeRF MLP

`models/nerf.py` implements a baseline NeRF network with:

- point-processing trunk
- density head for `sigma`
- feature branch
- view-direction branch for RGB

Inputs:

- encoded 3D point
- encoded viewing direction

Outputs:

- RGB in `[0, 1]`
- non-negative density `sigma`

## 6. Volume rendering

`rendering/volume_render.py` converts per-sample predictions into final outputs per ray:

- rendered RGB
- depth map
- accumulated opacity
- per-sample weights

This stage computes alpha values, transmittance, and weighted sums along the ray.

## 7. Training loop

`training/trainer.py` performs:

1. random ray sampling from one training image
2. point sampling along rays
3. positional encoding
4. NeRF forward pass
5. volume rendering
6. MSE loss against target RGB
7. PSNR logging
8. validation rendering and checkpoint save

## 8. Outputs

The training pipeline produces:

- checkpoint files under `outputs/<exp_name>/checkpoints/`
- validation renders under `outputs/<exp_name>/val_renders/`
