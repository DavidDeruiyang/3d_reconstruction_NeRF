from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from datasets.llff import load_llff_data
from rendering.rays import get_rays_flat
from rendering.sampler import sample_points
from models.embedder import PositionalEncoder
from models.nerf import NeRF
from rendering.volume_render import volume_render


SCENE_PATH = Path("data/nerf_llff_data/fern")

'''
This test will run the nerf pipeline in a small scale, only to verify if the pipeline works"
'''


def main():
    # ---------------------------
    # 1. Load LLFF data
    # ---------------------------
    data = load_llff_data(SCENE_PATH)

    print("Loaded LLFF data")
    print("images shape:", data.images.shape)
    print("poses shape:", data.poses.shape)
    print("bounds shape:", data.bounds.shape)
    print("focal:", data.focal)

    # ---------------------------
    # 2. Choose one image / pose
    # ---------------------------
    image_idx = 0
    pose = data.poses[image_idx]
    H, W = data.height, data.width
    focal = data.focal

    # ---------------------------
    # 3. Generate rays
    # ---------------------------
    rays_o, rays_d = get_rays_flat(H, W, focal, pose)
    print("\nGenerated rays")
    print("rays_o shape:", rays_o.shape)
    print("rays_d shape:", rays_d.shape)

    # ---------------------------
    # 4. Take a small subset of rays
    # ---------------------------
    n_rays = 1024
    rays_o = rays_o[:n_rays]
    rays_d = rays_d[:n_rays]

    print("\nSubsampled rays")
    print("rays_o subset shape:", rays_o.shape)
    print("rays_d subset shape:", rays_d.shape)

    # ---------------------------
    # 5. Sample points on rays
    # ---------------------------
    near = float(data.bounds.min())
    far = float(data.bounds.max())
    n_samples = 64

    z_vals, points = sample_points(
        rays_o=rays_o,
        rays_d=rays_d,
        near=near,
        far=far,
        n_samples=n_samples,
    )

    print("\nSampled points")
    print("z_vals shape:", z_vals.shape)
    print("points shape:", points.shape)
    print("near:", near)
    print("far:", far)

    # ---------------------------
    # 6. Positional encoding
    # ---------------------------
    xyz_encoder = PositionalEncoder(num_freqs=10)
    dir_encoder = PositionalEncoder(num_freqs=4)

    encoded_points = xyz_encoder.encode(points)   # (N_rays, N_samples, 63)

    # Encode one direction per ray, then expand to all samples
    encoded_dirs = dir_encoder.encode(rays_d)     # (N_rays, 27)
    encoded_dirs = np.repeat(encoded_dirs[:, None, :], n_samples, axis=1)

    print("\nEncoded features")
    print("encoded_points shape:", encoded_points.shape)
    print("encoded_dirs shape:", encoded_dirs.shape)

    # ---------------------------
    # 7. Convert to torch
    # ---------------------------
    encoded_points_t = torch.from_numpy(encoded_points).float()
    encoded_dirs_t = torch.from_numpy(encoded_dirs).float()
    z_vals_t = torch.from_numpy(z_vals).float()
    rays_d_t = torch.from_numpy(rays_d).float()

    # Flatten points/samples for NeRF forward pass
    n_rays_actual, n_samples_actual, feat_dim = encoded_points_t.shape
    dir_dim = encoded_dirs_t.shape[-1]

    encoded_points_flat = encoded_points_t.reshape(-1, feat_dim)
    encoded_dirs_flat = encoded_dirs_t.reshape(-1, dir_dim)

    print("\nTorch tensors")
    print("encoded_points_flat shape:", encoded_points_flat.shape)
    print("encoded_dirs_flat shape:", encoded_dirs_flat.shape)

    # ---------------------------
    # 8. Build NeRF model
    # ---------------------------
    model = NeRF(
        input_ch=xyz_encoder.output_dim(3),
        input_ch_dir=dir_encoder.output_dim(3),
    )

    # ---------------------------
    # 9. Run NeRF forward
    # ---------------------------
    rgb_flat, sigma_flat = model(encoded_points_flat, encoded_dirs_flat)

    # Reshape back to per-ray layout
    rgb = rgb_flat.reshape(n_rays_actual, n_samples_actual, 3)
    sigma = sigma_flat.reshape(n_rays_actual, n_samples_actual, 1)

    print("\nNeRF outputs")
    print("rgb shape:", rgb.shape)
    print("sigma shape:", sigma.shape)

    # ---------------------------
    # 10. Volume render
    # ---------------------------
    rgb_map, depth_map, acc_map, weights = volume_render(
        rgb=rgb,
        sigma=sigma,
        z_vals=z_vals_t,
        rays_d=rays_d_t,
        white_bkgd=False,
    )

    print("\nRendered outputs")
    print("rgb_map shape:", rgb_map.shape)
    print("depth_map shape:", depth_map.shape)
    print("acc_map shape:", acc_map.shape)
    print("weights shape:", weights.shape)

    # ---------------------------
    # 11. Quick sanity prints
    # ---------------------------
    print("\nSanity checks")
    print("rgb_map min/max:", rgb_map.min().item(), rgb_map.max().item())
    print("depth_map min/max:", depth_map.min().item(), depth_map.max().item())
    print("acc_map min/max:", acc_map.min().item(), acc_map.max().item())
    print("weights min/max:", weights.min().item(), weights.max().item())

    print("rgb min/max:", rgb.min().item(), rgb.max().item())
    print("sigma min/max:", sigma.min().item(), sigma.max().item())
    print("sigma mean:", sigma.mean().item())


if __name__ == "__main__":
    main()