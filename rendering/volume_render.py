from __future__ import annotations

import torch


def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_bkgd: bool = False,
):
    """
    Volume render NeRF outputs along rays.

    Args:
        rgb: Predicted colors, shape (N_rays, N_samples, 3)
        sigma: Predicted densities, shape (N_rays, N_samples, 1) or (N_rays, N_samples)
        z_vals: Sample depths, shape (N_rays, N_samples)
        rays_d: Ray directions, shape (N_rays, 3)
        white_bkgd: If True, composite onto a white background

    Returns:
        rgb_map: Rendered RGB colors, shape (N_rays, 3)
        depth_map: Rendered depth, shape (N_rays,)
        acc_map: Accumulated opacity, shape (N_rays,)
        weights: Sample weights, shape (N_rays, N_samples)
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (N_rays, N_samples, 3), got {rgb.shape}")

    if sigma.ndim == 3 and sigma.shape[-1] == 1:
        sigma = sigma.squeeze(-1)
    elif sigma.ndim != 2:
        raise ValueError(
            f"sigma must have shape (N_rays, N_samples, 1) or (N_rays, N_samples), got {sigma.shape}"
        )

    if z_vals.ndim != 2:
        raise ValueError(f"z_vals must have shape (N_rays, N_samples), got {z_vals.shape}")

    if rays_d.ndim != 2 or rays_d.shape[-1] != 3:
        raise ValueError(f"rays_d must have shape (N_rays, 3), got {rays_d.shape}")

    n_rays, n_samples = z_vals.shape

    if rgb.shape[0] != n_rays or rgb.shape[1] != n_samples:
        raise ValueError("rgb shape does not match z_vals shape")
    if sigma.shape[0] != n_rays or sigma.shape[1] != n_samples:
        raise ValueError("sigma shape does not match z_vals shape")
    if rays_d.shape[0] != n_rays:
        raise ValueError("rays_d shape does not match number of rays")

    # Distance between consecutive samples along each ray
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)

    # Large distance for the last sample
    delta_inf = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, delta_inf], dim=-1)  # (N_rays, N_samples)

    # Scale deltas by ray length in world coordinates
    ray_norms = torch.norm(rays_d, dim=-1, keepdim=True)  # (N_rays, 1)
    deltas = deltas * ray_norms

    # Convert density to alpha
    alpha = 1.0 - torch.exp(-sigma * deltas)  # (N_rays, N_samples)

    # Compute transmittance T
    # T_i = prod_{j<i} (1 - alpha_j)
    eps = 1e-10
    trans = torch.cumprod(
        torch.cat(
            [torch.ones((n_rays, 1), device=alpha.device, dtype=alpha.dtype), 1.0 - alpha + eps],
            dim=-1,
        ),
        dim=-1,
    )
    trans = trans[:, :-1]  # (N_rays, N_samples)

    # Weights for each sample
    weights = alpha * trans  # (N_rays, N_samples)

    # Final rendered outputs
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)   # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, dim=-1)         # (N_rays,)
    acc_map = torch.sum(weights, dim=-1)                    # (N_rays,)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


if __name__ == "__main__":
    # Simple sanity test
    N_rays = 4
    N_samples = 8

    rgb = torch.rand(N_rays, N_samples, 3)
    sigma = torch.rand(N_rays, N_samples, 1)
    z_vals = torch.linspace(2.0, 6.0, N_samples).unsqueeze(0).repeat(N_rays, 1)
    rays_d = torch.randn(N_rays, 3)

    rgb_map, depth_map, acc_map, weights = volume_render(rgb, sigma, z_vals, rays_d)

    print("rgb shape:", rgb.shape)
    print("sigma shape:", sigma.shape)
    print("z_vals shape:", z_vals.shape)
    print("rays_d shape:", rays_d.shape)
    print("rgb_map shape:", rgb_map.shape)
    print("depth_map shape:", depth_map.shape)
    print("acc_map shape:", acc_map.shape)
    print("weights shape:", weights.shape)