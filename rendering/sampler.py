from __future__ import annotations

import numpy as np


def sample_points(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    near: float,
    far: float,
    n_samples: int,
):
    """
    Uniformly sample 3D points along each ray.

    Args:
        rays_o: Ray origins, shape (N_rays, 3)
        rays_d: Ray directions, shape (N_rays, 3)
        near: Near bound
        far: Far bound
        n_samples: Number of sample points per ray

    Returns:
        z_vals: Sample depths, shape (N_rays, n_samples)
        points: 3D sampled points, shape (N_rays, n_samples, 3)
    """
    if rays_o.ndim != 2 or rays_o.shape[-1] != 3:
        raise ValueError(f"rays_o must have shape (N_rays, 3), got {rays_o.shape}")
    if rays_d.ndim != 2 or rays_d.shape[-1] != 3:
        raise ValueError(f"rays_d must have shape (N_rays, 3), got {rays_d.shape}")
    if rays_o.shape[0] != rays_d.shape[0]:
        raise ValueError("rays_o and rays_d must have the same number of rays")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if near >= far:
        raise ValueError("near must be smaller than far")

    n_rays = rays_o.shape[0]

    # Uniform depth samples between near and far
    z_steps = np.linspace(near, far, n_samples, dtype=np.float32)  # (n_samples,)
    z_vals = np.broadcast_to(z_steps[None, :], (n_rays, n_samples)).copy()

    # points = o + t * d
    points = rays_o[:, None, :] + z_vals[:, :, None] * rays_d[:, None, :]

    return z_vals.astype(np.float32), points.astype(np.float32)


if __name__ == "__main__":
    # Simple sanity test
    rays_o = np.array([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0]], dtype=np.float32)
    rays_d = np.array([[0.0, 0.0, -1.0],
                       [0.0, 1.0,  0.0]], dtype=np.float32)

    z_vals, points = sample_points(rays_o, rays_d, near=2.0, far=6.0, n_samples=4)

    print("z_vals shape:", z_vals.shape)
    print("points shape:", points.shape)
    print("z_vals[0]:", z_vals[0])
    print("points[0]:")
    print(points[0])