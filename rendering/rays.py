from __future__ import annotations

import numpy as np


def get_rays(H: int, W: int, focal: float, c2w: np.ndarray):
    """
    Generate ray origins and directions for all pixels in one image.

    Args:
        H: image height
        W: image width
        focal: focal length
        c2w: camera-to-world pose matrix, shape (3, 4) or (4, 4)

    Returns:
        rays_o: ray origins, shape (H, W, 3)
        rays_d: ray directions, shape (H, W, 3)
    """
    if c2w.shape == (3, 4):
        rotation = c2w[:, :3]
        translation = c2w[:, 3]
    elif c2w.shape == (4, 4):
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
    else:
        raise ValueError(f"Expected c2w shape (3,4) or (4,4), got {c2w.shape}")

    # Pixel coordinate grid
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )

    # Camera-frame ray directions
    # NeRF/LLFF-style convention
    dirs = np.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -np.ones_like(i),
        ],
        axis=-1,
    )  # (H, W, 3)

    # Rotate ray directions into world frame
    rays_d = dirs @ rotation.T  # (H, W, 3)

    # All rays originate from the camera center
    rays_o = np.broadcast_to(translation, rays_d.shape).copy()

    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def get_rays_flat(H: int, W: int, focal: float, c2w: np.ndarray):
    """
    Same as get_rays(), but flattened to (H*W, 3).
    Useful for batching rays during training.
    """
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


if __name__ == "__main__":
    # Simple standalone sanity test
    H, W, focal = 100, 100, 50.0
    c2w = np.eye(4, dtype=np.float32)

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    print("rays_o shape:", rays_o.shape)  # (H, W, 3)
    print("rays_d shape:", rays_d.shape)  # (H, W, 3)
    print("center ray origin:", rays_o[H // 2, W // 2])
    print("center ray direction:", rays_d[H // 2, W // 2])