from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np


@dataclass
class LLFFData:
    """Container for LLFF scene data.

    Attributes:
        images: RGB images as float32 in [0, 1], shape (N, H, W, 3)
        poses: Camera pose matrices, shape (N, 3, 4)
        bounds: Near/far bounds per image, shape (N, 2)
        hwf: Per-image [H, W, focal] values, shape (N, 3)
        image_paths: Paths to the loaded image files
    """

    images: np.ndarray
    poses: np.ndarray
    bounds: np.ndarray
    hwf: np.ndarray
    image_paths: List[Path]

    @property
    def n_images(self) -> int:
        return int(self.images.shape[0])

    @property
    def height(self) -> int:
        return int(self.images.shape[1])

    @property
    def width(self) -> int:
        return int(self.images.shape[2])

    @property
    def focal(self) -> float:
        # After loading, H/W/focal are typically the same for all images.
        return float(self.hwf[0, 2])


class LLFFDatasetError(RuntimeError):
    """Raised when the LLFF dataset folder is malformed."""



def _list_image_files(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix in exts)
    if not image_paths:
        raise LLFFDatasetError(f"No images found in: {images_dir}")
    return image_paths



def _read_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)

    if image.ndim == 2:
        # Convert grayscale to RGB by repeating the channel.
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        # Drop alpha if present.
        image = image[..., :3]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise LLFFDatasetError(f"Unsupported image shape {image.shape} for file: {path}")

    return image.astype(np.float32) / 255.0



def _load_images(image_paths: Sequence[Path]) -> np.ndarray:
    images = [_read_image(path) for path in image_paths]

    first_shape = images[0].shape
    for path, img in zip(image_paths, images):
        if img.shape != first_shape:
            raise LLFFDatasetError(
                "All LLFF images must have the same shape. "
                f"Expected {first_shape}, got {img.shape} for {path}"
            )

    return np.stack(images, axis=0)



def _load_pose_bounds(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_bounds_path = scene_dir / "poses_bounds.npy"
    if not pose_bounds_path.exists():
        raise LLFFDatasetError(f"Missing poses_bounds.npy in: {scene_dir}")

    poses_bounds = np.load(pose_bounds_path)
    if poses_bounds.ndim != 2 or poses_bounds.shape[1] != 17:
        raise LLFFDatasetError(
            "poses_bounds.npy must have shape (N, 17). "
            f"Got {poses_bounds.shape}"
        )

    # First 15 values reshape to (3, 5): [R|t|hwf]
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5).astype(np.float32)

    # Last 2 values are near/far bounds.
    bounds = poses_bounds[:, 15:].astype(np.float32)

    poses = poses_hwf[:, :, :4]  # (N, 3, 4)
    hwf = poses_hwf[:, :, 4]     # (N, 3) -> [H, W, focal]
    return poses, bounds, hwf



def load_llff_data(scene_dir: str | Path) -> LLFFData:
    """Load an LLFF scene from disk.

    Expected structure:
        scene_dir/
            images/
                *.jpg / *.png
            poses_bounds.npy

    Args:
        scene_dir: Path to one LLFF scene, e.g. data/llff/fern

    Returns:
        LLFFData with images, poses, bounds, and intrinsics metadata.
    """
    scene_dir = Path(scene_dir)
    images_dir = scene_dir / "images"

    if not scene_dir.exists():
        raise LLFFDatasetError(f"Scene directory does not exist: {scene_dir}")
    if not images_dir.exists():
        raise LLFFDatasetError(f"Missing images/ directory in: {scene_dir}")

    image_paths = _list_image_files(images_dir)
    images = _load_images(image_paths)
    poses, bounds, hwf = _load_pose_bounds(scene_dir)

    n_images = images.shape[0]
    if poses.shape[0] != n_images:
        raise LLFFDatasetError(
            "Image count does not match poses count. "
            f"Found {n_images} images but {poses.shape[0]} poses."
        )

    loaded_h, loaded_w = images.shape[1], images.shape[2]
    pose_h = int(round(float(hwf[0, 0])))
    pose_w = int(round(float(hwf[0, 1])))

    # In most LLFF scenes, the H/W in poses_bounds should match the image size.
    # If not, we keep loading but make the mismatch explicit.
    if pose_h != loaded_h or pose_w != loaded_w:
        print(
            "[load_llff_data] Warning: image size from poses_bounds.npy "
            f"is ({pose_h}, {pose_w}), but loaded images are ({loaded_h}, {loaded_w})."
        )

    return LLFFData(
        images=images,
        poses=poses,
        bounds=bounds,
        hwf=hwf,
        image_paths=image_paths,
    )



def train_val_test_split(
    n_images: int,
    hold_every: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple LLFF-style split.

    Args:
        n_images: Number of images in the scene
        hold_every: Every k-th image is placed into val/test.

    Returns:
        train_idx, val_idx, test_idx as integer numpy arrays.

    Notes:
        This is a simple split that mirrors common LLFF evaluation setups.
        For a class project, using the same held-out indices for validation and
        testing is acceptable at first; you can refine it later if needed.
    """
    if n_images <= 0:
        raise ValueError("n_images must be positive")
    if hold_every <= 0:
        raise ValueError("hold_every must be positive")

    all_idx = np.arange(n_images)
    val_test_idx = all_idx[::hold_every]
    train_mask = np.ones(n_images, dtype=bool)
    train_mask[val_test_idx] = False
    train_idx = all_idx[train_mask]

    return train_idx, val_test_idx.copy(), val_test_idx.copy()



def summarize_llff(scene_dir: str | Path) -> None:
    """Convenience function for quick debugging from a script or terminal."""
    data = load_llff_data(scene_dir)
    train_idx, val_idx, test_idx = train_val_test_split(data.n_images)

    print(f"Scene: {scene_dir}")
    print(f"Number of images: {data.n_images}")
    print(f"Images shape: {data.images.shape}")
    print(f"Poses shape: {data.poses.shape}")
    print(f"Bounds shape: {data.bounds.shape}")
    print(f"H, W, focal: {data.height}, {data.width}, {data.focal:.4f}")
    print(f"Near/Far range: {data.bounds.min():.4f} to {data.bounds.max():.4f}")
    print(f"Train images: {len(train_idx)}")
    print(f"Val images: {len(val_idx)}")
    print(f"Test images: {len(test_idx)}")


if __name__ == "__main__":
    # Example usage:
    # python datasets/llff.py
    default_scene = Path("data/llff/fern")
    summarize_llff(default_scene)
