from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from training.trainer import NeRFTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a demo output from a trained NeRF checkpoint.")
    parser.add_argument("--scene_path", type=str, default="data/nerf_llff_data/fern")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="demo_outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--render_chunk_size", type=int, default=16384)
    return parser.parse_args()


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        scene_path=args.scene_path,
        exp_name="demo_run",
        num_epochs=1,
        steps_per_epoch=1,
        render_chunk_size=args.render_chunk_size,
        device=args.device if args.device is not None else "cpu",
    )

    trainer = NeRFTrainer(cfg)
    trainer.load_checkpoint(args.checkpoint)

    image_idx = int(args.image_idx)
    if image_idx < 0 or image_idx >= trainer.data.n_images:
        raise ValueError(f"image_idx must be in [0, {trainer.data.n_images - 1}]")

    pose = trainer.data.poses[image_idx]
    target = trainer.data.images[image_idx]
    rendered = trainer.render_image(pose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(output_dir / "target.png", to_uint8(target))
    imageio.imwrite(output_dir / "pred_rgb.png", to_uint8(rendered["rgb"]))
    np.save(output_dir / "pred_depth.npy", rendered["depth"])
    np.save(output_dir / "pred_acc.npy", rendered["acc"])

    summary = [
        f"scene_path: {args.scene_path}",
        f"checkpoint: {args.checkpoint}",
        f"image_idx: {image_idx}",
        f"image_size: {trainer.height} x {trainer.width}",
        f"near_far: {trainer.near:.6f}, {trainer.far:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
