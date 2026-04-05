from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.trainer import NeRFTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline NeRF on LLFF data.")

    parser.add_argument("--scene_path", type=str, default="data/nerf_llff_data/fern")
    parser.add_argument("--exp_name", type=str, default="baseline_nerf")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_samples", type=int, default=64)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_decay_step", type=int, default=250)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)

    parser.add_argument("--xyz_num_freqs", type=int, default=10)
    parser.add_argument("--dir_num_freqs", type=int, default=4)

    parser.add_argument("--hold_every", type=int, default=8)
    parser.add_argument("--white_bkgd", action="store_true")

    parser.add_argument("--validate_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)

    parser.add_argument("--render_chunk_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        scene_path=args.scene_path,
        exp_name=args.exp_name,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        xyz_num_freqs=args.xyz_num_freqs,
        dir_num_freqs=args.dir_num_freqs,
        hold_every=args.hold_every,
        white_bkgd=args.white_bkgd,
        validate_every=args.validate_every,
        save_every=args.save_every,
        log_every=args.log_every,
        render_chunk_size=args.render_chunk_size,
        device=args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        seed=args.seed,
    )

    trainer = NeRFTrainer(cfg)

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()