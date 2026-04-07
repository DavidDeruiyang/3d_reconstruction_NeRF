from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from datasets.llff import load_llff_data, train_val_test_split
from models.embedder import PositionalEncoder
from models.nerf import NeRF
from rendering.rays import get_rays_flat
from rendering.sampler import sample_points
from rendering.volume_render import volume_render


@dataclass
class TrainConfig:
    scene_path: str = "data/nerf_llff_data/fern"
    exp_name: str = "baseline_nerf"

    num_epochs: int = 10
    steps_per_epoch: int = 1000

    batch_size: int = 1024
    n_samples: int = 64

    lr: float = 5e-4
    lr_decay_step: int = 250
    lr_decay_gamma: float = 0.5

    xyz_num_freqs: int = 10
    dir_num_freqs: int = 4

    hold_every: int = 8
    white_bkgd: bool = False

    validate_every: int = 200
    save_every: int = 500
    log_every: int = 25

    render_chunk_size: int = 16384
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class NeRFTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.scene_path = Path(config.scene_path)
        self.exp_dir = Path("outputs") / config.exp_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.val_dir = self.exp_dir / "val_renders"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.data = load_llff_data(self.scene_path)
        self.train_idx, self.val_idx, self.test_idx = train_val_test_split(
            self.data.n_images,
            hold_every=config.hold_every,
        )

        self.height = self.data.height
        self.width = self.data.width
        self.focal = self.data.focal
        self.near = float(self.data.bounds.min())
        self.far = float(self.data.bounds.max())

        self.xyz_encoder = PositionalEncoder(num_freqs=config.xyz_num_freqs)
        self.dir_encoder = PositionalEncoder(num_freqs=config.dir_num_freqs)

        self.model = NeRF(
            input_ch=self.xyz_encoder.output_dim(3),
            input_ch_dir=self.dir_encoder.output_dim(3),
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_gamma,
        )

        self.global_step = 0

        print(f"[trainer] Scene path: {self.scene_path}")
        print(f"[trainer] Train images: {len(self.train_idx)}")
        print(f"[trainer] Val images: {len(self.val_idx)}")
        print(f"[trainer] Test images: {len(self.test_idx)}")
        print(f"[trainer] Image size: {self.height} x {self.width}")
        print(f"[trainer] Near/Far: {self.near:.4f} / {self.far:.4f}")
        print(f"[trainer] Device: {self.device}")

    def sample_random_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample rays from a single randomly selected training image.
        This avoids storing all rays from all images in memory.
        """
        img_idx = int(np.random.choice(self.train_idx))
        pose = self.data.poses[img_idx]
        image = self.data.images[img_idx]  # (H, W, 3)

        rays_o_np, rays_d_np = get_rays_flat(self.height, self.width, self.focal, pose)
        rgb_np = image.reshape(-1, 3).astype(np.float32)

        n_pixels = rays_o_np.shape[0]
        replace = self.cfg.batch_size > n_pixels
        pixel_idx = np.random.choice(n_pixels, size=self.cfg.batch_size, replace=replace)

        rays_o = torch.from_numpy(rays_o_np[pixel_idx]).float().to(self.device)
        rays_d = torch.from_numpy(rays_d_np[pixel_idx]).float().to(self.device)
        target_rgb = torch.from_numpy(rgb_np[pixel_idx]).float().to(self.device)

        return rays_o, rays_d, target_rgb

    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rays_o_np = rays_o.detach().cpu().numpy()
        rays_d_np = rays_d.detach().cpu().numpy()

        z_vals_np, points_np = sample_points(
            rays_o=rays_o_np,
            rays_d=rays_d_np,
            near=self.near,
            far=self.far,
            n_samples=self.cfg.n_samples,
        )

        encoded_points_np = self.xyz_encoder.encode(points_np)
        encoded_dirs_np = self.dir_encoder.encode(rays_d_np)
        encoded_dirs_np = np.repeat(
            encoded_dirs_np[:, None, :],
            self.cfg.n_samples,
            axis=1,
        )

        encoded_points = torch.from_numpy(encoded_points_np).float().to(self.device)
        encoded_dirs = torch.from_numpy(encoded_dirs_np).float().to(self.device)
        z_vals = torch.from_numpy(z_vals_np).float().to(self.device)

        n_rays, n_samples, feat_dim = encoded_points.shape
        dir_dim = encoded_dirs.shape[-1]

        encoded_points_flat = encoded_points.reshape(-1, feat_dim)
        encoded_dirs_flat = encoded_dirs.reshape(-1, dir_dim)

        rgb_flat, sigma_flat = self.model(encoded_points_flat, encoded_dirs_flat)

        rgb = rgb_flat.reshape(n_rays, n_samples, 3)
        sigma = sigma_flat.reshape(n_rays, n_samples, 1)

        rgb_map, depth_map, acc_map, weights = volume_render(
            rgb=rgb,
            sigma=sigma,
            z_vals=z_vals,
            rays_d=rays_d,
            white_bkgd=self.cfg.white_bkgd,
        )

        return {
            "rgb_map": rgb_map,
            "depth_map": depth_map,
            "acc_map": acc_map,
            "weights": weights,
            "z_vals": z_vals,
        }

    def train_step(self) -> Dict[str, float]:
        self.model.train()

        rays_o, rays_d, target_rgb = self.sample_random_batch()
        outputs = self.render_rays(rays_o, rays_d)
        pred_rgb = outputs["rgb_map"]

        loss = F.mse_loss(pred_rgb, target_rgb)
        psnr = -10.0 * torch.log10(loss + 1e-10)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        return {
            "loss": float(loss.item()),
            "psnr": float(psnr.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def render_image(self, pose: np.ndarray) -> Dict[str, np.ndarray]:
        self.model.eval()

        rays_o_np, rays_d_np = get_rays_flat(self.height, self.width, self.focal, pose)
        rays_o = torch.from_numpy(rays_o_np).float().to(self.device)
        rays_d = torch.from_numpy(rays_d_np).float().to(self.device)

        rgb_chunks = []
        depth_chunks = []
        acc_chunks = []

        chunk = self.cfg.render_chunk_size
        for start in range(0, rays_o.shape[0], chunk):
            end = start + chunk
            outputs = self.render_rays(rays_o[start:end], rays_d[start:end])
            rgb_chunks.append(outputs["rgb_map"].cpu())
            depth_chunks.append(outputs["depth_map"].cpu())
            acc_chunks.append(outputs["acc_map"].cpu())

        rgb = torch.cat(rgb_chunks, dim=0).reshape(self.height, self.width, 3).numpy()
        depth = torch.cat(depth_chunks, dim=0).reshape(self.height, self.width).numpy()
        acc = torch.cat(acc_chunks, dim=0).reshape(self.height, self.width).numpy()

        return {"rgb": rgb, "depth": depth, "acc": acc}

    @torch.no_grad()
    def validate(self, image_idx: int | None = None) -> Dict[str, float]:
        self.model.eval()

        if len(self.val_idx) == 0:
            raise RuntimeError("Validation split is empty.")

        if image_idx is None:
            image_idx = int(self.val_idx[0])

        target = self.data.images[image_idx]
        pose = self.data.poses[image_idx]

        outputs = self.render_image(pose)
        pred = outputs["rgb"]

        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).float()

        mse = F.mse_loss(pred_t, target_t)
        psnr = -10.0 * torch.log10(mse + 1e-10)

        pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        imageio.imwrite(
            self.val_dir / f"step_{self.global_step:06d}_view_{image_idx}.png",
            pred_u8,
        )

        return {
            "val_mse": float(mse.item()),
            "val_psnr": float(psnr.item()),
            "val_image_idx": float(image_idx),
        }

    def save_checkpoint(self, name: str | None = None) -> Path:
        if name is None:
            name = f"step_{self.global_step:06d}.pt"

        ckpt_path = self.ckpt_dir / name
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": vars(self.cfg),
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = int(ckpt["global_step"])
        print(f"[trainer] Resumed from checkpoint: {ckpt_path}")

    def train(self) -> None:
        for epoch in range(self.cfg.num_epochs):
            print(f"\n[epoch {epoch + 1}/{self.cfg.num_epochs}]")

            for _ in range(self.cfg.steps_per_epoch):
                metrics = self.train_step()

                if self.global_step % self.cfg.log_every == 0:
                    print(
                        f"step={self.global_step:06d} "
                        f"loss={metrics['loss']:.6f} "
                        f"psnr={metrics['psnr']:.2f} "
                        f"lr={metrics['lr']:.6e}"
                    )

                if self.global_step % self.cfg.validate_every == 0:
                    val_metrics = self.validate()
                    print(
                        f"[val] step={self.global_step:06d} "
                        f"mse={val_metrics['val_mse']:.6f} "
                        f"psnr={val_metrics['val_psnr']:.2f}"
                    )

                if self.global_step % self.cfg.save_every == 0:
                    ckpt_path = self.save_checkpoint()
                    print(f"[ckpt] saved to {ckpt_path}")

            self.scheduler.step()

        final_ckpt = self.save_checkpoint("final.pt")
        print(f"\nTraining complete. Final checkpoint: {final_ckpt}")