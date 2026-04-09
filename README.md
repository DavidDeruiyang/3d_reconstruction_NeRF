# 3D Scene Reconstruction with NeRF

This repository implements a compact PyTorch version of **Neural Radiance Fields (NeRF)** for **3D scene reconstruction** and **novel view synthesis** from multi-view images in **LLFF** format.

The codebase covers the full baseline pipeline:

1. load LLFF images, poses, and scene bounds
2. generate camera rays per pixel
3. sample 3D points along rays
4. apply positional encoding
5. predict color and density with a NeRF MLP
6. volume-render RGB and depth
7. train on held-out views and save checkpoints

---

## Repository structure

```text
3d_reconstruction_NeRF/
├── datasets/
│   └── llff.py                  # LLFF loading and split logic
├── models/
│   ├── embedder.py              # Positional encoding
│   └── nerf.py                  # NeRF MLP
├── rendering/
│   ├── rays.py                  # Ray generation
│   ├── sampler.py               # Point sampling along rays
│   └── volume_render.py         # Differentiable volume rendering
├── training/
│   └── trainer.py               # Training / validation / checkpoint logic
├── docs/
│   ├── architecture.md          # Extended architecture notes
│   └── training_and_demo.md     # Training, checkpoint, and demo guide
├── nerf_colab.ipynb             # Notebook workflow / experimentation
├── test_pipeline.py             # Small end-to-end pipeline sanity test
├── train.py                     # Main training entrypoint
├── demo.py                      # Demo script for checkpoint-based rendering
├── requirements.txt             # Reproducible Python dependencies
└── README.md
```

---

## What the project currently supports

- LLFF scene loading from `images/` and `poses_bounds.npy`
- Train/validation/test split for held-out view evaluation
- Baseline NeRF model with positional encoding and view directions
- Differentiable volume rendering
- Training with MSE loss and PSNR logging
- Validation image rendering during training
- Checkpoint save/resume
- Demo rendering from a saved checkpoint

---

## Environment setup

### Option A: `venv`

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate      # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n nerf3d python=3.10 -y
conda activate nerf3d
pip install -r requirements.txt
```

---

## Data format

The code expects an **LLFF-style** scene folder.

```text
data/
└── nerf_llff_data/
    └── fern/
        ├── images/
        │   ├── 000.jpg
        │   ├── 001.jpg
        │   └── ...
        └── poses_bounds.npy
```

### Download example data

#### Option A (download from NeRF repo):
```
git clone https://github.com/bmild/nerf
conda env create -f environment.yml
conda activate nerf
bash download_example_data.sh

mkdir ./3d_reconstruction/data
// move data to ./3d_reconstruction/data
```
#### Option B (directly download zip file):
```
go to http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
upzip to your ./3d_reconstruction/data directory


---

## Quick checks

### 1) Verify the dataset loader

```bash
python datasets/llff.py
```

This prints:

- number of images
- image tensor shape
- pose shape
- bounds shape
- focal length
- train/val/test counts

### 2) Run an end-to-end pipeline sanity test

```bash
python test_pipeline.py
```

This script:

- loads one LLFF scene
- generates rays
- samples points
- encodes positions and directions
- runs a forward pass through the NeRF model
- volume-renders RGB/depth/opacity
- saves diagnostic plots such as sigma and rendering weights

Outputs are written under `outputs/`.

---

## Training

Train the baseline NeRF model:

```bash
python train.py \
  --scene_path data/nerf_llff_data/fern \
  --exp_name baseline_fern \
  --num_epochs 10 \
  --steps_per_epoch 1000 \
  --batch_size 1024 \
  --n_samples 64
```

### Important training arguments

- `--scene_path`: path to one LLFF scene
- `--exp_name`: output folder name under `outputs/`
- `--num_epochs`: number of epochs
- `--steps_per_epoch`: optimization steps per epoch
- `--batch_size`: number of rays per batch
- `--n_samples`: samples along each ray
- `--device`: `cuda` or `cpu`
- `--resume`: path to a saved checkpoint

### Training outputs

The trainer writes files under:

```text
outputs/<exp_name>/
├── checkpoints/
│   ├── step_000500.pt
│   ├── step_001000.pt
│   └── final.pt
└── val_renders/
    ├── step_000200_view_0.png
    └── ...
```

During training, the console logs loss, PSNR, validation PSNR, and checkpoint saves.

---

## Resume training

```bash
python train.py \
  --scene_path data/nerf_llff_data/fern \
  --exp_name baseline_fern \
  --resume outputs/baseline_fern/checkpoints/final.pt
```

---

## Demo: render from a trained checkpoint

Once you have a checkpoint, render one held-out or chosen view:

```bash
python demo.py \
  --scene_path data/nerf_llff_data/fern \
  --checkpoint outputs/baseline_fern/checkpoints/final.pt \
  --image_idx 0 \
  --output_dir demo_outputs
```

The demo script saves:

- `target.png` — ground-truth input image from the dataset
- `pred_rgb.png` — rendered RGB output from the model
- `pred_depth.npy` — predicted depth map
- `pred_acc.npy` — accumulated opacity map
- `summary.txt` — quick metadata summary

---

## Code walkthrough

### `datasets/llff.py`
Loads images, camera poses, near/far bounds, and focal length from LLFF data.

### `rendering/rays.py`
Converts each camera view into one ray origin and direction per pixel.

### `rendering/sampler.py`
Uniformly samples 3D query points between near and far planes for each ray.

### `models/embedder.py`
Applies sinusoidal positional encoding to improve high-frequency scene representation.

### `models/nerf.py`
Implements the baseline NeRF MLP that predicts:

- RGB color
- density (`sigma`)

### `rendering/volume_render.py`
Aggregates per-sample predictions into final RGB, depth, opacity, and per-sample weights.

### `training/trainer.py`
Handles batching, optimization, validation, checkpointing, and rendering.

---

## Extended documentation

See the `/docs` folder for more detail:

- `docs/architecture.md`
- `docs/training_and_demo.md`

---

## Reference

> Alexander, J.A. & Mozer, M.C.(1995) Template-based algorithms for connectionist rule extraction. In G.Tesauro, D.S.Touretzky and T.K.Leen(eds.), *Advances in Neural Information Processing Systems 7*, pp.609--616. Cambridge, MA: MIT Press.


> Bower, J.M.& Beeman, D.(1995) *The Book of GENESIS: Exploring Realistic Neural Models with the GEneral NEural SImulation System.* New York:TELOS/Springer--Verlag.


> Hasselmo, M.E., Schnell, E.& Barkai, E.(1995) Dynamics of learning and recall at excitatory recurrent synapses and cholinergic modulation in rat hippocampal region CA3. Journal of Neuroscience (7):5249-5262.