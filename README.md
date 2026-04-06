# 3D Scene Reconstruction with NeRF

This project implements Neural Radiance Fields (NeRF) for 3D scene reconstruction using the LLFF dataset format.

## Project structure

```text
3d_reconstruction_NeRF/
├── data/
│   └── nerf_llff_data/
│       └── fern/
│           ├── images/
│           └── poses_bounds.npy
├── datasets/
│   └── llff.py                 # LLFF data loading and preprocessing
├── models/
│   ├── embedder.py             # Positional encoding
│   └── nerf.py                 # NeRF MLP model
├── rendering/
│   ├── rays.py                 # Ray generation
│   ├── sampler.py              # Point sampling along rays
│   └── ...                     # Volume rendering / rendering utilities
├── training/
│   └── trainer.py              # Training loop / trainer logic
├── nerf_colab.ipynb            # Notebook version / experiments
├── test_pipeline.py            # End-to-end pipeline test
├── train.py                    # Main training entry point
├── .gitignore
└── README.md
```

## Create the data directory

From the root of the project, create the dataset folders:

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
```
## Expected data structure

```
data/
└── nerf_llff_data/
    └── fern/
        ├── images/
        └── poses_bounds.npy
```
## Requirements
```
pip install torch numpy matplotlib imageio
```

## Preprocessing

Implemented in `datasets/llff.py`:

- Load RGB images
- Load camera poses and bounds
- Extract camera-to-world matrices and focal length
- Train / validation / test split

Test:

```
python datasets/llff.py
```


## Pipeline Overview

- Image → Rays → Sample Points → NeRF → Volume Rendering → RGB Output


### Ray Generation

File: `rendering/rays.py`

- Compute ray origins and directions per pixel


### Point Sampling

File: `rendering/sampler.py`

- Uniform sampling between near and far bounds


### Model Architecture

- `models/embedder.py`
- `models/nerf.py`

### Positional Encoding

- Maps coordinates into high-frequency space.

### NeRF Network

- Input: encoded 3D position (+ view direction)
- Output: density (σ) and RGB

### Volume Rendering

File: `rendering/volume_render.py`

- Converts predictions into pixel colors
- Outputs RGB, depth, and weights


## Test the Pipeline
```
python test_pipeline.py
```


### Training

```
python train.py
```

## Training Steps

1. Sample rays
2. Sample points
3. Encode
4. Forward pass
5. Render
6. Compute MSE loss
7. Backpropagation


### Metrics

- PSNR



