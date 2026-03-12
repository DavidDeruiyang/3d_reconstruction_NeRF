# 3D Scene Reconstruction with NeRF

This project implements Neural Radiance Fields (NeRF) for 3D scene reconstruction using the LLFF dataset format.

## Project structure

```text
3d_recon/
├── data/
│   └── llff/
│       └── fern/
│           ├── images/
│           └── poses_bounds.npy
├── datasets/
│   └── llff.py
├── models/
│   ├── embedder.py
│   └── nerf.py
├── rendering/
│   ├── rays.py
│   ├── sampler.py
│   └── render.py
├── training/
│   └── trainer.py
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
├── utils/
├── test.py
├── render_novel_views.py
└── train.py
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

