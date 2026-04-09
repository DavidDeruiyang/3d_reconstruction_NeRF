# Training and Demo Guide

## Training example

```bash
python train.py \
  --scene_path data/nerf_llff_data/fern \
  --exp_name baseline_fern \
  --num_epochs 10 \
  --steps_per_epoch 1000 \
  --batch_size 1024 \
  --n_samples 64 \
  --device cuda
```

## Resume example

```bash
python train.py \
  --scene_path data/nerf_llff_data/fern \
  --exp_name baseline_fern \
  --resume outputs/baseline_fern/checkpoints/final.pt
```

## Demo example

```bash
python demo.py \
  --scene_path data/nerf_llff_data/fern \
  --checkpoint outputs/baseline_fern/checkpoints/final.pt \
  --image_idx 0 \
  --output_dir demo_outputs
```

## Expected demo outputs

After a successful run, the output folder should contain:

```text
demo_outputs/
├── pred_rgb.png
├── pred_depth.npy
├── pred_acc.npy
├── target.png
└── summary.txt
```

## Interpreting the files

- `target.png`: the real dataset image for the selected view
- `pred_rgb.png`: the model's reconstructed image for that camera pose
- `pred_depth.npy`: estimated depth values from volume rendering
- `pred_acc.npy`: opacity / occupancy accumulation along rays
- `summary.txt`: scene path, checkpoint path, image size, and selected index

## Typical workflow

1. verify dataset loading with `python datasets/llff.py`
2. run `python test_pipeline.py`
3. train with `python train.py`
4. render outputs with `python demo.py`
5. use saved RGB and depth images in report or presentation material
