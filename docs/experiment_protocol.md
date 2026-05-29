# Experiment Protocol

## Dataset Split

Follows Sultani et al. (2018) exact protocol:

| Split | Anomaly | Normal | Total |
|-------|---------|--------|-------|
| Train | ~390    | 400    | ~790  |
| Test  | ~143    | 150    | ~293  |

Split files: `data/splits/ucf_crime_train.txt` and `data/splits/ucf_crime_test.txt`.
One video path per line (format: `Category/VideoID_x264.mp4`).

## Reproducibility Harness

Every training run calls `seed_everything(seed)` before any tensor operations:

```python
from ttss.training.reproducibility import seed_everything
seed_everything(42)
```

This seeds: `random`, `numpy`, `torch` (CPU + CUDA), and sets
`cudnn.deterministic=True`, `cudnn.benchmark=False`.

## Run Config Logging

Each experiment writes `outputs/<name>/run_config.yaml` containing:

```yaml
experiment_name: ttss-baseline
seed: 42
git_commit: <short hash>
git_diff_hash: clean | <md5>
timestamp: 2026-05-29T...
python_version: 3.13.x
torch_version: 2.12.0+cu130
config_hash: <sha256[:16]>
training: {epochs: 30, lr: 0.0001, ...}
model: {yolo_weights: yolov8m.pt, vit_unfreeze_blocks: 2}
data: {frame_stride: 8, clip_length: 64, ...}
```

## Frame Rate and Preprocessing

- Source FPS: 30
- `frame_stride: 8` → effective rate: 3.75 fps
- `clip_length: 64` → ~17 seconds per clip
- Frames resized to 224×224, ImageNet-normalised before ViT

## Two Separate Runs Must Produce Identical Results

```bash
python -m ttss.scripts.train --seed 42 --dry-run
python -m ttss.scripts.train --seed 42 --dry-run
# Both should write identical run_config.yaml and produce the same loss
```
