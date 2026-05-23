# Temporal Threat Scoring System for Proactive Crime Detection (TTSS)

TTSS is an academic research framework for temporal threat assessment on long-form surveillance videos, with the UCF-Crime dataset as the primary benchmark. The project combines spatial recognition, scene-level interpretation, and temporal sequence modeling to estimate a continuous threat score in the range $[0, 1]$ before, during, and after criminal events.

## Aim and Motivation

Classical video anomaly detection pipelines typically operate as binary or weakly supervised detectors that react to deviations from a learned notion of normality. That formulation is often insufficient for proactive crime analysis for three reasons:

1. It is reactive rather than anticipatory, providing low interpretability before an event materializes.
2. It tends to collapse heterogeneous criminal behaviors into a single anomaly score with poor semantic grounding.
3. It rarely models the temporal transition from pre-crime activity to the active crime window and into the post-crime aftermath.

TTSS addresses these limitations with a three-layered architecture:

1. **Recognition Layer** using YOLOv8 for object and person detection.
2. **Detection Layer** using ViT-B/16 for scene-level semantic encoding.
3. **Prediction Layer** using a Bidirectional LSTM to estimate a continuous threat score from fused temporal features.

The central research hypothesis is that semantically grounded spatial cues, fused with scene context and sequence dynamics, can improve early threat awareness relative to purely anomaly-based baselines.

## System Architecture

```mermaid
flowchart LR
    A[UCF-Crime video stream] --> B[Frame sampling and clip generation]
    B --> C[Recognition Layer\nYOLOv8 object-person detection]
    B --> D[Detection Layer\nViT-B/16 scene recognition]
    C --> E[Feature fusion]
    D --> E
    E --> F[Prediction Layer\nBiLSTM temporal modeling]
    F --> G[Threat score in [0.0, 1.0]]
    G --> H[Temporal labels\npre-crime / crime / post-crime]
```

## Temporal Labeling Scheme

TTSS uses a three-stage temporal annotation protocol around the crime interval. Let the annotated crime segment begin at frame $s$ and end at frame $e$.

- **Pre-crime**: frames in $[s - 120, s - 1]$
- **Crime**: frames in $[s, e]$
- **Post-crime**: frames in $[e + 1, e + 120]$

Default window sizes assume $120$ frames for both pre-crime and post-crime context. At $30$ FPS, each context window corresponds to roughly $4$ seconds. Windows are clipped at video boundaries when the full context is unavailable.

### Threat Score Interpretation

- `0.00 - 0.20`: low threat or routine activity
- `0.21 - 0.50`: emerging risk indicators
- `0.51 - 0.80`: elevated threat with actionable cues
- `0.81 - 1.00`: imminent or active threat

## Repository Layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── scripts/
│   ├── infer.py
│   └── train.py
├── tests/
│   └── test_labeling.py
├── ttss/
│   ├── __init__.py
│   └── labeling.py
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

### Conda Environment

```bash
conda create -n ttss python=3.10 -y
conda activate ttss
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional PyTorch CUDA Install

If you need a CUDA-specific PyTorch build, install the appropriate wheel from the official PyTorch index before running the final `pip install -r requirements.txt` step.

## Dataset Preparation

1. Request and download the UCF-Crime dataset from the official source.
2. Store raw videos under `data/raw/UCF-Crime/`.
3. Create annotation metadata with crime start and end frames.
4. Generate frame- or clip-level samples for the temporal windows.
5. Store processed outputs under `data/processed/`.

Recommended structure:

```text
data/
├── raw/
│   └── UCF-Crime/
├── processed/
│   ├── frames/
│   ├── clips/
│   └── annotations/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each metadata row should minimally contain:

- `video_id`
- `label`
- `crime_start_frame`
- `crime_end_frame`
- `fps`
- `split`

## Usage

### Training

```bash
python scripts/train.py \
  --data-root data/processed \
  --annotations data/processed/annotations/train.csv \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 1e-4
```

### Inference

```bash
python scripts/infer.py \
  --video data/raw/UCF-Crime/sample.mp4 \
  --crime-start 300 \
  --crime-end 480 \
  --frame-index 320
```

## Benchmark Placeholder

| Model Variant | Backbone(s) | AUC | AP | Early Warning Lead Time | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| TTSS-YOLOv8-ViT-BiLSTM | YOLOv8 + ViT-B/16 + BiLSTM | TBA | TBA | TBA | Main model |
| TTSS w/o Recognition Layer | ViT-B/16 + BiLSTM | TBA | TBA | TBA | Ablation |
| TTSS w/o Temporal Labels | YOLOv8 + ViT-B/16 + BiLSTM | TBA | TBA | TBA | Ablation |
| Classical Anomaly Baseline | TBD | TBA | TBA | TBA | Baseline |

## Testing

```bash
pytest
```

## Citation

If you use this repository in academic work, cite it as:

```bibtex
@misc{ttss2026,
  title        = {Temporal Threat Scoring System for Proactive Crime Detection},
  author       = {Author Name and Co-Author Name},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Code and models for temporal threat scoring on UCF-Crime},
  url          = {https://github.com/your-username/ttss}
}
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Disclaimer

This repository is intended for academic research on video understanding and temporal risk estimation. Dataset use, privacy handling, and downstream deployment must comply with the original dataset terms, institutional review requirements, and applicable law.