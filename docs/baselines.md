# Baseline Comparison Suite

This document describes the four comparison baselines implemented in
`ttss/baselines/` for reproducing the UCF-Crime benchmark comparisons
required by CVPR / ICCV / ECCV reviewers.

---

## Baselines

### Sultani et al. (2018) — C3D + MIL Ranking Loss

| | |
|---|---|
| **Paper** | Sultani, W., Chen, C., & Shah, M. "Real-world Anomaly Detection in Surveillance Videos." *CVPR 2018* |
| **arXiv** | https://arxiv.org/abs/1801.04264 |
| **Class** | `ttss.baselines.sultani2018.Sultani2018Baseline` |
| **Key** | `sultani2018` |

**Architecture**: C3D backbone (16-frame clips) + two-stream MIL ranking classifier. The model assigns anomaly scores to non-overlapping 32-frame segments and interpolates to per-frame scores.

**Checkpoint URL**: *(to be filled once official weights are released or hosted)*

```bash
# Download checkpoint (placeholder)
wget -O checkpoints/sultani2018.pt <checkpoint_url>
```

---

### RTFM — Robust Temporal Feature Magnitude (Tian et al., 2021)

| | |
|---|---|
| **Paper** | Tian, Y. et al. "Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning." *ICCV 2021* |
| **arXiv** | https://arxiv.org/abs/2101.10030 |
| **Class** | `ttss.baselines.rtfm.RTFMBaseline` |
| **Key** | `rtfm` |

**Architecture**: Temporal segment feature extractor (I3D / C3D) + MIL head that scores anomalies by the L2 magnitude of segment feature vectors. Magnitude-based scoring consistently ranks high-anomaly segments above normal ones without explicit label supervision.

**Checkpoint URL**: *(to be filled)*

---

### Zhong et al. (2019) — GCN + Self-Supervised Feature Learning

| | |
|---|---|
| **Paper** | Zhong, J.-X. et al. "Graph Convolutional Label Noise Cleaner: Train a Plug-and-Play Action Classifier for Anomaly Detection." *CVPR 2019* |
| **Note** | Not yet wrapped; planned for a future issue |

---

### MIST — Multiple Instance Self-Training (Feng et al., 2021)

| | |
|---|---|
| **Paper** | Feng, J. et al. "MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection." *CVPR 2021* |
| **Note** | Not yet wrapped; planned for a future issue |

---

### Mean ViT-B/16 Feature + Linear SVM

| | |
|---|---|
| **Class** | `ttss.baselines.mean_feature_svm.MeanFeatureSVMBaseline` |
| **Key** | `mean_feature_svm` |

**Architecture**: Per-frame ViT-B/16 CLS tokens (768-d) aggregated with mean-pooling over a sliding window; scored by the signed distance to a scikit-learn `OneClassSVM` decision hyperplane fit on normal training frames.

**Fitting the SVM**:

```python
from sklearn.svm import OneClassSVM
import pickle, numpy as np

# Assume `normal_features` is (N, 768) from normal UCF-Crime frames
svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
svm.fit(normal_features)
with open("checkpoints/mean_feature_svm.pkl", "wb") as f:
    pickle.dump(svm, f)
```

---

## Running the Evaluation

```bash
# Evaluate all baselines (synthetic data when UCF-Crime is not present)
python -m ttss.scripts.evaluate_baselines --baseline all --split test

# Evaluate a single baseline
python -m ttss.scripts.evaluate_baselines --baseline rtfm --split test

# Evaluate with real checkpoints
python -m ttss.scripts.evaluate_baselines \
  --baseline sultani2018 rtfm mean_feature_svm \
  --split test \
  --output evaluation/baseline_results.json
```

Output JSON schema:

```json
{
  "split": "test",
  "timestamp": "...",
  "threshold": 0.5,
  "results": [
    {
      "baseline": "sultani2018",
      "video_id": "UCF-Crime/Testing/Robbery001.mp4",
      "frame_auc": 0.7231,
      "early_alarm_rate": 0.6400
    }
  ]
}
```

---

## Reproduction Instructions

1. Request the UCF-Crime dataset from the [official source](https://www.crcv.ucf.edu/projects/real-world/).
2. Place test videos under `data/raw/UCF-Crime/Testing/`.
3. Download or train baseline checkpoints (see URLs above).
4. Place checkpoints under `checkpoints/`.
5. Run `python -m ttss.scripts.evaluate_baselines --baseline all --split test`.

Results are written to `evaluation/baseline_results.json`.
