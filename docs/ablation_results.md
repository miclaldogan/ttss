# Ablation Study Results

This document contains the result table template for the TTSS systematic ablation study. Rows are filled after training on UCF-Crime and running `python -m ttss.scripts.run_ablation --experiment arch window fusion`.

---

## 1. Architecture Component Ablation

| Variant | Input dim | BiLSTM | Attention | AUC-ROC | EAR | Notes |
|---------|----------:|--------|-----------|--------:|----:|-------|
| **Full TTSS** | 1536 | Bi | ✓ | TBA | TBA | Baseline |
| No ViT | 8 | Bi | ✓ | TBA | TBA | YOLO-8 features only |
| No attention | 1536 | Bi | ✗ (mean pool) | TBA | TBA | |
| Unidirectional | 1536 | Uni | ✓ | TBA | TBA | `bidirectional=False` |
| No pre-crime loss | 1536 | Bi | ✓ | TBA | TBA | `precrime_weight=0` |
| No temporal consistency | 1536 | Bi | ✓ | TBA | TBA | `λ2=0` |

---

## 2. Pre-crime Window K Sweep

| K (frames) | AUC-ROC | EAR | Notes |
|-----------:|--------:|----:|-------|
| 0 | TBA | TBA | No pre-crime context |
| 30 | TBA | TBA | ≈1 s at 30 FPS |
| 60 | TBA | TBA | ≈2 s |
| 90 | TBA | TBA | ≈3 s |
| **120** | TBA | TBA | Default |
| 150 | TBA | TBA | ≈5 s |

---

## 3. Feature Fusion Strategy

| Fusion | Input dim | AUC-ROC | EAR | Notes |
|--------|----------:|--------:|----:|-------|
| **Concatenation** | 776 | TBA | TBA | Default — YOLO-8 ∥ ViT-768 |
| Additive | 768 | TBA | TBA | YOLOv8 projected + ViT-768 |
| Attention-weighted | 768 | TBA | TBA | Cross-attention over YOLO and ViT |

---

## Reproduction

```bash
# Generate all ablation JSON files
python -m ttss.scripts.run_ablation --experiment all

# Individual experiments
python -m ttss.scripts.run_ablation --experiment arch
python -m ttss.scripts.run_ablation --experiment window
python -m ttss.scripts.run_ablation --experiment fusion
```

Output files:
- `evaluation/ablation_arch.json`
- `evaluation/ablation_window.json`
- `evaluation/ablation_fusion.json`
