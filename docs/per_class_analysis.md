# Per-Category Performance Analysis

## Overview

UCF-Crime spans 13 crime categories. Per-category metrics reveal model biases and guide future work.

## Categories and Expected Difficulty

| Category | Visual cues | Expected difficulty |
|----------|-------------|---------------------|
| Shooting | Weapon, body falling | Easy |
| Explosion | Smoke, flash, crowd scatter | Easy |
| Fighting | Person proximity, motion | Medium |
| RoadAccidents | Vehicle trajectories | Medium |
| Robbery | Weapon, fast movement | Medium |
| Arrest | Multiple persons, restraint | Hard |
| Burglary | Slow entry, low action | Hard |
| Shoplifting | Subtle concealment | Hard |

## Generating the Results

```bash
# Run evaluation on test set
python -m ttss.scripts.evaluate_per_class --checkpoint checkpoints/best.pt

# Plot Figure 7
python -m ttss.scripts.plot_per_class --results evaluation/per_class_results.json
```

## Interpreting Results

- **High AUC categories** (Shooting, Explosion): strong visual signals align with YOLOv8 detection features.
- **Low AUC categories** (Shoplifting, Burglary): subtle actions require fine-grained temporal reasoning beyond object detection.
- **Macro-average** is weighted by video count; compare it to the global AUC to check for sampling bias.

## Discussion Template (Paper Section 4.3)

> Table 2 shows per-category AUC breakdown. TTSS achieves the highest AUC on X (0.XX) and Y (0.XX), consistent with these categories exhibiting strong object-level cues that YOLOv8m reliably detects. Performance is lowest on Shoplifting (0.XX) and Burglary (0.XX), where crimes unfold through subtle actions over long temporal spans — a known limitation of frame-level feature extractors. This motivates future work on longer-range temporal modelling and action-level representations.
