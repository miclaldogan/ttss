# Cross-Dataset Generalization

Zero-shot evaluation on ShanghaiTech-Campus and XD-Violence using a checkpoint
trained exclusively on UCF-Crime.

## Dataset Annotation Formats

### ShanghaiTech-Campus
- Per-frame binary labels in `test_frame_mask/<clip_id>.npy` (1 = anomaly, 0 = normal)
- Frame images in `frames/<clip_id>/*.jpg`
- 13 scenes, 107 test clips

### XD-Violence
- Per-frame binary labels in `annotations/<video_id>.npy`
- Videos in `videos/<video_id>.mp4`
- Split lists in `splits/train.txt` and `splits/test.txt`
- 6 violence categories: Fighting, Shooting, Riot, Abuse, CarAccident, Explosion

## Adaptation Notes

Both loaders follow the same interface as `UcfCrimeDataset`:
- `load_frames=False` for label-only mode (fast evaluation)
- `frame_stride` matches the UCF-Crime training stride

## Running Zero-Shot Evaluation

```bash
python -m ttss.scripts.eval_crossdataset \
    --checkpoint checkpoints/best.pt \
    --shanghaitech data/ShanghaiTech \
    --xd-violence  data/XD-Violence \
    --output       evaluation/cross_dataset_results.json
```

## Expected Results Schema

```json
{
  "shanghaitech": {"auc": 0.0, "ear": 0.0, "n_sequences": 107},
  "xd_violence":  {"auc": 0.0, "ear": 0.0, "n_sequences": 0}
}
```
