# Reproducibility Guide

This document lets an independent reader reproduce the experiments end to end.

## 1. Environment
- Python 3.10, PyTorch 2.x, CUDA 11.8, timm ≥ 0.9.2 (see `requirements.txt` / `environment.yml`).
- Fixed random seed (`seed = 42`) set in `src/train.py` for `random`, `numpy`, and `torch`.
- A CUDA GPU is recommended; the code falls back to CPU automatically (much slower).

## 2. Data
- Download FF++, DFDC, and Celeb-DF from their official sources (README).
- Arrange as in `scripts/download_data.md` and create `labels.json`.

## 3. Pipeline
1. **Preprocess:** `python -m src.preprocess --input_dir data/raw --output_dir data/face_videos`
   - dlib HOG/CNN face detection, crop, resize, write face-only clips.
2. **Train:** `python -m src.train --data_dir data/face_videos --labels data/labels.json`
   - EfficientNet-B4 + Transformer, AdamW (lr 1e-4, wd 1e-4), CosineAnnealingLR, AMP.
   - Best checkpoint (by validation balanced accuracy) saved to `checkpoints/`.
3. **Evaluate / predict:** `python -m src.predict --checkpoint checkpoints/hct_best.pth --video_dir data/face_videos`

## 4. Configurations / ablations
Reproduce ablation rows by varying flags:
- Sequence length: `--sequence_length {10,20,30}`
- Batch size: `--batch_size {4,6,8}`
- Backbone / Transformer depth: edit `cnn_backbone` / `num_transformer_layers` in `src/config.py`.

## 5. Metrics
All metrics are computed in `src/metrics.py` (accuracy, balanced accuracy, precision,
recall, F1, AUC-ROC, PR-AUC, MCC, per-class recall) to account for class imbalance.

## 6. Expected artefacts
- `checkpoints/hct_best.pth` — trained weights + config + validation metrics.
- Notebook runs additionally export figures (training curves, confusion matrix, ROC/PR).
