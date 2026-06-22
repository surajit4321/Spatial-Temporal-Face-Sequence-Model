# Spatial-Temporal-Face-Sequence-Model
### Spatial–Temporal Face Sequence Modelling for Robust Deepfake Video Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
<!-- Add after archiving on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

> **This repository contains the official source code directly related to the manuscript
> _"Spatial–Temporal Face Sequence Modelling for Robust Deepfake Video Detection"_,
> submitted to _The Visual Computer_ (2026).**
> **If you use this code, data, or any part of this work, please cite the manuscript
> (see [Citation](#-citation)).**

A **Hybrid CNN-Transformer (HCT)** framework for video-level deepfake detection. It
couples **EfficientNet-B4** per-frame spatial feature extraction with a **Transformer
encoder** that models inter-frame temporal inconsistencies, enabling joint detection of
spatial manipulation artefacts and temporal flicker introduced by face synthesis.

---

## Table of Contents
- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Configuration](#configuration)
- [Results & Evaluation](#results--evaluation)
- [Ethical Use](#ethical-use)
- [Citation](#-citation)
- [License](#license)

---

## Highlights
- **Hybrid CNN-Transformer** architecture: EfficientNet-B4 (1792-d spatial features)
  → linear projection (→256-d) → sinusoidal positional encoding → Transformer encoder
  → temporal mean pooling → MLP classifier.
- **Reusable face-only preprocessing pipeline** using the dlib HOG/CNN frontal-face
  detector (`face_recognition`).
- **Imbalance-aware evaluation**: accuracy, balanced accuracy, precision, recall,
  F1, AUC-ROC, PR-AUC, Matthews correlation coefficient (MCC) and per-class recall.
- **Reproducible**: pinned dependencies, fixed seeds, single config file, and
  both runnable notebooks and importable `src/` modules.

## Method Overview

```
Input video
   │  uniform sampling → N = 30 face-cropped frames  [B, T, 3, 224, 224]
   ▼
EfficientNet-B4 (per frame)            → [B·T, 1792]
   ▼
Linear(1792→256) + LayerNorm + GELU    → [B, T, 256]
   ▼
+ Sinusoidal Positional Encoding (dropout 0.3)
   ▼
Transformer Encoder (Pre-LN, 2 layers, 4 heads, FFN 512)
   ▼
Temporal mean pooling                  → [B, 256]
   ▼
MLP classifier (256→256→2)             → REAL / FAKE
```

The full, importable implementation is in [`src/model.py`](src/model.py); the
annotated, figure-producing version is in
[`notebooks/Model_and_train_hybrid_WITH_VISUALIZATIONS_MODIFIED_30_BS_8.ipynb`](notebooks/).

## Repository Structure

```
Spatial-Temporal-Face-Sequence-Model/
├── README.md                  # this file
├── LICENSE                    # MIT
├── CITATION.cff               # how to cite the code + manuscript
├── CONTRIBUTING.md
├── requirements.txt           # pip dependencies
├── environment.yml            # conda environment
├── configs/
│   └── default.yaml           # primary config (N=30, B=8)
├── src/                       # importable implementation of key algorithms
│   ├── config.py              # experiment configuration (dataclass)
│   ├── model.py               # Hybrid CNN-Transformer (HCT)
│   ├── dataset.py             # VideoDataset + transforms
│   ├── preprocess.py          # face-only preprocessing pipeline
│   ├── train.py               # training loop (AdamW + CosineAnnealingLR + AMP)
│   ├── predict.py             # inference on a video / folder
│   └── metrics.py             # imbalance-aware metrics
├── notebooks/                 # original, annotated, figure-producing notebooks
│   ├── preprocessing_hybrid.ipynb
│   ├── Model_and_train_hybrid_WITH_VISUALIZATIONS_MODIFIED_30_BS_8.ipynb
│   └── Predict_hybrid_fixed2_final.ipynb
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── GITHUB_ZENODO_DOI_GUIDE.md
│   └── PRE_ACCEPTANCE_CHECKLIST.md
└── scripts/
    └── download_data.md
```

## Installation

**Option A — conda (recommended; handles the dlib/CMake build):**
```bash
git clone https://github.com/<your-username>/Spatial-Temporal-Face-Sequence-Model.git
cd Spatial-Temporal-Face-Sequence-Model
conda env create -f environment.yml
conda activate hct-deepfake
```

**Option B — pip:**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> `dlib` (used by `face_recognition`) requires CMake and a C++ compiler. On Ubuntu:
> `sudo apt-get install build-essential cmake`. A CUDA-enabled GPU is recommended for training.

**Tested environment:** Python 3.10, PyTorch 2.x, CUDA 11.8, timm ≥ 0.9.2.

## Datasets

This work uses three public benchmarks. Due to licensing and privacy, **the videos
themselves are not redistributed here** — download them from the official sources and
agree to their terms:

| Dataset | Source |
|---|---|
| FaceForensics++ (FF++) | https://github.com/ondyari/FaceForensics |
| DFDC (Deepfake Detection Challenge) | https://ai.meta.com/datasets/dfdc/ |
| Celeb-DF (v2) | https://github.com/yuezunli/celeb-deepfakeforensics |

See [`scripts/download_data.md`](scripts/download_data.md) for the expected folder
layout and the label-file (`labels.json`) format.

## Reproducing the Experiments

**1. Preprocess (extract + crop faces):**
```bash
python -m src.preprocess --input_dir ./data/raw --output_dir ./data/face_videos
```

**2. Train (primary configuration N=30, B=8):**
```bash
python -m src.train \
    --data_dir ./data/face_videos \
    --labels   ./data/labels.json \
    --epochs 30 --batch_size 8 --sequence_length 30 \
    --out checkpoints/hct_best.pth
```

**3. Predict / evaluate:**
```bash
python -m src.predict --checkpoint checkpoints/hct_best.pth --video_dir ./data/face_videos
```

The notebooks in `notebooks/` reproduce the same pipeline interactively and generate
all paper figures (architecture diagrams, training curves, confusion matrix, ROC/PR curves).

## Configuration

All hyperparameters live in [`src/config.py`](src/config.py) / [`configs/default.yaml`](configs/default.yaml):

| Group | Setting | Value |
|---|---|---|
| Input | image size / sequence length (N) | 224 / 30 |
| Backbone | CNN | EfficientNet-B4 (timm, ImageNet-pretrained) |
| Transformer | d_model / heads / layers / FFN / dropout | 256 / 4 / 2 / 512 / 0.3 |
| Optimiser | AdamW lr / weight decay | 1e-4 / 1e-4 |
| Schedule | CosineAnnealingLR (eta_min) | 1e-6 |
| Training | epochs / batch size / split | 30 / 8 / 80–20 |
| Precision | mixed precision (AMP) | enabled on CUDA |

The manuscript's ablation study sweeps **N ∈ {10, 20, 30}** and **B ∈ {4, 6, 8}**
(plus CNN-backbone and Transformer-depth ablations); change the corresponding flags
to reproduce each configuration.

## Results & Evaluation

Because the corpus is imbalanced (**FAKE ≈ 86.7%**), evaluation uses balanced
accuracy, MCC, PR-AUC and per-class recall in addition to accuracy/AUC-ROC, computed
in [`src/metrics.py`](src/metrics.py). Refer to Section IV of the manuscript for the
full result tables and threshold-sensitivity analysis.

## Ethical Use

This software is intended for **research and forensic-assistance** purposes only.
Deepfake detectors can produce false positives and false negatives and must **not** be
used as the sole basis for any accusation or automated decision affecting a person.
Datasets may carry demographic bias; detectors may be vulnerable to adversarial
adaptation. See the ethics discussion in the manuscript.

## 📌 Citation

This code is **directly related to** the following manuscript. If you use this
repository in your research, **please cite it**:

```bibtex
@article{paul2026hct,
  title   = {Spatial--Temporal Face Sequence Modelling for Robust Deepfake Video Detection},
  author  = {Paul, Surajit and Chakraborty, Angshuman and Devi, Bishnulatpam Pushpa},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript under review. DOI to be added upon acceptance.}
}
```

If you also wish to cite the archived software release, use the Zenodo DOI (added once
minted) and the metadata in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
