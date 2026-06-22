# Data layout & label format

Videos are **not** redistributed in this repository. Download them from the
official sources (see the README) and arrange as below.

## Expected folder layout
```
data/
├── raw/                 # original downloaded videos (.mp4)
│   ├── 000_real.mp4
│   ├── 001_fake.mp4
│   └── ...
├── face_videos/         # produced by src/preprocess.py (face-cropped clips)
│   └── ...
└── labels.json          # filename -> REAL | FAKE
```

## labels.json format
```json
{
  "000_real.mp4": "REAL",
  "001_fake.mp4": "FAKE"
}
```
Labels may also be given as integers (`REAL = 1`, `FAKE = 0`).

## Notes
- Keep `data/` out of git (already in `.gitignore`).
- If you archive a small, license-permitting **derived** subset for DOI minting,
  store only what the dataset licences allow you to redistribute.
