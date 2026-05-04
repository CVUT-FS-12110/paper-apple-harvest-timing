# Apple Harvest Timing Model

Repository for Delta 2026 conference submission: Image-Based Prediction of Apple Harvest Timing

The image data are not stored in git. Download the dataset from Figshare and place it in:

```text
data/dataset_class/
```

The `data/` directory is ignored by git, so each user should download the data locally.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

- `src/dataset.py` defines the PyTorch dataset, image loading, resizing, target parsing from filenames, and optional Albumentations augmentation.
- `src/run.py` runs the training/evaluation loop, creates train/validation/test day splits, saves checkpoints, and writes logs.

## Run

```bash
cd src
python run.py
```

Outputs are written to `logs/` and `checkpoints/`.
