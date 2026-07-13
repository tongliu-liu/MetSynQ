# LC-MS Data Processing and Benchmark Analysis

This repository contains executable workflows for LC-MS training-data
preparation and benchmark analysis.

## Repository layout

```text
.
├── training/          # YOLO dataset splitting and horizontal augmentation
├── widely_targeted/   # widely targeted benchmark analysis
└── untargeted/        # untargeted benchmark and ablation analysis
```

Each subdirectory has its own README and dependency file. Python 3.9 or newer
is recommended.

## Data layout

Pass the corresponding analysis-data directory explicitly when running either
benchmark workflow:

```text
paper_supplement/
├── widely_targeted/
└── untargeted/
```

### Widely targeted benchmark

```bash
pip install -r widely_targeted/requirements.txt
python widely_targeted/run_all.py \
  --data-dir /path/to/paper_supplement/widely_targeted \
  --output-dir results/widely_targeted
```

### Untargeted benchmark

```bash
pip install -r untargeted/requirements.txt
python untargeted/run_all.py \
  --data-dir /path/to/paper_supplement/untargeted \
  --output-dir results/untargeted
```

Both workflows verify input checksums before writing calculated tables and
figures.

### Training-data preprocessing

See `training/README.md` for dataset validation, resizing, splitting, and
horizontal augmentation commands.

