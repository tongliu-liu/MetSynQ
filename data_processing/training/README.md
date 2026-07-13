# Training-data Preprocessing

These scripts preprocess annotated LC-MS chromatogram images before YOLO
training.

## Workflow

1. `split_dataset.py` validates image-label pairs, resizes images to 224 x 224
   pixels, performs a reproducible stratified 8:1:1 split, and reports the
   class-distribution chi-square test.
2. `data_flipped.py` horizontally flips original images independently inside
   the train, validation, and test subsets. It is idempotent and verifies that
   image, label, and peak counts double exactly.

Class IDs 0, 1, and 2 correspond to peak classes A, B, and C.

## Installation

```bash
pip install -r training/requirements.txt
```

## Usage

```bash
python training/split_dataset.py \
  --images-dir RAW/images \
  --labels-dir RAW/labels \
  --output-dir DATASET \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42

python training/data_flipped.py \
  --dataset-dir DATASET \
  --prefix flip_
```

For the reference dataset, 39,747 original images and 44,868 annotated peaks
become 79,494 images and 89,736 peaks after augmentation. The expected final
image counts are 63,594 training, 7,950 validation, and 7,950 test images.

`my_coco_test.yaml` is an environment-specific training configuration. Update
its dataset root before use; neither preprocessing script reads this file.

