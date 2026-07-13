"""Create a reproducible stratified YOLO train/validation/test split.

The script expects one directory of original images and one directory of YOLO
label files. Images are resized to 224 x 224 while being copied. Because YOLO
box coordinates are normalized, resizing does not change the label values.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")
CLASS_IDS = (0, 1, 2)


@dataclass(frozen=True)
class Sample:
    """A validated image/label pair and its A/B/C peak-count signature."""

    stem: str
    image_path: Path
    label_path: Path
    class_counts: Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing original JPG/JPEG/PNG images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory containing YOLO TXT labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New dataset root for train/val/test outputs.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training-set ratio (default: 0.8).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation-set ratio (default: 0.1).")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test-set ratio (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--image-size", type=int, default=224, help="Square output image size in pixels (default: 224).")
    return parser.parse_args()


def discover_images(images_dir: Path) -> Dict[str, Path]:
    """Return one supported image path per filename stem."""

    images: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            raise ValueError(f"Duplicate image stem with multiple extensions: {path.stem}")
        images[path.stem] = path
    if not images:
        raise ValueError(f"No supported images found in: {images_dir}")
    return images


def discover_labels(labels_dir: Path) -> Dict[str, Path]:
    """Return one TXT label path per filename stem."""

    labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt")) if path.is_file()}
    if not labels:
        raise ValueError(f"No TXT labels found in: {labels_dir}")
    return labels


def parse_yolo_label(label_path: Path) -> Tuple[int, int, int]:
    """Validate a YOLO label file and count class 0/1/2 objects."""

    counts = [0, 0, 0]
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields, found {len(fields)}")
            try:
                class_value = float(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_number}: non-numeric YOLO value") from exc
            if not class_value.is_integer() or int(class_value) not in CLASS_IDS:
                raise ValueError(f"{label_path}:{line_number}: class ID must be 0, 1, or 2")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError(f"{label_path}:{line_number}: normalized coordinates must be within [0, 1]")
            counts[int(class_value)] += 1
    return tuple(counts)


def load_samples(images_dir: Path, labels_dir: Path) -> List[Sample]:
    """Load and validate all image/label pairs."""

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = discover_images(images_dir)
    labels = discover_labels(labels_dir)
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"images without labels: {missing_labels[:10]}")
        if missing_images:
            details.append(f"labels without images: {missing_images[:10]}")
        raise ValueError("Image/label pairing failed; " + "; ".join(details))

    return [
        Sample(stem, images[stem], labels[stem], parse_yolo_label(labels[stem]))
        for stem in sorted(images)
    ]


def validate_ratios(ratios: Sequence[float]) -> None:
    """Validate positive ratios that sum to one."""

    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, observed {sum(ratios):.12g}")


def largest_remainder_counts(total: int, ratios: Sequence[float]) -> List[int]:
    """Convert fractional split targets to exact integer counts."""

    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def allocate_strata(
    strata_sizes: Dict[Tuple[int, int, int], int],
    ratios: Sequence[float],
    global_targets: Sequence[int],
    rng: random.Random,
) -> Dict[Tuple[int, int, int], List[int]]:
    """Allocate every class-count signature proportionally and hit exact totals."""

    allocations: Dict[Tuple[int, int, int], List[int]] = {}
    ideals: Dict[Tuple[int, int, int], List[float]] = {}
    for signature in sorted(strata_sizes):
        size = strata_sizes[signature]
        raw = [size * ratio for ratio in ratios]
        counts = [int(value) for value in raw]
        remainder = size - sum(counts)
        tie_break = {index: rng.random() for index in range(len(ratios))}
        order = sorted(
            range(len(ratios)),
            key=lambda index: (-(raw[index] - counts[index]), tie_break[index]),
        )
        for index in order[:remainder]:
            counts[index] += 1
        allocations[signature] = counts
        ideals[signature] = raw

    current = [sum(counts[index] for counts in allocations.values()) for index in range(len(ratios))]
    while current != list(global_targets):
        over = [index for index in range(len(ratios)) if current[index] > global_targets[index]]
        under = [index for index in range(len(ratios)) if current[index] < global_targets[index]]
        candidates = []
        for source in over:
            for destination in under:
                for signature, counts in allocations.items():
                    if counts[source] == 0:
                        continue
                    ideal = ideals[signature]
                    before = (counts[source] - ideal[source]) ** 2 + (counts[destination] - ideal[destination]) ** 2
                    after = (counts[source] - 1 - ideal[source]) ** 2 + (counts[destination] + 1 - ideal[destination]) ** 2
                    candidates.append((after - before, rng.random(), source, destination, signature))
        if not candidates:
            raise RuntimeError("Unable to reconcile stratum allocations with global split totals")
        _, _, source, destination, signature = min(candidates)
        allocations[signature][source] -= 1
        allocations[signature][destination] += 1
        current[source] -= 1
        current[destination] += 1
    return allocations


def stratified_split(samples: Sequence[Sample], ratios: Sequence[float], seed: int) -> Dict[str, List[Sample]]:
    """Stratify by the full A/B/C object-count signature."""

    rng = random.Random(seed)
    strata: Dict[Tuple[int, int, int], List[Sample]] = {}
    for sample in samples:
        strata.setdefault(sample.class_counts, []).append(sample)
    for group in strata.values():
        rng.shuffle(group)

    targets = largest_remainder_counts(len(samples), ratios)
    allocations = allocate_strata(
        {signature: len(group) for signature, group in strata.items()}, ratios, targets, rng
    )
    result = {name: [] for name in SPLIT_NAMES}
    for signature in sorted(strata):
        group = strata[signature]
        offset = 0
        for split_index, split_name in enumerate(SPLIT_NAMES):
            count = allocations[signature][split_index]
            result[split_name].extend(group[offset : offset + count])
            offset += count
    for split_name in SPLIT_NAMES:
        rng.shuffle(result[split_name])
    observed = [len(result[name]) for name in SPLIT_NAMES]
    if observed != targets:
        raise AssertionError(f"Split-size mismatch: expected {targets}, observed {observed}")
    return result


def prepare_output(output_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Create an empty YOLO train/val/test directory tree."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    destinations = {}
    for split_name in SPLIT_NAMES:
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        destinations[split_name] = (images_dir, labels_dir)
    return destinations


def copy_and_resize(sample: Sample, image_destination: Path, label_destination: Path, image_size: int) -> None:
    """Copy one label and resize one image to the requested square dimensions."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required; install it with 'pip install Pillow'") from exc

    try:
        with Image.open(sample.image_path) as image:
            resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            resized.save(image_destination)
        with Image.open(image_destination) as written_image:
            if written_image.size != (image_size, image_size):
                raise OSError(
                    f"Written image has size {written_image.size}, expected "
                    f"{image_size}x{image_size}: {image_destination}"
                )
    except OSError as exc:
        raise OSError(f"Unable to process image {sample.image_path}: {exc}") from exc
    shutil.copy2(sample.label_path, label_destination)


def class_count_table(split_samples: Dict[str, Sequence[Sample]]) -> List[List[int]]:
    """Return a 3 x 3 table of split-by-class object counts."""

    table = []
    for split_name in SPLIT_NAMES:
        table.append(
            [sum(sample.class_counts[class_id] for sample in split_samples[split_name]) for class_id in CLASS_IDS]
        )
    return table


def chi_square_test(table: Sequence[Sequence[int]]) -> Tuple[float, float]:
    """Run the chi-square independence test on split-by-class peak counts."""

    if any(sum(row[class_id] for row in table) == 0 for class_id in CLASS_IDS):
        raise ValueError("Chi-square testing requires at least one peak from every class")
    try:
        from scipy.stats import chi2_contingency
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SciPy is required for the chi-square test") from exc
    chi2, p_value, _, _ = chi2_contingency(table)
    return float(chi2), float(p_value)


def write_manifest(output_dir: Path, split_samples: Dict[str, Sequence[Sample]]) -> None:
    """Write a deterministic record of every original sample assignment."""

    path = output_dir / "split_manifest.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "split", "class_0_peaks", "class_1_peaks", "class_2_peaks"])
        for split_name in SPLIT_NAMES:
            for sample in sorted(split_samples[split_name], key=lambda item: item.stem):
                writer.writerow([sample.stem, split_name, *sample.class_counts])


def main() -> None:
    """Run validation, stratification, resizing, copying, and distribution testing."""

    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    validate_ratios(ratios)
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than zero")

    samples = load_samples(args.images_dir.resolve(), args.labels_dir.resolve())
    split_samples = stratified_split(samples, ratios, args.seed)
    destinations = prepare_output(args.output_dir.resolve())

    for split_name in SPLIT_NAMES:
        images_dir, labels_dir = destinations[split_name]
        for sample in split_samples[split_name]:
            copy_and_resize(
                sample,
                images_dir / sample.image_path.name,
                labels_dir / sample.label_path.name,
                args.image_size,
            )

    write_manifest(args.output_dir.resolve(), split_samples)
    table = class_count_table(split_samples)
    chi2, p_value = chi_square_test(table)
    print("Dataset split completed successfully.")
    print("Split      Images   Class A   Class B   Class C   Total peaks")
    for split_name, counts in zip(SPLIT_NAMES, table):
        print(
            f"{split_name:<10} {len(split_samples[split_name]):>7} "
            f"{counts[0]:>9} {counts[1]:>9} {counts[2]:>9} {sum(counts):>13}"
        )
    print(f"Chi-square statistic: {chi2:.6f}")
    print(f"Chi-square p-value:   {p_value:.6f}")
    print(f"Manifest: {args.output_dir.resolve() / 'split_manifest.tsv'}")


if __name__ == "__main__":
    main()
