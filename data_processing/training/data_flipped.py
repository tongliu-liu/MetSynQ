"""Horizontally flip every image in an already split YOLO dataset.

The dataset must contain ``train``, ``val``, and ``test`` directories, each
with ``images`` and ``labels`` subdirectories. Augmented files are written
back to the same subset so that no sample can cross a split boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUBSETS = ("train", "val", "test")
VALID_CLASS_IDS = {0, 1, 2}


@dataclass(frozen=True)
class YoloObject:
    """One validated YOLO bounding-box annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def horizontally_flipped(self) -> "YoloObject":
        """Return the annotation reflected across the vertical image axis."""
        return YoloObject(
            class_id=self.class_id,
            x_center=1.0 - self.x_center,
            y_center=self.y_center,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class SubsetSummary:
    """Counts collected before and after augmenting one dataset subset."""

    subset: str
    original_images: int
    final_images: int
    original_peaks: int
    final_peaks: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Horizontally flip the original images in the train, val, and test "
            "subsets of an already split YOLO dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root containing train/images, train/labels, and equivalent val/test paths.",
    )
    parser.add_argument(
        "--prefix",
        default="flip_",
        help="Filename prefix for augmented images and labels (default: flip_).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Required square image size in pixels (default: 224).",
    )
    return parser.parse_args()


def discover_images(images_dir: Path) -> dict[str, Path]:
    """Return supported images keyed by stem and reject duplicate stems."""
    images: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise ValueError(
                f"Duplicate image stem '{path.stem}' in {images_dir}: "
                f"{images[path.stem].name}, {path.name}"
            )
        images[path.stem] = path
    return images


def discover_labels(labels_dir: Path) -> dict[str, Path]:
    """Return TXT label files keyed by stem."""
    return {
        path.stem: path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def validate_pairs(
    images: dict[str, Path], labels: dict[str, Path], subset: str
) -> None:
    """Require a one-to-one image/label mapping within a subset."""
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        details: list[str] = []
        if missing_labels:
            details.append(f"images without labels: {missing_labels[:10]}")
        if missing_images:
            details.append(f"labels without images: {missing_images[:10]}")
        raise ValueError(f"Unpaired files in {subset}: " + "; ".join(details))


def read_yolo_label(path: Path) -> list[YoloObject]:
    """Read and validate a standard five-column normalized YOLO label file."""
    objects: list[YoloObject] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}:{line_number} must contain exactly five columns; "
                f"found {len(fields)}."
            )
        try:
            class_value = float(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number} contains a non-numeric YOLO value."
            ) from exc

        if not class_value.is_integer():
            raise ValueError(f"{path}:{line_number} has a non-integer class ID.")
        class_id = int(class_value)
        if class_id not in VALID_CLASS_IDS:
            raise ValueError(
                f"{path}:{line_number} has class ID {class_id}; expected 0, 1, or 2."
            )
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(
                f"{path}:{line_number} has coordinates outside the normalized [0, 1] range."
            )

        objects.append(YoloObject(class_id, *coordinates))
    return objects


def format_number(value: float) -> str:
    """Format a normalized coordinate compactly without losing useful precision."""
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.10f}".rstrip("0").rstrip(".")


def write_yolo_label(path: Path, objects: Sequence[YoloObject]) -> None:
    """Write annotations in standard YOLO format."""
    lines = [
        " ".join(
            (
                str(obj.class_id),
                format_number(obj.x_center),
                format_number(obj.y_center),
                format_number(obj.width),
                format_number(obj.height),
            )
        )
        for obj in objects
    ]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def load_pillow():
    """Import Pillow only after argument parsing so ``--help`` stays lightweight."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Install it with 'pip install Pillow'."
        ) from exc
    return Image


def validate_image(path: Path, image_size: int, image_module):
    """Read an image and require the expected 224 x 224 dimensions."""
    try:
        with image_module.open(path) as image:
            if image.size != (image_size, image_size):
                raise ValueError(
                    f"Image {path} is {image.size[0]}x{image.size[1]}; expected "
                    f"{image_size}x{image_size}. Run split_dataset.py first."
                )
            image.load()
            return image.copy()
    except OSError as exc:
        raise ValueError(f"Pillow could not read image {path}: {exc}") from exc


def annotations_match(
    actual: Sequence[YoloObject], expected: Sequence[YoloObject], tolerance: float = 1e-9
) -> bool:
    """Compare two annotation lists while allowing harmless decimal rounding."""
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if left.class_id != right.class_id:
            return False
        left_values = (left.x_center, left.y_center, left.width, left.height)
        right_values = (right.x_center, right.y_center, right.width, right.height)
        if any(abs(a - b) > tolerance for a, b in zip(left_values, right_values)):
            return False
    return True


def augment_subset(
    dataset_dir: Path, subset: str, prefix: str, image_size: int, image_module
) -> SubsetSummary:
    """Create or validate one horizontal-flip counterpart per original sample."""
    images_dir = dataset_dir / subset / "images"
    labels_dir = dataset_dir / subset / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected directories {images_dir} and {labels_dir}."
        )

    initial_images = discover_images(images_dir)
    initial_labels = discover_labels(labels_dir)
    validate_pairs(initial_images, initial_labels, subset)

    original_stems = sorted(stem for stem in initial_images if not stem.startswith(prefix))
    if not original_stems:
        raise ValueError(f"No original images found in the {subset} subset.")

    unexpected_augmented_stems = sorted(
        stem
        for stem in initial_images
        if stem.startswith(prefix) and stem[len(prefix) :] not in original_stems
    )
    if unexpected_augmented_stems:
        raise ValueError(
            f"Unexpected prefixed files in {subset}: {unexpected_augmented_stems[:10]}"
        )

    original_peak_count = 0
    created = 0
    skipped = 0
    for stem in original_stems:
        source_image_path = initial_images[stem]
        source_label_path = initial_labels[stem]
        source_image = validate_image(source_image_path, image_size, image_module)
        source_objects = read_yolo_label(source_label_path)
        original_peak_count += len(source_objects)
        flipped_objects = [obj.horizontally_flipped() for obj in source_objects]

        output_stem = f"{prefix}{stem}"
        output_image_path = images_dir / f"{output_stem}{source_image_path.suffix}"
        output_label_path = labels_dir / f"{output_stem}.txt"
        image_exists = output_image_path.exists()
        label_exists = output_label_path.exists()
        if image_exists != label_exists:
            raise ValueError(
                f"Incomplete augmented pair for {output_stem} in {subset}; "
                "remove or restore the unmatched file before retrying."
            )

        if image_exists:
            validate_image(output_image_path, image_size, image_module)
            existing_objects = read_yolo_label(output_label_path)
            if not annotations_match(existing_objects, flipped_objects):
                raise ValueError(
                    f"Existing augmented label does not match a horizontal flip: "
                    f"{output_label_path}"
                )
            skipped += 1
            continue

        flipped_image = source_image.transpose(image_module.Transpose.FLIP_LEFT_RIGHT)
        try:
            flipped_image.save(output_image_path)
        except OSError as exc:
            raise OSError(f"Failed to write flipped image {output_image_path}: {exc}") from exc
        write_yolo_label(output_label_path, flipped_objects)
        validate_image(output_image_path, image_size, image_module)
        created += 1

    final_images = discover_images(images_dir)
    final_labels = discover_labels(labels_dir)
    validate_pairs(final_images, final_labels, subset)
    final_peak_count = sum(len(read_yolo_label(path)) for path in final_labels.values())

    expected_final_samples = 2 * len(original_stems)
    if len(final_images) != expected_final_samples:
        raise RuntimeError(
            f"{subset} contains {len(final_images)} images after augmentation; "
            f"expected {expected_final_samples}."
        )
    if len(final_labels) != expected_final_samples:
        raise RuntimeError(
            f"{subset} contains {len(final_labels)} labels after augmentation; "
            f"expected {expected_final_samples}."
        )
    if final_peak_count != 2 * original_peak_count:
        raise RuntimeError(
            f"{subset} contains {final_peak_count} peaks after augmentation; "
            f"expected {2 * original_peak_count}."
        )

    print(
        f"{subset}: originals={len(original_stems):,}, created={created:,}, "
        f"already_present={skipped:,}, final_images={len(final_images):,}, "
        f"final_peaks={final_peak_count:,}"
    )
    return SubsetSummary(
        subset=subset,
        original_images=len(original_stems),
        final_images=len(final_images),
        original_peaks=original_peak_count,
        final_peaks=final_peak_count,
    )


def run(dataset_dir: Path, prefix: str, image_size: int) -> list[SubsetSummary]:
    """Augment all three subsets and verify that the full dataset doubles."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not prefix:
        raise ValueError("--prefix must not be empty.")
    if any(separator in prefix for separator in ("/", "\\")):
        raise ValueError("--prefix must be a filename prefix, not a path.")
    if image_size <= 0:
        raise ValueError("--image-size must be a positive integer.")

    image_module = load_pillow()
    summaries = [
        augment_subset(dataset_dir, subset, prefix, image_size, image_module)
        for subset in SUBSETS
    ]

    original_images = sum(summary.original_images for summary in summaries)
    final_images = sum(summary.final_images for summary in summaries)
    original_peaks = sum(summary.original_peaks for summary in summaries)
    final_peaks = sum(summary.final_peaks for summary in summaries)
    if final_images != 2 * original_images or final_peaks != 2 * original_peaks:
        raise RuntimeError("Full-dataset augmentation totals did not double exactly.")

    print("\nFull dataset verification")
    print(f"Images: {original_images:,} -> {final_images:,}")
    print(f"Peaks:  {original_peaks:,} -> {final_peaks:,}")
    return summaries


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run(args.dataset_dir, args.prefix, args.image_size)


if __name__ == "__main__":
    main()
