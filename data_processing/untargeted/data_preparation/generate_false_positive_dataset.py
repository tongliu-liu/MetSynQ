"""Generate a controlled false-positive set from one centroided mzML file.

The script treats individual centroid peaks from MS1 scans as candidates. It
does not group peaks across scans into chromatographic features and does not
deduplicate candidates.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_N_FP = 836
DEFAULT_MIN_MZ_DIFF_DA = 1.0
DEFAULT_MAX_INTENSITY = 100_000.0
DEFAULT_SEED = 42
REQUIRED_TP_COLUMNS = ("mz", "RT")
OUTPUT_COLUMNS = ("Compound Name", "mz", "RT", "Intensity")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tp-file",
        type=Path,
        required=True,
        help="CSV containing the true-positive mz and RT columns.",
    )
    parser.add_argument(
        "--mzml-file",
        type=Path,
        required=True,
        help="One centroided mzML file from which MS1 peaks are extracted.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("TP_FP_eval_strict.csv"),
        help="Combined true-positive and false-positive CSV output.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("TP_FPs_mzRT.png"),
        help="Output path for the m/z-versus-RT plot.",
    )
    parser.add_argument(
        "--n-fp",
        type=int,
        default=DEFAULT_N_FP,
        help=f"Number of false positives to sample (default: {DEFAULT_N_FP}).",
    )
    parser.add_argument(
        "--min-mz-diff-da",
        type=float,
        default=DEFAULT_MIN_MZ_DIFF_DA,
        help=(
            "Minimum absolute m/z difference from every true positive in Da "
            f"(default: {DEFAULT_MIN_MZ_DIFF_DA})."
        ),
    )
    parser.add_argument(
        "--max-intensity",
        type=float,
        default=DEFAULT_MAX_INTENSITY,
        help=(
            "Maximum allowed candidate intensity, inclusive "
            f"(default: {DEFAULT_MAX_INTENSITY:g})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random sampling seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    return parser.parse_args()


def validate_parameters(n_fp: int, min_mz_diff_da: float, max_intensity: float) -> None:
    """Validate numeric filtering and sampling parameters."""
    if n_fp <= 0:
        raise ValueError("--n-fp must be a positive integer.")
    if not np.isfinite(min_mz_diff_da) or min_mz_diff_da <= 0:
        raise ValueError("--min-mz-diff-da must be a finite positive number.")
    if not np.isfinite(max_intensity) or max_intensity < 0:
        raise ValueError("--max-intensity must be a finite non-negative number.")


def validate_output_paths(output_csv: Path, plot_output: Path, overwrite: bool) -> None:
    """Reject ambiguous or unintended output replacement."""
    if output_csv.resolve() == plot_output.resolve():
        raise ValueError("--output-csv and --plot-output must be different files.")
    existing = [path for path in (output_csv, plot_output) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output already exists: {joined}. Use --overwrite to replace it."
        )


def load_true_positives(path: Path) -> pd.DataFrame:
    """Read and validate the true-positive peak list."""
    if not path.is_file():
        raise FileNotFoundError(f"True-positive CSV does not exist: {path}")
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read true-positive CSV {path}: {exc}") from exc

    missing = [column for column in REQUIRED_TP_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"True-positive CSV is missing required columns: {', '.join(missing)}"
        )
    if table.empty:
        raise ValueError("True-positive CSV contains no rows.")

    validated = table.copy()
    for column in REQUIRED_TP_COLUMNS:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"True-positive column '{column}' must contain only numeric values."
            ) from exc
        values = validated[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"True-positive column '{column}' contains non-finite values."
            )

    if (validated["mz"] <= 0).any():
        raise ValueError("True-positive m/z values must be greater than zero.")
    if (validated["RT"] < 0).any():
        raise ValueError("True-positive RT values must be non-negative.")

    if "Compound Name" not in validated.columns:
        validated["Compound Name"] = [
            f"TP_{index + 1}" for index in range(len(validated))
        ]
    elif validated["Compound Name"].isna().any():
        raise ValueError("True-positive 'Compound Name' values must not be missing.")
    validated["Compound Name"] = validated["Compound Name"].astype(str)
    return validated.loc[:, ["Compound Name", "mz", "RT"]].copy()


def open_mzml(path: Path) -> Iterable[Any]:
    """Open one mzML file with pymzML."""
    if not path.is_file():
        raise FileNotFoundError(f"mzML file does not exist: {path}")
    try:
        import pymzml
    except ImportError as exc:
        raise RuntimeError(
            "pymzML is required. Install dependencies from requirements.txt."
        ) from exc
    return pymzml.run.Reader(str(path))


def centroid_peaks(spectrum: Any) -> Iterator[tuple[float, float]]:
    """Yield numeric centroided m/z and intensity pairs from one spectrum."""
    try:
        peaks = spectrum.peaks("centroided")
    except TypeError:
        peaks = spectrum.peaks()
    for peak in peaks:
        if len(peak) < 2:
            continue
        try:
            mz = float(peak[0])
            intensity = float(peak[1])
        except (TypeError, ValueError):
            continue
        yield mz, intensity


def collect_candidates(
    spectra: Iterable[Any],
    true_mz: np.ndarray,
    min_mz_diff_da: float,
    max_intensity: float,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Collect qualifying scan-level centroid peaks from MS1 spectra."""
    true_mz = np.asarray(true_mz, dtype=float)
    if true_mz.size == 0:
        raise ValueError("At least one true-positive m/z value is required.")

    candidates: list[tuple[float, float, float]] = []
    ms1_retention_times: list[float] = []
    for spectrum in spectra:
        if getattr(spectrum, "ms_level", None) != 1:
            continue
        try:
            retention_time = float(spectrum.scan_time_in_minutes())
        except (TypeError, ValueError):
            continue
        if not np.isfinite(retention_time) or retention_time < 0:
            continue
        ms1_retention_times.append(retention_time)

        for mz, intensity in centroid_peaks(spectrum):
            if (
                not np.isfinite(mz)
                or not np.isfinite(intensity)
                or mz <= 0
                or intensity < 0
                or intensity > max_intensity
            ):
                continue
            if float(np.min(np.abs(mz - true_mz))) < min_mz_diff_da:
                continue
            candidates.append((mz, retention_time, intensity))

    if not ms1_retention_times:
        raise ValueError("The mzML input contains no valid MS1 scans.")
    rt_range = (min(ms1_retention_times), max(ms1_retention_times))
    table = pd.DataFrame(candidates, columns=["mz", "RT", "Intensity"])
    return table, rt_range


def sample_false_positives(
    candidates: pd.DataFrame, n_fp: int, seed: int
) -> pd.DataFrame:
    """Select a reproducible false-positive subset and assign stable names."""
    if len(candidates) < n_fp:
        raise ValueError(
            f"Only {len(candidates):,} eligible candidates were found; "
            f"{n_fp:,} are required. Use another mzML file or revise the thresholds."
        )
    sampled = candidates.sample(n=n_fp, random_state=seed).reset_index(drop=True)
    sampled.insert(
        0, "Compound Name", [f"FP_{index + 1}" for index in range(len(sampled))]
    )
    return sampled.loc[:, list(OUTPUT_COLUMNS)]


def combine_evaluation_table(
    true_positives: pd.DataFrame, false_positives: pd.DataFrame
) -> pd.DataFrame:
    """Combine TP and FP rows using the four-column output schema."""
    tp_output = true_positives.loc[:, ["Compound Name", "mz", "RT"]].copy()
    tp_output["Intensity"] = np.nan
    return pd.concat(
        [tp_output.loc[:, list(OUTPUT_COLUMNS)], false_positives], ignore_index=True,
    )


def save_mz_rt_plot(
    true_positives: pd.DataFrame, false_positives: pd.DataFrame, output_path: Path,
) -> None:
    """Save the TP/FP m/z-versus-RT scatter plot without opening a GUI."""
    figure, axis = plt.subplots(figsize=(6, 4), dpi=200)
    axis.scatter(
        true_positives["mz"],
        true_positives["RT"],
        s=10,
        color=(186 / 255, 147 / 255, 142 / 255),
        label="TP",
        alpha=0.9,
    )
    axis.scatter(
        false_positives["mz"],
        false_positives["RT"],
        s=10,
        color=(109 / 255, 159 / 255, 176 / 255),
        label="FP (noise)",
        alpha=0.7,
    )
    axis.set_xlabel("m/z")
    axis.set_ylabel("RT (min)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute validation, candidate extraction, sampling, and output writing."""
    validate_parameters(args.n_fp, args.min_mz_diff_da, args.max_intensity)
    validate_output_paths(args.output_csv, args.plot_output, args.overwrite)
    true_positives = load_true_positives(args.tp_file)
    spectra = open_mzml(args.mzml_file)
    candidates, rt_range = collect_candidates(
        spectra,
        true_positives["mz"].to_numpy(dtype=float),
        args.min_mz_diff_da,
        args.max_intensity,
    )
    LOGGER.info("MS1 acquisition RT range: %.4f-%.4f min", rt_range[0], rt_range[1])
    LOGGER.info("Eligible scan-level centroid peaks: %s", f"{len(candidates):,}")

    false_positives = sample_false_positives(candidates, args.n_fp, args.seed)
    evaluation_table = combine_evaluation_table(true_positives, false_positives)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_table.to_csv(args.output_csv, index=False)
    save_mz_rt_plot(true_positives, false_positives, args.plot_output)
    LOGGER.info("Evaluation table written to %s", args.output_csv.resolve())
    LOGGER.info("m/z-RT plot written to %s", args.plot_output.resolve())
    return evaluation_table, false_positives


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
