"""Tests for the controlled false-positive data preparation workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from data_preparation.generate_false_positive_dataset import (
    OUTPUT_COLUMNS,
    collect_candidates,
    combine_evaluation_table,
    load_true_positives,
    sample_false_positives,
    validate_output_paths,
)


@dataclass
class FakeSpectrum:
    """Minimal pymzML-like spectrum used for filtering tests."""

    ms_level: int
    retention_time: float
    peak_values: list[tuple[float, float]]

    def scan_time_in_minutes(self) -> float:
        return self.retention_time

    def peaks(self, mode: str = "centroided") -> list[tuple[float, float]]:
        assert mode == "centroided"
        return self.peak_values


def test_collect_candidates_applies_ms1_mz_and_intensity_boundaries() -> None:
    boundary_peaks = []
    boundary_peaks.append((101.0, 100_000.0))
    boundary_peaks.append((100.999, 500.0))
    boundary_peaks.append((102.0, 100_000.1))
    boundary_peaks.append((103.0, 50.0))
    spectra = [
        FakeSpectrum(2, 1.0, [(110.0, 10.0)]),
        FakeSpectrum(1, 2.0, boundary_peaks),
        FakeSpectrum(1, 4.0, [(104.0, 25.0)]),
    ]

    candidates, rt_range = collect_candidates(
        spectra, true_mz=np.array([100.0]), min_mz_diff_da=1.0, max_intensity=100_000.0,
    )

    assert candidates["mz"].tolist() == [101.0, 103.0, 104.0]
    assert candidates["Intensity"].tolist() == [100_000.0, 50.0, 25.0]
    assert rt_range == (2.0, 4.0)


def test_collect_candidates_rejects_input_without_valid_ms1_scans() -> None:
    spectra = [FakeSpectrum(2, 1.0, [(110.0, 10.0)])]
    with pytest.raises(ValueError, match="no valid MS1 scans"):
        collect_candidates(
            spectra,
            true_mz=np.array([100.0]),
            min_mz_diff_da=1.0,
            max_intensity=100_000.0,
        )


def test_sampling_is_reproducible_and_reports_insufficient_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "mz": [101.0, 102.0, 103.0, 104.0],
            "RT": [1.0, 2.0, 3.0, 4.0],
            "Intensity": [10.0, 20.0, 30.0, 40.0],
        }
    )
    first = sample_false_positives(candidates, n_fp=3, seed=42)
    second = sample_false_positives(candidates, n_fp=3, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first["Compound Name"].tolist() == ["FP_1", "FP_2", "FP_3"]

    with pytest.raises(ValueError, match="eligible candidates"):
        sample_false_positives(candidates, n_fp=5, seed=42)


def test_true_positive_validation_and_combined_output_schema(tmp_path) -> None:
    missing_column_path = tmp_path / "missing_rt.csv"
    pd.DataFrame({"mz": [100.0]}).to_csv(missing_column_path, index=False)
    with pytest.raises(ValueError, match="missing required columns: RT"):
        load_true_positives(missing_column_path)

    valid_path = tmp_path / "valid.csv"
    pd.DataFrame({"mz": [100.0], "RT": [2.5]}).to_csv(valid_path, index=False)
    true_positives = load_true_positives(valid_path)
    assert true_positives["Compound Name"].tolist() == ["TP_1"]

    false_positives = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    false_positives.loc[0] = ["FP_1", 102.0, 3.0, 50.0]
    combined = combine_evaluation_table(true_positives, false_positives)
    assert combined.columns.tolist() == list(OUTPUT_COLUMNS)
    assert pd.isna(combined.loc[0, "Intensity"])


def test_existing_output_requires_explicit_overwrite(tmp_path) -> None:
    output_csv = tmp_path / "evaluation.csv"
    plot_output = tmp_path / "plot.png"
    output_csv.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        validate_output_paths(output_csv, plot_output, overwrite=False)
    validate_output_paths(output_csv, plot_output, overwrite=True)
