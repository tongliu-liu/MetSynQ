"""Calculate qualitative FP/FN errors for the AB benchmark.

The implementation preserves the calculation used for the author-confirmed
final result: rows missing in more than half of the eight samples are removed,
and a feature is considered detected when its area is non-missing and non-zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    clean_feature_ids,
    find_feature_column,
    read_feature_set,
    read_flagged_feature_set,
    read_table,
    sample_columns,
    sorted_feature_ids,
    write_tsv,
)


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOFTWARE=PATH.")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Software name cannot be empty.")
    return name.strip(), Path(path)


def calculate_for_software(
    software: str,
    path: Path,
    true_features: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], set[str]]:
    df = read_table(path)
    feature_column = find_feature_column(df)
    samples = sample_columns(df, feature_column)

    working = df.copy()
    working[feature_column] = clean_feature_ids(working[feature_column])
    if working[feature_column].isna().any() or (working[feature_column] == "").any():
        raise ValueError(f"{path} contains missing feature IDs.")
    if working[feature_column].duplicated().any():
        duplicated = working.loc[working[feature_column].duplicated(), feature_column].iloc[0]
        raise ValueError(f"{path} contains duplicate feature ID: {duplicated}")

    numeric = working[samples].apply(pd.to_numeric, errors="coerce")
    keep_mask = numeric.isna().sum(axis=1) <= len(samples) / 2
    filtered = working.loc[keep_mask, [feature_column]].copy()
    filtered_numeric = numeric.loc[keep_mask].copy()
    filtered_ids = filtered[feature_column].astype(str)

    metric_rows: list[dict[str, object]] = []
    error_types: dict[str, set[str]] = {}
    for sample in samples:
        values = filtered_numeric[sample]
        detected_mask = values.notna() & (values != 0)
        detected = set(filtered_ids.loc[detected_mask])
        tp_ids = detected & true_features
        fp_ids = detected - true_features
        fn_ids = true_features - detected

        for feature_id in fp_ids:
            error_types.setdefault(feature_id, set()).add("FP")
        for feature_id in fn_ids:
            error_types.setdefault(feature_id, set()).add("FN")

        tp = len(tp_ids)
        fp = len(fp_ids)
        fn = len(fn_ids)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metric_rows.append(
            {
                "software": software,
                "sample": sample,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )

    error_ids = set(error_types)
    error_rows = [
        {
            "software": software,
            "feature_id": feature_id,
            "qualitative_error_types": ";".join(sorted(error_types[feature_id])),
        }
        for feature_id in sorted_feature_ids(error_ids)
    ]
    qc = {
        "software": software,
        "qualitative_input_rows": len(df),
        "qualitative_rows_after_missing_filter": int(keep_mask.sum()),
        "qualitative_rows_removed_missing": int((~keep_mask).sum()),
        "qualitative_true_feature_count": len(true_features),
        "sample_count": len(samples),
        "qualitative_error_count": len(error_ids),
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(error_rows), qc, error_ids


def run_qualitative(
    true_features_path: Path,
    software_tables: list[tuple[str, Path]],
    output_dir: Path,
    evaluated_features_path: Path | None = None,
    truth_flag_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    reference_features = read_feature_set(true_features_path)
    if (evaluated_features_path is None) != (truth_flag_column is None):
        raise ValueError(
            "evaluated_features_path and truth_flag_column must be provided together."
        )
    true_features = reference_features
    if evaluated_features_path is not None and truth_flag_column is not None:
        true_features = read_flagged_feature_set(
            evaluated_features_path, truth_flag_column
        )
        outside_reference = true_features - reference_features
        if outside_reference:
            raise ValueError(
                "Flagged qualitative features are missing from the reference table: "
                + ", ".join(sorted_feature_ids(outside_reference)[:3])
            )
    all_metrics: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    error_sets: dict[str, set[str]] = {}

    for software, path in software_tables:
        metrics, errors, qc, error_ids = calculate_for_software(
            software, path, true_features
        )
        write_tsv(errors, output_dir / f"{software}_qualitative_error_ids.tsv")
        all_metrics.append(metrics)
        qc_rows.append(qc)
        error_sets[software] = error_ids

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)
    write_tsv(metrics_df, output_dir / "qualitative_metrics.tsv")
    write_tsv(qc_df, output_dir / "qualitative_calculation_qc.tsv")
    return metrics_df, qc_df, error_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate qualitative FP/FN errors.")
    parser.add_argument("--true-features", type=Path, required=True)
    parser.add_argument("--evaluated-features", type=Path)
    parser.add_argument(
        "--truth-flag-column",
        help="Binary column selecting qualitative true features.",
    )
    parser.add_argument(
        "--software-table",
        type=parse_name_path,
        action="append",
        required=True,
        help="Software table as SOFTWARE=PATH. Repeat for multiple software tables.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_qualitative(
        args.true_features,
        args.software_table,
        args.output_dir,
        args.evaluated_features,
        args.truth_flag_column,
    )


if __name__ == "__main__":
    main()
