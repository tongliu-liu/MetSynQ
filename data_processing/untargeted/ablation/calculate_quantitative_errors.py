"""Calculate fold-change quantitative errors for the AB benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    clean_feature_ids,
    find_feature_column,
    read_table,
    sample_columns,
    write_tsv,
)


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOFTWARE=PATH.")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Software name cannot be empty.")
    return name.strip(), Path(path)


def read_true_fold_changes(path: Path) -> pd.Series:
    df = read_table(path)
    feature_column = find_feature_column(df)
    fold_column = next(
        (
            column
            for column in ["true_fold_change", "Fold change", "FoldChange"]
            if column in df.columns
        ),
        None,
    )
    if fold_column is None:
        raise ValueError(f"No true fold-change column found in {path}")
    working = pd.DataFrame(
        {
            "feature_id": clean_feature_ids(df[feature_column]),
            "true_fold_change": pd.to_numeric(df[fold_column], errors="coerce"),
        }
    ).dropna()
    if working["feature_id"].duplicated().any():
        raise ValueError(f"Duplicate feature IDs found in {path}")
    if (working["true_fold_change"] == 0).any():
        raise ValueError(f"Zero true fold change found in {path}")
    return working.set_index("feature_id")["true_fold_change"]


def calculate_for_software(
    software: str,
    path: Path,
    true_fold_changes: pd.Series,
    threshold: float,
    pseudocount: float,
    group_a_token: str = "SampleA",
    group_b_token: str = "SampleB",
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

    numeric = working[samples].apply(pd.to_numeric, errors="coerce").fillna(0)
    group_a = [column for column in samples if group_a_token in str(column)]
    group_b = [column for column in samples if group_b_token in str(column)]
    if not group_a or not group_b or len(group_a) != len(group_b):
        raise ValueError(
            f"{path} must contain equal non-zero SampleA and SampleB columns; "
            f"found {len(group_a)} and {len(group_b)}."
        )

    mean_a = numeric[group_a].mean(axis=1)
    mean_b = numeric[group_b].mean(axis=1)
    software_fc = (mean_b + pseudocount) / (mean_a + pseudocount)
    comparison = pd.DataFrame(
        {
            "feature_id": working[feature_column].astype(str),
            "software_fold_change": software_fc,
        }
    ).set_index("feature_id")
    comparison = comparison.join(
        true_fold_changes.rename("true_fold_change"), how="inner"
    ).dropna()
    comparison["absolute_relative_error"] = (
        comparison["software_fold_change"] - comparison["true_fold_change"]
    ).abs() / comparison["true_fold_change"].abs()
    comparison["software"] = software
    comparison["is_quantitative_error"] = (
        comparison["absolute_relative_error"] >= threshold
    )
    comparison = comparison.reset_index()

    errors = comparison.loc[
        comparison["is_quantitative_error"],
        [
            "software",
            "feature_id",
            "software_fold_change",
            "true_fold_change",
            "absolute_relative_error",
        ],
    ].copy()
    errors["_sort"] = errors["feature_id"].str.extract(r"(\d+)$")[0].astype(int)
    errors = errors.sort_values(["_sort", "feature_id"]).drop(columns="_sort")
    error_ids = set(errors["feature_id"].astype(str))
    qc = {
        "software": software,
        "quantitative_input_rows": len(df),
        "quantitative_compared_feature_count": len(comparison),
        "quantitative_error_count": len(error_ids),
        "quantitative_error_threshold": threshold,
        "pseudocount": pseudocount,
        "group_a_sample_count": len(group_a),
        "group_b_sample_count": len(group_b),
    }
    return comparison, errors, qc, error_ids


def run_quantitative(
    true_fold_changes_path: Path,
    software_tables: list[tuple[str, Path]],
    output_dir: Path,
    threshold: float = 0.2,
    pseudocount: float = 1e-6,
    group_a_token: str = "SampleA",
    group_b_token: str = "SampleB",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    true_fold_changes = read_true_fold_changes(true_fold_changes_path)
    all_comparisons: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    error_sets: dict[str, set[str]] = {}

    for software, path in software_tables:
        comparison, errors, qc, error_ids = calculate_for_software(
            software,
            path,
            true_fold_changes,
            threshold,
            pseudocount,
            group_a_token,
            group_b_token,
        )
        write_tsv(errors, output_dir / f"{software}_quantitative_error_ids.tsv")
        all_comparisons.append(comparison)
        qc_rows.append(qc)
        error_sets[software] = error_ids

    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)
    write_tsv(
        comparisons_df,
        output_dir / "quantitative_feature_comparison.tsv",
    )
    write_tsv(qc_df, output_dir / "quantitative_calculation_qc.tsv")
    return comparisons_df, qc_df, error_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate quantitative fold-change errors.")
    parser.add_argument("--true-fold-changes", type=Path, required=True)
    parser.add_argument(
        "--software-table",
        type=parse_name_path,
        action="append",
        required=True,
        help="Software table as SOFTWARE=PATH. Repeat for multiple software tables.",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_quantitative(
        args.true_fold_changes,
        args.software_table,
        args.output_dir,
        args.threshold,
        args.pseudocount,
    )


if __name__ == "__main__":
    main()
