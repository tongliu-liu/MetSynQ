"""Combine qualitative and quantitative error IDs into the final AB sum."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_feature_set, read_table, sorted_feature_ids, write_tsv


DEFAULT_ORDER = ["MetSynQ", "Quanformer", "PeakOnly", "XCMS"]


def error_ids_from_file(path: Path) -> set[str]:
    df = read_table(path)
    if "feature_id" not in df.columns:
        raise ValueError(f"Expected a feature_id column in {path}")
    return set(df["feature_id"].dropna().astype(str).str.strip())


def summarize_errors(
    output_dir: Path,
    methods: list[str],
    evaluated_features_path: Path,
    unavailable_methods: set[str] | None = None,
    software_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, set[str]]]]:
    unavailable_methods = unavailable_methods or set()
    software_order = software_order or DEFAULT_ORDER
    evaluated_features = read_feature_set(evaluated_features_path)
    total_features = len(evaluated_features)

    sets_by_method: dict[str, dict[str, set[str]]] = {}
    summary_rows: list[dict[str, object]] = []
    final_id_rows: list[dict[str, object]] = []

    for software in software_order:
        if software in unavailable_methods:
            summary_rows.append(
                {
                    "software": software,
                    "qualitative_error_count": pd.NA,
                    "quantitative_error_count": pd.NA,
                    "overlap_error_count": pd.NA,
                    "unique_error_count": pd.NA,
                    "evaluated_feature_entries": total_features,
                    "total_error_rate": pd.NA,
                    "total_error_rate_percent": pd.NA,
                    "status": "not_available",
                }
            )
            continue
        if software not in methods:
            continue

        qualitative = error_ids_from_file(
            output_dir / f"{software}_qualitative_error_ids.tsv"
        )
        quantitative = error_ids_from_file(
            output_dir / f"{software}_quantitative_error_ids.tsv"
        )
        overlap = qualitative & quantitative
        union = qualitative | quantitative
        sets_by_method[software] = {
            "qualitative": qualitative,
            "quantitative": quantitative,
            "overlap": overlap,
            "union": union,
        }

        error_rate = len(union) / total_features
        summary_rows.append(
            {
                "software": software,
                "qualitative_error_count": len(qualitative),
                "quantitative_error_count": len(quantitative),
                "overlap_error_count": len(overlap),
                "unique_error_count": len(union),
                "evaluated_feature_entries": total_features,
                "total_error_rate": error_rate,
                "total_error_rate_percent": round(error_rate * 100, 2),
                "status": "calculated",
            }
        )

        for feature_id in sorted_feature_ids(union):
            is_qualitative = feature_id in qualitative
            is_quantitative = feature_id in quantitative
            source = "both" if is_qualitative and is_quantitative else (
                "qualitative" if is_qualitative else "quantitative"
            )
            final_id_rows.append(
                {
                    "software": software,
                    "feature_id": feature_id,
                    "qualitative_error": int(is_qualitative),
                    "quantitative_error": int(is_quantitative),
                    "error_source": source,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    count_columns = [
        "qualitative_error_count",
        "quantitative_error_count",
        "overlap_error_count",
        "unique_error_count",
        "evaluated_feature_entries",
    ]
    for column in count_columns:
        summary_df[column] = summary_df[column].astype("Int64")
    final_ids_df = pd.DataFrame(final_id_rows)
    write_tsv(summary_df, output_dir / "final_sum_summary.tsv")
    write_tsv(final_ids_df, output_dir / "final_error_ids.tsv")
    return summary_df, final_ids_df, sets_by_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final unique error features.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--evaluated-features", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["MetSynQ", "Quanformer", "XCMS"]
    )
    parser.add_argument("--unavailable-methods", nargs="+", default=["PeakOnly"])
    parser.add_argument("--software-order", nargs="+", default=DEFAULT_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_errors(
        args.input_dir,
        args.methods,
        args.evaluated_features,
        set(args.unavailable_methods),
        args.software_order,
    )


if __name__ == "__main__":
    main()

