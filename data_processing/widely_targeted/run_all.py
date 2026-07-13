"""Run the widely targeted benchmark analysis and generate its outputs."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"Pandas requires version .* of 'numexpr'")

import pandas as pd

from figures import generate_all
from analysis import (
    InputPaths,
    calculate_alignment,
    calculate_boundary_iou,
    calculate_detection,
    calculate_integration,
    calculate_overall_errors,
    load_area_inputs,
    verify_source_manifest,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent
    package_dir = code_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=package_dir / "data",
        help="Directory containing the benchmark input TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_dir / "results",
        help="Output directory for tables and figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG export resolution.")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Recalculate tables without rendering figures.",
    )
    return parser.parse_args()


def _write_tables(
    output_dir: Path,
    manifest: pd.DataFrame,
    detection: dict[str, object],
    alignment: dict[str, object],
    integration: dict[str, object],
    boundary: dict[str, object],
    overall: dict[str, pd.DataFrame],
) -> None:
    tables = output_dir / "tables"
    write_tsv(manifest, tables / "source_manifest_check.tsv")
    write_tsv(detection["detection_table"], tables / "table_s1_detection_metrics.tsv")
    write_tsv(detection["long"], tables / "table_s1_detection_metrics_long.tsv")
    write_tsv(detection["summary"], tables / "table_s1_detection_summary.tsv")

    detection_error_rows = []
    for method in ["MetSynQ", "MultiQuant"]:
        detection_error_rows.append(
            {
                "Software": method,
                "Feature_FP": len(detection["feature_errors"][method]["FP"]),
                "Feature_FN": len(detection["feature_errors"][method]["FN"]),
                "FP_IDs": ";".join(sorted(detection["feature_errors"][method]["FP"])),
                "FN_IDs": ";".join(sorted(detection["feature_errors"][method]["FN"])),
            }
        )
    write_tsv(
        pd.DataFrame(detection_error_rows),
        tables / "table_s1_detection_feature_errors.tsv",
    )

    alignment_frame = alignment["summary"].copy()
    write_tsv(alignment_frame, tables / "table_s2_alignment.tsv")
    write_tsv(
        pd.DataFrame({"Component": sorted(alignment["common_feature_ids"])}),
        tables / "table_s2_alignment_common_feature_ids.tsv",
    )

    write_tsv(integration["cv_summary"], tables / "table_s3_peak_area_cv.tsv")
    write_tsv(integration["relative_summary"], tables / "table_s3_relative_error.tsv")
    write_tsv(integration["regression_summary"], tables / "table_s3_regression.tsv")
    write_tsv(
        pd.DataFrame({"Component": integration["common_feature_ids"]}),
        tables / "table_s3_common_feature_ids.tsv",
    )

    write_tsv(boundary["summary"], tables / "table_s4_boundary_iou.tsv")
    write_tsv(boundary["width_summary"], tables / "table_s4_boundary_width_qc.tsv")
    all_bad_rows = []
    for method, frame in boundary["error_rows"].items():
        all_bad_rows.append(frame.assign(Software=method))
    write_tsv(
        pd.concat(all_bad_rows, ignore_index=True),
        tables / "table_s4_iou_lt_0.8_peak_rows.tsv",
    )

    write_tsv(overall["summary"], tables / "table_s5_overall_error.tsv")
    write_tsv(
        overall["assignments"], tables / "table_s5_component_error_assignments.tsv"
    )

    notes = pd.DataFrame(
        [
            {
                "Item": "Detection reference override",
                "Value": "hso022-n is treated as absent for sample-level detection only",
                "Reason": "Applies the sample-specific detection evaluation rule",
            },
            {
                "Item": "MultiQuant detection filter",
                "Value": "Feature retained when at least 5 of 9 sample areas are non-missing",
                "Reason": "Requires non-missing values in at least 5 of 9 samples",
            },
            {
                "Item": "Overall error total",
                "Value": "FP + FN + union(NA difference, alignment error, boundary inconsistency)",
                "Reason": "Combines detection errors with the quantitative-stage error union",
            },
        ]
    )
    write_tsv(notes, tables / "methodology_notes.tsv")


def main() -> None:
    args = parse_args()
    paths = InputPaths(args.data_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = verify_source_manifest(paths)
    if not bool(manifest["passed"].all()):
        failed = manifest.loc[~manifest["passed"]]
        raise AssertionError(
            "Input manifest check failed:\n" + failed.to_string(index=False)
        )

    manual, metsynq, multiquant_detection, multiquant_integration = load_area_inputs(
        paths
    )
    detection = calculate_detection(paths, manual, metsynq, multiquant_detection)
    alignment = calculate_alignment(paths, manual, metsynq, multiquant_detection)
    integration = calculate_integration(manual, metsynq, multiquant_integration)
    boundary = calculate_boundary_iou(paths)
    overall = calculate_overall_errors(paths, detection, alignment, boundary)

    _write_tables(
        output_dir, manifest, detection, alignment, integration, boundary, overall
    )

    if not args.skip_figures:
        chart_map = generate_all(
            paths,
            detection,
            alignment,
            integration,
            boundary,
            output_dir / "figures" / "main",
            dpi=args.dpi,
        )
        write_tsv(chart_map, output_dir / "tables" / "chart_map.tsv")

    print("Widely targeted benchmark analysis completed successfully.")
    print("\nDetection summary:")
    print(detection["summary"].to_string(index=False))
    print("\nOverall error summary:")
    print(overall["summary"].to_string(index=False))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
