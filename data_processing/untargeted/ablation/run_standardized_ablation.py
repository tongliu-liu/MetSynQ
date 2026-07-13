"""Score one dataset's pre-standardized ablation matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from calculate_qualitative_errors import run_qualitative
from calculate_quantitative_errors import run_quantitative
from common import write_tsv
from plot_ablation_comparison import generate_ablation_figures
from summarize_final_errors import summarize_errors


VARIANTS = ["FullPipeline", "ReplaceAlignment", "ReplacePeakPicking"]
DISPLAY_NAMES = {
    "FullPipeline": "Full pipeline",
    "ReplaceAlignment": "Replace alignment",
    "ReplacePeakPicking": "Replace peak picking",
}


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["QE_HF", "TripleTOF6600"], required=True)
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser.parse_args()


def load_spec(data_dir: Path, dataset: str) -> dict[str, object]:
    with (data_dir / "datasets.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)[dataset]


def run_dataset(
    dataset: str, data_dir: Path, output_dir: Path, figure_dir: Path,
) -> pd.DataFrame:
    spec = load_spec(data_dir, dataset)
    full_pipeline = data_dir / spec["qualitative_tables"]["MetSynQ"]
    matrix_paths = {
        "FullPipeline": full_pipeline,
        **{
            variant: data_dir / relative_path
            for variant, relative_path in spec["ablation_tables"].items()
        },
    }
    required = [
        *matrix_paths.values(),
        data_dir / spec["true_features"],
        data_dir / spec["true_fold_changes"],
        data_dir / spec["evaluated_features"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing standardized ablation inputs:\n" + "\n".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = [(variant, matrix_paths[variant]) for variant in VARIANTS]
    _, qualitative_qc, _ = run_qualitative(
        data_dir / spec["true_features"],
        tables,
        output_dir,
        data_dir / spec["evaluated_features"],
        str(spec["qualitative_truth_flag"]),
    )
    _, quantitative_qc, _ = run_quantitative(
        data_dir / spec["true_fold_changes"],
        tables,
        output_dir,
        group_a_token=str(spec["group_a_token"]),
        group_b_token=str(spec["group_b_token"]),
    )
    final_summary, _, _ = summarize_errors(
        output_dir,
        VARIANTS,
        data_dir / spec["evaluated_features"],
        unavailable_methods=set(),
        software_order=VARIANTS,
    )
    summary = final_summary.rename(columns={"software": "variant"}).copy()
    summary.insert(1, "display_name", summary["variant"].map(DISPLAY_NAMES))
    write_tsv(summary, output_dir / "ablation_comparison_summary.tsv")

    qc = qualitative_qc.merge(quantitative_qc, on="software", how="outer")
    qc = qc.merge(
        final_summary[
            [
                "software",
                "overlap_error_count",
                "unique_error_count",
                "evaluated_feature_entries",
                "total_error_rate_percent",
            ]
        ],
        on="software",
        how="left",
    ).rename(columns={"software": "variant"})
    qc.insert(1, "display_name", qc["variant"].map(DISPLAY_NAMES))
    qc["input_stage"] = "standardized_input"
    write_tsv(qc, output_dir / "ablation_calculation_qc.tsv")

    generate_ablation_figures(
        output_dir / "ablation_comparison_summary.tsv", figure_dir
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_dataset(
        args.dataset,
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        args.figure_dir.resolve(),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
