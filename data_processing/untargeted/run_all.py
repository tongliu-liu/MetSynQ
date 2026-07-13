"""Run the QE HF, TripleTOF 6600, and ablation benchmark analyses."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from figures import generate_all
from analysis import load_config, read_tsv, run_dataset, write_tsv


def parse_args() -> argparse.Namespace:
    code_dir = Path(__file__).resolve().parent
    package_dir = code_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=package_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=package_dir / "results")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-manifest-check", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_dir: Path) -> pd.DataFrame:
    manifest = read_tsv(data_dir / "source_manifest.tsv")
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        path = data_dir / record.file
        observed = sha256_file(path) if path.exists() else "MISSING"
        rows.append(
            {
                "file": record.file,
                "expected_sha256": record.sha256,
                "observed_sha256": observed,
                "passed": observed == record.sha256,
            }
        )
    return pd.DataFrame(rows)


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_ablation(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    code_dir = Path(__file__).resolve().parent
    ablation_code = code_dir / "ablation"
    tables_root = output_dir / "tables" / "ablation"
    figures_root = output_dir / "figures" / "ablation"

    def command_for(dataset: str) -> list[str]:
        return [
            sys.executable,
            str(ablation_code / "run_standardized_ablation.py"),
            "--dataset",
            dataset,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tables_root / dataset),
            "--figure-dir",
            str(figures_root / dataset),
        ]

    tof_command = command_for("TripleTOF6600")
    qe_command = command_for("QE_HF")
    run_command(qe_command, ablation_code)
    run_command(tof_command, ablation_code)

    rows: list[dict[str, object]] = []
    for dataset in ["QE_HF", "TripleTOF6600"]:
        summary = read_tsv(tables_root / dataset / "ablation_comparison_summary.tsv")
        label = "QE" if dataset == "QE_HF" else "TOF"
        for record in summary.itertuples(index=False):
            rows.append(
                {
                    "dataset": label,
                    "variant": record.variant,
                    "display_name": record.display_name,
                    "qualitative_error_count": int(record.qualitative_error_count),
                    "quantitative_error_count": int(record.quantitative_error_count),
                    "overlap_error_count": int(record.overlap_error_count),
                    "unique_error_count": int(record.unique_error_count),
                    "evaluated_feature_entries": int(record.evaluated_feature_entries),
                    "overall_accuracy": 1 - float(record.total_error_rate),
                    "overall_accuracy_percent": round(
                        (1 - float(record.total_error_rate)) * 100, 2
                    ),
                }
            )
    combined = pd.DataFrame(rows)
    combined_path = tables_root / "combined_ablation_accuracy.tsv"
    write_tsv(combined, combined_path)
    sys.path.insert(0, str(ablation_code))
    from plot_combined_ablation import plot_combined_accuracy

    plot_combined_accuracy(
        combined_path, figures_root / "combined_qe_tof_overall_accuracy",
    )
    return combined


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_manifest_check:
        manifest_check = verify_manifest(args.data_dir)
        write_tsv(manifest_check, args.output_dir / "tables" / "manifest_check.tsv")
        if not bool(manifest_check["passed"].all()):
            raise AssertionError("Attachment data manifest check failed.")

    configs = load_config(args.data_dir)
    results = {
        dataset: run_dataset(dataset, spec, args.data_dir, args.output_dir)
        for dataset, spec in configs.items()
    }
    font = generate_all(results, configs, args.output_dir)
    combined_main = pd.concat(
        [result["final_summary"] for result in results.values()], ignore_index=True
    )
    write_tsv(combined_main, args.output_dir / "tables" / "combined_main_summary.tsv")

    if not args.skip_ablation:
        run_ablation(args.data_dir, args.output_dir)

    print(combined_main.to_string(index=False))
    print(f"\nResults: {args.output_dir}")
    print(f"Figure font: {font}")


if __name__ == "__main__":
    main()
