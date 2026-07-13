"""Shared QE HF and TripleTOF 6600 benchmark calculations.

All dataset-specific settings are defined in ``datasets.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def sorted_ids(values: set[str]) -> list[str]:
    return sorted(values, key=natural_key)


def load_config(data_dir: Path) -> dict[str, dict[str, Any]]:
    with (data_dir / "datasets.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = read_tsv(path)
    if "feature_id" not in frame.columns:
        raise ValueError(f"Missing feature_id column: {path}")
    frame["feature_id"] = frame["feature_id"].astype("string").str.strip()
    if frame["feature_id"].isna().any() or (frame["feature_id"] == "").any():
        raise ValueError(f"Missing feature ID: {path}")
    if frame["feature_id"].duplicated().any():
        duplicate = frame.loc[frame["feature_id"].duplicated(), "feature_id"].iloc[0]
        raise ValueError(f"Duplicate feature ID {duplicate}: {path}")
    samples = [column for column in frame.columns if column != "feature_id"]
    if len(samples) < 2 or len(samples) % 2:
        raise ValueError(f"Expected an even number of sample columns: {path}")
    return frame, samples


def _groups(
    samples: list[str], spec: dict[str, Any], path: Path
) -> tuple[list[str], list[str]]:
    group_a = [column for column in samples if spec["group_a_token"] in column]
    group_b = [column for column in samples if spec["group_b_token"] in column]
    if not group_a or len(group_a) != len(group_b):
        raise ValueError(
            f"Unequal A/B replicates in {path}: {len(group_a)} and {len(group_b)}"
        )
    return group_a, group_b


def _feature_set(path: Path) -> set[str]:
    frame = read_tsv(path)
    if "feature_id" not in frame.columns:
        raise ValueError(f"Missing feature_id column: {path}")
    return set(frame["feature_id"].dropna().astype(str).str.strip())


def run_qualitative(
    dataset: str, spec: dict[str, Any], data_dir: Path, output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    truth = _feature_set(data_dir / spec["true_features"])
    metric_rows: list[dict[str, object]] = []
    error_sets: dict[str, set[str]] = {}
    qc_rows: list[dict[str, object]] = []

    for method, relative_path in spec["qualitative_tables"].items():
        path = data_dir / relative_path
        frame, samples = _load_matrix(path)
        numeric = frame[samples].apply(pd.to_numeric, errors="coerce")
        keep = numeric.isna().sum(axis=1) <= len(samples) / 2
        filtered_ids = frame.loc[keep, "feature_id"].astype(str)
        filtered_numeric = numeric.loc[keep]
        error_types: dict[str, set[str]] = {}

        for sample in samples:
            detected_mask = filtered_numeric[sample].notna() & (
                filtered_numeric[sample] != 0
            )
            detected = set(filtered_ids.loc[detected_mask])
            tp_ids = detected & truth
            fp_ids = detected - truth
            fn_ids = truth - detected
            for feature_id in fp_ids:
                error_types.setdefault(feature_id, set()).add("FP")
            for feature_id in fn_ids:
                error_types.setdefault(feature_id, set()).add("FN")
            tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            metric_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "sample": sample,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                }
            )

        if spec.get("qualitative_error_scope", "sample_union") == "component_set":
            component_ids = set(filtered_ids.astype(str))
            component_error_types = {
                feature_id: {"FP"} for feature_id in component_ids - truth
            }
            component_error_types.update(
                {feature_id: {"FN"} for feature_id in truth - component_ids}
            )
            error_types = component_error_types
        errors = set(error_types)
        error_sets[method] = errors
        error_frame = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "method": method,
                    "feature_id": feature_id,
                    "qualitative_error_types": ";".join(
                        sorted(error_types[feature_id])
                    ),
                }
                for feature_id in sorted_ids(errors)
            ]
        )
        write_tsv(error_frame, output_dir / f"{method}_qualitative_error_ids.tsv")
        qc_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "input_rows": len(frame),
                "rows_after_missing_filter": int(keep.sum()),
                "rows_removed_missing": int((~keep).sum()),
                "true_feature_count": len(truth),
                "sample_count": len(samples),
                "qualitative_error_count": len(errors),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, object]] = []
    for method in spec["methods"]:
        if method not in error_sets:
            continue
        group = metrics.loc[metrics["method"] == method]
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "qualitative_error_count": len(error_sets[method]),
            "sample_count": len(group),
        }
        for metric in ["precision", "recall", "F1"]:
            key = metric.lower()
            row[f"{key}_mean"] = float(group[metric].mean())
            row[f"{key}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_tsv(metrics, output_dir / "qualitative_metrics.tsv")
    write_tsv(summary, output_dir / "qualitative_summary.tsv")
    write_tsv(pd.DataFrame(qc_rows), output_dir / "qualitative_qc.tsv")
    return metrics, summary, error_sets


def _true_quantification(path: Path) -> pd.DataFrame:
    frame = read_tsv(path)
    required = ["feature_id", "true_fold_change", "true_p_value"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing quantitative truth columns: {path}")
    frame = frame[required].copy()
    frame["feature_id"] = frame["feature_id"].astype(str).str.strip()
    frame["true_fold_change"] = pd.to_numeric(frame["true_fold_change"], errors="raise")
    frame["true_p_value"] = pd.to_numeric(frame["true_p_value"], errors="raise")
    return frame


def run_quantitative(
    dataset: str,
    spec: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    threshold: float = 0.2,
    pseudocount: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    truth = _true_quantification(data_dir / spec["true_fold_changes"]).set_index(
        "feature_id"
    )
    comparison_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    error_sets: dict[str, set[str]] = {}

    for method, relative_path in spec.get(
        "quantitative_tables", spec["analysis_tables"]
    ).items():
        path = data_dir / relative_path
        frame, samples = _load_matrix(path)
        group_a, group_b = _groups(samples, spec, path)
        numeric = frame[samples].apply(pd.to_numeric, errors="coerce")
        zero_policy = spec.get("quant_zero_policy", "zero")
        if zero_policy == "missing":
            numeric = numeric.replace(0, np.nan)
        elif zero_policy == "zero":
            numeric = numeric.fillna(0)
        else:
            raise ValueError(f"Unknown quant_zero_policy: {zero_policy}")
        fold_change = (numeric[group_b].mean(axis=1) + pseudocount) / (
            numeric[group_a].mean(axis=1) + pseudocount
        )
        comparison = pd.DataFrame(
            {
                "feature_id": frame["feature_id"].astype(str),
                "software_fold_change": fold_change,
            }
        ).set_index("feature_id")
        comparison = comparison.join(truth, how="inner").dropna()
        comparison["absolute_relative_error"] = (
            comparison["software_fold_change"] - comparison["true_fold_change"]
        ).abs() / comparison["true_fold_change"].abs()
        comparison["is_quantitative_error"] = (
            comparison["absolute_relative_error"] >= threshold
        )
        comparison.insert(0, "method", method)
        comparison.insert(0, "dataset", dataset)
        comparison = comparison.reset_index()
        comparison_frames.append(comparison)
        errors = set(
            comparison.loc[comparison["is_quantitative_error"], "feature_id"].astype(
                str
            )
        )
        error_sets[method] = errors
        error_frame = comparison.loc[
            comparison["is_quantitative_error"],
            [
                "dataset",
                "method",
                "feature_id",
                "software_fold_change",
                "true_fold_change",
                "absolute_relative_error",
            ],
        ].copy()
        error_frame["_sort"] = error_frame["feature_id"].map(natural_key)
        error_frame = error_frame.sort_values("_sort").drop(columns="_sort")
        write_tsv(error_frame, output_dir / f"{method}_quantitative_error_ids.tsv")
        qc_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "input_rows": len(frame),
                "compared_feature_count": len(comparison),
                "quantitative_error_count": len(errors),
                "quantitative_error_threshold": threshold,
                "zero_policy": zero_policy,
                "group_a_sample_count": len(group_a),
                "group_b_sample_count": len(group_b),
            }
        )

    comparisons = pd.concat(comparison_frames, ignore_index=True)
    qc = pd.DataFrame(qc_rows)
    write_tsv(comparisons, output_dir / "quantitative_feature_comparison.tsv")
    write_tsv(qc, output_dir / "quantitative_qc.tsv")
    return comparisons, qc, error_sets


def run_cv(
    dataset: str,
    spec: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    pseudocount: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    group_columns: dict[str, tuple[list[str], list[str]]] = {}
    for method in spec["cv_methods"]:
        path = data_dir / spec["analysis_tables"][method]
        frame, samples = _load_matrix(path)
        group_a, group_b = _groups(samples, spec, path)
        numeric = frame[samples].apply(pd.to_numeric, errors="coerce").fillna(0)
        numeric.index = frame["feature_id"].astype(str)
        matrices[method] = numeric
        group_columns[method] = (group_a, group_b)

    common: set[str] | None = None
    if spec.get("cv_scope") == "common":
        common = set.intersection(*(set(matrix.index) for matrix in matrices.values()))
    value_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method in spec["cv_methods"]:
        matrix = matrices[method]
        ids = sorted_ids(common if common is not None else set(matrix.index))
        matrix = matrix.loc[ids]
        for group_name, columns in zip(["A", "B"], group_columns[method]):
            values = matrix[columns].to_numpy(dtype=float)
            cvs = values.std(axis=1, ddof=0) * 100 / (values.mean(axis=1) + pseudocount)
            value_rows.extend(
                {
                    "dataset": dataset,
                    "method": method,
                    "sample_group": group_name,
                    "feature_id": feature_id,
                    "cv_percent": float(cv),
                }
                for feature_id, cv in zip(ids, cvs)
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "sample_group": group_name,
                    "feature_count": len(cvs),
                    "q25_percent": float(np.percentile(cvs, 25)),
                    "median_percent": float(np.median(cvs)),
                    "q75_percent": float(np.percentile(cvs, 75)),
                    "mean_percent": float(np.mean(cvs)),
                    "ddof": 0,
                    "missing_value_policy": "fill_with_zero",
                    "feature_scope": spec.get("cv_scope", "per_method"),
                }
            )
    values_frame = pd.DataFrame(value_rows)
    summary_frame = pd.DataFrame(summary_rows)
    write_tsv(values_frame, output_dir / "cv_feature_values.tsv")
    write_tsv(summary_frame, output_dir / "cv_summary.tsv")
    return values_frame, summary_frame


def _welch(values_b: np.ndarray, values_a: np.ndarray) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan
    with np.errstate(all="ignore"):
        return float(ttest_ind(values_b, values_a, equal_var=False).pvalue)


def run_differential(
    dataset: str,
    spec: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    pseudocount: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = _true_quantification(data_dir / spec["true_fold_changes"])
    up_inclusive = bool(spec.get("fold_up_inclusive", True))
    true_up = set(
        truth.loc[
            (
                (truth["true_fold_change"] >= 2)
                if up_inclusive
                else (truth["true_fold_change"] > 2)
            )
            & (truth["true_p_value"] < 0.05),
            "feature_id",
        ].astype(str)
    )
    true_down = set(
        truth.loc[
            (truth["true_fold_change"] < 0.5) & (truth["true_p_value"] < 0.05),
            "feature_id",
        ].astype(str)
    )
    true_all = true_up | true_down
    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for method, relative_path in spec["analysis_tables"].items():
        path = data_dir / relative_path
        frame, samples = _load_matrix(path)
        group_a, group_b = _groups(samples, spec, path)
        numeric = frame[samples].apply(pd.to_numeric, errors="coerce")
        detected_up: set[str] = set()
        detected_down: set[str] = set()
        for row_index, feature_id in enumerate(frame["feature_id"].astype(str)):
            values_a = numeric.loc[row_index, group_a].dropna().to_numpy(dtype=float)
            values_b = numeric.loc[row_index, group_b].dropna().to_numpy(dtype=float)
            if spec.get("differential_zero_policy") == "exclude":
                values_a = values_a[values_a != 0]
                values_b = values_b[values_b != 0]
            if not len(values_a) or not len(values_b):
                fold_change, p_value = np.nan, np.nan
            else:
                fold_change = (values_b.mean() + pseudocount) / (
                    values_a.mean() + pseudocount
                )
                p_value = _welch(values_b, values_a)
            is_up = bool(
                np.isfinite(fold_change)
                and np.isfinite(p_value)
                and ((fold_change >= 2) if up_inclusive else (fold_change > 2))
                and p_value < 0.05
            )
            is_down = bool(
                np.isfinite(fold_change)
                and np.isfinite(p_value)
                and fold_change < 0.5
                and p_value < 0.05
            )
            if is_up:
                detected_up.add(feature_id)
            if is_down:
                detected_down.add(feature_id)
            assignment_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "feature_id": feature_id,
                    "fold_change": fold_change,
                    "p_value": p_value,
                    "predicted_direction": "up"
                    if is_up
                    else ("down" if is_down else "not_differential"),
                    "true_direction": "up"
                    if feature_id in true_up
                    else ("down" if feature_id in true_down else "not_differential"),
                }
            )
        detected_all = detected_up | detected_down
        direction_mismatch = (detected_up & true_down) | (detected_down & true_up)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "input_feature_count": len(frame),
                "true_differential_count": len(true_all),
                "predicted_differential_count": len(detected_all),
                "identified_count": len(
                    (detected_up & true_up) | (detected_down & true_down)
                ),
                "false_positive_count": len(detected_all - true_all),
                "missed_count": len(true_all - detected_all),
                "direction_mismatch_count": len(direction_mismatch),
                "fold_up_rule": ">= 2" if up_inclusive else "> 2",
                "fold_down_rule": "< 0.5",
                "p_value_rule": "Welch p < 0.05",
                "multiple_testing_correction": "none",
                "zero_value_policy": spec.get("differential_zero_policy", "keep"),
            }
        )
    assignments = pd.DataFrame(assignment_rows)
    summary = pd.DataFrame(summary_rows)
    write_tsv(assignments, output_dir / "differential_feature_assignments.tsv")
    write_tsv(summary, output_dir / "differential_summary.tsv")
    return assignments, summary


def run_final_summary(
    dataset: str,
    spec: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    qualitative_errors: dict[str, set[str]],
    quantitative_errors: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluated = _feature_set(data_dir / spec["evaluated_features"])
    rows: list[dict[str, object]] = []
    id_rows: list[dict[str, object]] = []
    for method in spec["methods"]:
        if method not in qualitative_errors or method not in quantitative_errors:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "qualitative_error_count": pd.NA,
                    "quantitative_error_count": pd.NA,
                    "overlap_error_count": pd.NA,
                    "unique_error_count": pd.NA,
                    "evaluated_feature_entries": len(evaluated),
                    "overall_accuracy": pd.NA,
                    "status": "not_available",
                }
            )
            continue
        qualitative = qualitative_errors[method]
        quantitative = quantitative_errors[method]
        outside = (qualitative | quantitative) - evaluated
        if outside:
            raise ValueError(
                f"{dataset}/{method} error IDs outside evaluation universe: {sorted_ids(outside)[:3]}"
            )
        union = qualitative | quantitative
        for feature_id in sorted_ids(union):
            id_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "feature_id": feature_id,
                    "qualitative_error": int(feature_id in qualitative),
                    "quantitative_error": int(feature_id in quantitative),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "qualitative_error_count": len(qualitative),
                "quantitative_error_count": len(quantitative),
                "overlap_error_count": len(qualitative & quantitative),
                "unique_error_count": len(union),
                "evaluated_feature_entries": len(evaluated),
                "overall_accuracy": 1 - len(union) / len(evaluated),
                "status": "calculated",
            }
        )
    summary = pd.DataFrame(rows)
    summary["overall_accuracy_percent"] = (
        pd.to_numeric(summary["overall_accuracy"], errors="coerce") * 100
    )
    ids = pd.DataFrame(id_rows)
    write_tsv(summary, output_dir / "final_summary.tsv")
    write_tsv(ids, output_dir / "final_error_ids.tsv")
    return summary, ids


def run_dataset(
    dataset: str, spec: dict[str, Any], data_dir: Path, result_root: Path,
) -> dict[str, pd.DataFrame]:
    output_dir = result_root / "tables" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, qualitative_summary, qualitative_errors = run_qualitative(
        dataset, spec, data_dir, output_dir
    )
    comparisons, quantitative_qc, quantitative_errors = run_quantitative(
        dataset, spec, data_dir, output_dir
    )
    cv_values, cv_summary = run_cv(dataset, spec, data_dir, output_dir)
    differential_assignments, differential_summary = run_differential(
        dataset, spec, data_dir, output_dir
    )
    final_summary, final_ids = run_final_summary(
        dataset, spec, data_dir, output_dir, qualitative_errors, quantitative_errors,
    )
    quant_truth_count = len(_true_quantification(data_dir / spec["true_fold_changes"]))
    quantitative_accuracy = quantitative_qc[
        ["dataset", "method", "compared_feature_count", "quantitative_error_count"]
    ].copy()
    quantitative_accuracy["quantitative_truth_features"] = quant_truth_count
    quantitative_accuracy["quantitative_accuracy"] = (
        1 - quantitative_accuracy["quantitative_error_count"] / quant_truth_count
    )
    write_tsv(quantitative_accuracy, output_dir / "quantitative_accuracy_summary.tsv")
    return {
        "qualitative_metrics": metrics,
        "qualitative_summary": qualitative_summary,
        "quantitative_comparison": comparisons,
        "quantitative_accuracy": quantitative_accuracy,
        "cv_values": cv_values,
        "cv_summary": cv_summary,
        "differential_assignments": differential_assignments,
        "differential_summary": differential_summary,
        "final_summary": final_summary,
        "final_error_ids": final_ids,
    }
