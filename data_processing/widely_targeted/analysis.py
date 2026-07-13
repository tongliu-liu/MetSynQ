"""Core calculations for the widely targeted benchmark."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", message=r"Pandas requires version .* of 'numexpr'")

import numpy as np
import pandas as pd


SAMPLES = [f"mix{i:02d}" for i in range(7, 16)]
METHODS = ["MetSynQ", "MultiQuant"]
DETECTION_METRICS = ["TP", "FP", "FN", "Precision", "Recall", "F1"]


@dataclass(frozen=True)
class InputPaths:
    data_dir: Path

    @property
    def manual_area(self) -> Path:
        return self.data_dir / "manual_area.tsv"

    @property
    def metsynq_area(self) -> Path:
        return self.data_dir / "metsynq_area.tsv"

    @property
    def multiquant_detection(self) -> Path:
        return self.data_dir / "multiquant_detection_area.tsv"

    @property
    def multiquant_area(self) -> Path:
        return self.data_dir / "multiquant_integration_area.tsv"

    @property
    def detection_overrides(self) -> Path:
        return self.data_dir / "detection_reference_overrides.tsv"

    @property
    def boundary_universe(self) -> Path:
        return self.data_dir / "boundary_universe.tsv"

    @property
    def boundary_iou_errors(self) -> Path:
        return self.data_dir / "boundary_iou_errors.tsv"

    @property
    def evaluation_universe(self) -> Path:
        return self.data_dir / "evaluation_universe.tsv"

    @property
    def alignment_summary(self) -> Path:
        return self.data_dir / "alignment_summary.tsv"

    @property
    def alignment_error_ids(self) -> Path:
        return self.data_dir / "alignment_error_ids.tsv"

    @property
    def na_difference_ids(self) -> Path:
        return self.data_dir / "na_difference_ids.tsv"

    @property
    def cdf_curve_vertices(self) -> Path:
        return self.data_dir / "cdf_curve_vertices.tsv"

    @property
    def manifest(self) -> Path:
        return self.data_dir / "source_manifest.tsv"


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest(paths: InputPaths) -> pd.DataFrame:
    manifest = pd.read_csv(paths.manifest, sep="\t")
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        path = paths.data_dir / record.file
        exists = path.is_file()
        observed = sha256_file(path) if exists else "MISSING"
        rows.append(
            {
                "file": record.file,
                "role": record.role,
                "expected_sha256": record.sha256,
                "observed_sha256": observed,
                "bytes": path.stat().st_size if exists else np.nan,
                "passed": bool(exists and observed == record.sha256),
            }
        )
    return pd.DataFrame(rows)


def _read_area(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t").set_index("Component")
    frame.index = frame.index.astype(str)
    return frame.loc[:, SAMPLES].apply(pd.to_numeric, errors="coerce")


def load_area_inputs(
    paths: InputPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual = _read_area(paths.manual_area)
    metsynq = _read_area(paths.metsynq_area)
    multiquant_detection = _read_area(paths.multiquant_detection)
    multiquant_integration = _read_area(paths.multiquant_area)
    if (multiquant_detection.isna().sum(axis=1) > len(SAMPLES) / 2).any():
        raise AssertionError(
            "Minimal MultiQuant detection matrix violates the remove-half-NA rule"
        )
    return manual, metsynq, multiquant_detection, multiquant_integration


def _presence_counts(
    reference: pd.DataFrame, observed: pd.DataFrame, sample: str
) -> dict[str, float]:
    universe = reference.index.union(observed.index)
    truth = reference.reindex(universe)[sample].notna()
    prediction = observed.reindex(universe)[sample].notna()
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def calculate_detection(
    paths: InputPaths,
    manual: pd.DataFrame,
    metsynq: pd.DataFrame,
    multiquant_detection: pd.DataFrame,
) -> dict[str, object]:
    detection_reference = manual.copy()
    overrides = pd.read_csv(paths.detection_overrides, sep="\t")
    absent_ids = set(
        overrides.loc[
            overrides["Action"].eq("force_absent")
            & overrides["Scope"].eq("sample_detection"),
            "Component",
        ].astype(str)
    )
    for feature_id in absent_ids:
        if feature_id in detection_reference.index:
            detection_reference.loc[feature_id, :] = np.nan

    by_method: dict[str, pd.DataFrame] = {}
    for method, observed in [
        ("MetSynQ", metsynq),
        ("MultiQuant", multiquant_detection),
    ]:
        rows = []
        for sample in SAMPLES:
            row = {"Sample": sample}
            row.update(_presence_counts(detection_reference, observed, sample))
            rows.append(row)
        by_method[method] = pd.DataFrame(rows)

    detection_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        row: dict[str, object] = {"Sample": sample}
        for method in METHODS:
            record = by_method[method].set_index("Sample").loc[sample]
            for metric in DETECTION_METRICS:
                row[f"{method} {metric}"] = record[metric]
        detection_rows.append(row)

    for label, reducer in [("Mean", "mean"), ("STD", "std")]:
        row = {"Sample": label}
        for method in METHODS:
            for metric in ["TP", "FP", "FN"]:
                row[f"{method} {metric}"] = np.nan
            for metric in ["Precision", "Recall", "F1"]:
                series = by_method[method][metric]
                row[f"{method} {metric}"] = getattr(series, reducer)()
        detection_rows.append(row)

    feature_errors: dict[str, dict[str, set[str]]] = {}
    manual_ids = set(manual.index)
    for method, observed in [
        ("MetSynQ", metsynq),
        ("MultiQuant", multiquant_detection),
    ]:
        observed_ids = set(observed.index)
        feature_errors[method] = {
            "FP": observed_ids - manual_ids,
            "FN": manual_ids - observed_ids,
        }

    summary_rows = []
    for method in METHODS:
        for statistic in ["Mean", "STD"]:
            reducer = "mean" if statistic == "Mean" else "std"
            summary_rows.append(
                {
                    "Software": method,
                    "Statistic": statistic,
                    **{
                        metric: getattr(by_method[method][metric], reducer)()
                        for metric in ["Precision", "Recall", "F1"]
                    },
                }
            )

    return {
        "detection_table": pd.DataFrame(detection_rows),
        "long": pd.concat(
            [frame.assign(Software=method) for method, frame in by_method.items()],
            ignore_index=True,
        ),
        "summary": pd.DataFrame(summary_rows),
        "feature_errors": feature_errors,
        "reference": detection_reference,
    }


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return {
        "Median": float(np.median(array)),
        "Mean": float(np.mean(array)),
        "Q1": float(np.percentile(array, 25)),
        "Q3": float(np.percentile(array, 75)),
        "N": int(len(array)),
    }


def _calculate_cv(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for _, row in frame.iterrows():
        array = pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)
        if len(array) > 1 and np.mean(array) != 0:
            values.append(float(np.std(array, ddof=0) * 100 / np.mean(array)))
        else:
            values.append(0.0)
    return np.asarray(values)


def _per_feature_regression(
    reference: pd.DataFrame, observed: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    slopes: list[float] = []
    r2_values: list[float] = []
    for feature_id in reference.index:
        x = reference.loc[feature_id].to_numpy(dtype=float)
        y = observed.loc[feature_id].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 2:
            slopes.append(np.nan)
            r2_values.append(np.nan)
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        prediction = slope * x_valid + intercept
        ss_res = float(np.sum((y_valid - prediction) ** 2))
        ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
        slopes.append(float(slope))
        r2_values.append(float(1 - ss_res / ss_tot) if ss_tot else np.nan)
    return np.asarray(slopes), np.asarray(r2_values)


def calculate_integration(
    manual: pd.DataFrame, metsynq: pd.DataFrame, multiquant_integration: pd.DataFrame,
) -> dict[str, object]:
    common = manual.index.intersection(metsynq.index).intersection(
        multiquant_integration.index
    )
    common_manual = manual.loc[common, SAMPLES]
    common_ms = metsynq.loc[common, SAMPLES]
    common_mq = multiquant_integration.loc[common, SAMPLES]

    cv_values = {
        "Ground Truth": _calculate_cv(common_manual),
        "MetSynQ": _calculate_cv(common_ms),
        "MultiQuant": _calculate_cv(common_mq),
    }
    cv_summary = pd.DataFrame(
        [
            {"Software": method, **_numeric_summary(values)}
            for method, values in cv_values.items()
        ]
    )

    relative_errors: dict[str, np.ndarray] = {}
    relative_rows = []
    for method, observed in [
        ("MetSynQ", metsynq),
        ("MultiQuant", multiquant_integration),
    ]:
        pair_common = manual.index.intersection(observed.index)
        ref = manual.loc[pair_common, SAMPLES]
        pred = observed.loc[pair_common, SAMPLES]
        raw = ((ref - pred).abs() / ref).to_numpy().ravel()
        filtered = raw[np.isfinite(raw) & (raw <= 1)]
        relative_errors[method] = filtered
        relative_rows.append(
            {
                "Software": method,
                **_numeric_summary(filtered),
                "FeatureCount": len(pair_common),
                "ExcludedErrorAbove1": int(np.sum(raw > 1)),
            }
        )
    relative_summary = pd.DataFrame(relative_rows)

    regression_arrays: dict[str, dict[str, np.ndarray]] = {}
    regression_rows = []
    for method, observed in [("MetSynQ", common_ms), ("MultiQuant", common_mq)]:
        slopes, r2_values = _per_feature_regression(common_manual, observed)
        regression_arrays[method] = {"Slope": slopes, "R2": r2_values}
        regression_rows.extend(
            [
                {"Software / Metric": f"{method} R2", **_numeric_summary(r2_values)},
                {
                    "Software / Metric": f"{method} Slope (a)",
                    **_numeric_summary(slopes),
                },
            ]
        )

    return {
        "common_feature_ids": list(common),
        "cv_values": cv_values,
        "cv_summary": cv_summary,
        "relative_errors": relative_errors,
        "relative_summary": relative_summary,
        "regression_arrays": regression_arrays,
        "regression_summary": pd.DataFrame(regression_rows),
    }


def calculate_alignment(
    paths: InputPaths,
    manual: pd.DataFrame,
    metsynq: pd.DataFrame,
    multiquant_detection: pd.DataFrame,
) -> dict[str, object]:
    alignment_summary = pd.read_csv(paths.alignment_summary, sep="\t").set_index(
        "Software"
    )
    common_ids = (
        set(manual.index)
        & set(metsynq.index)
        & set(multiquant_detection.index) - {"IS001"}
    )
    if not common_ids:
        raise ValueError("The alignment evaluation scope is empty.")
    required_methods = {"MultiQuant", "MetSynQ"}
    if alignment_summary.index.has_duplicates:
        raise ValueError("Alignment summary software identifiers must be unique.")
    if not required_methods.issubset(set(alignment_summary.index.astype(str))):
        raise ValueError("Alignment summary must contain MultiQuant and MetSynQ rows.")
    error_table = pd.read_csv(paths.alignment_error_ids, sep="\t")

    def truth_mask(series: pd.Series) -> pd.Series:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})

    error_ids: dict[str, set[str]] = {}
    rows = []
    for method in ["MultiQuant", "MetSynQ"]:
        record = alignment_summary.loc[method]
        method_errors = error_table.loc[error_table["Software"].eq(method)].copy()
        for flag_column, summary_metric in [
            ("RT_SD_gt_0.05", "RT_SD_gt_0.05_count"),
            ("MaxDeltaRT_gt_0.1", "MaxDeltaRT_gt_0.1_count"),
        ]:
            threshold_count = (
                method_errors.loc[truth_mask(method_errors[flag_column]), "Component"]
                .astype(str)
                .nunique()
            )
            if threshold_count != int(record[summary_metric]):
                raise AssertionError(
                    f"{method} {flag_column} error-ID count does not match the summary"
                )
        ids = set(
            method_errors.loc[
                truth_mask(method_errors["ExceedAny"]), "Component"
            ].astype(str)
        )
        error_ids[method] = ids
        if len(ids) != int(record["Exceed_Any_Threshold"]):
            raise AssertionError(
                f"{method} alignment error-ID count does not match the summary"
            )
        rows.append(
            {
                "Software": method,
                "RT_SD_mean": float(record["RT_SD_mean"]),
                "RT_SD_gt_0.05_count": int(record["RT_SD_gt_0.05_count"]),
                "MaxDeltaRT_gt_0.1_count": int(record["MaxDeltaRT_gt_0.1_count"]),
                "Exceed_Any_Threshold": int(record["Exceed_Any_Threshold"]),
                "CommonFeatureCount": len(common_ids),
                "AlignmentErrorRate": len(ids) / len(common_ids),
            }
        )
    return {
        "summary": pd.DataFrame(rows),
        "error_ids": error_ids,
        "common_feature_ids": common_ids,
    }


def calculate_boundary_iou(paths: InputPaths) -> dict[str, object]:
    boundary_all = pd.read_csv(paths.boundary_universe, sep="\t")
    bad_all = pd.read_csv(paths.boundary_iou_errors, sep="\t")
    summary_rows = []
    width_rows = []
    error_ids: dict[str, set[str]] = {}
    error_rows: dict[str, pd.DataFrame] = {}
    for method in ["MultiQuant", "MetSynQ"]:
        boundary = boundary_all.loc[boundary_all["Software"].eq(method)].copy()
        bad = (
            bad_all.loc[bad_all["Software"].eq(method)].drop(columns="Software").copy()
        )
        if bad.duplicated(["Sample", "Component"]).any():
            raise AssertionError(f"{method} signal-IoU error rows are not unique")
        boundary_keys = set(
            zip(boundary["Sample"].astype(str), boundary["Component"].astype(str))
        )
        bad_keys = set(zip(bad["Sample"].astype(str), bad["Component"].astype(str)))
        if not bad_keys.issubset(boundary_keys):
            raise AssertionError(
                f"{method} signal-IoU error rows contain peaks outside the boundary universe"
            )
        total = len(boundary)
        inconsistent = len(bad)
        passing = total - inconsistent
        ids = set(bad["Component"].astype(str))
        error_ids[method] = ids
        error_rows[method] = bad
        summary_rows.append(
            {
                "Software": method,
                "TotalPeaks": total,
                "IOU_ge_0.8_Peaks": passing,
                "IOU_ge_0.8_Rate": passing / total,
                "IOU_lt_0.8_Peaks": inconsistent,
                "IOU_lt_0.8_Rate": inconsistent / total,
                "InconsistentFeatureCount": len(ids),
            }
        )
        width_rows.append(
            {
                "Software": method,
                **_numeric_summary(boundary["RelativeWidthDifference"]),
                "RelativeWidthDifference_gt_0.5_Count": int(
                    (boundary["RelativeWidthDifference"] > 0.5).sum()
                ),
            }
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "width_summary": pd.DataFrame(width_rows),
        "error_ids": error_ids,
        "error_rows": error_rows,
    }


def calculate_overall_errors(
    paths: InputPaths,
    detection: dict[str, object],
    alignment: dict[str, object],
    boundary: dict[str, object],
) -> dict[str, pd.DataFrame]:
    universe_table = pd.read_csv(paths.evaluation_universe, sep="\t")
    if "Component" not in universe_table.columns:
        raise ValueError("Evaluation universe must contain a Component column.")
    universe_values = universe_table["Component"].dropna().astype(str)
    if universe_values.empty:
        raise ValueError("The evaluation universe is empty.")
    if universe_values.duplicated().any():
        raise ValueError("Evaluation universe component identifiers must be unique.")
    universe = set(universe_values)
    na_table = pd.read_csv(paths.na_difference_ids, sep="\t")

    summary_rows = []
    assignment_rows = []
    for method in ["MultiQuant", "MetSynQ"]:
        fp = detection["feature_errors"][method]["FP"]
        fn = detection["feature_errors"][method]["FN"]
        na = set(na_table.loc[na_table["Software"].eq(method), "Component"].astype(str))
        align = alignment["error_ids"][method]
        iou = boundary["error_ids"][method]
        quantitative_union = na | align | iou
        all_unique = fp | fn | quantitative_union

        # Detection FP and FN counts are added to the union of the three
        # quantitative-error categories. Cross-stage overlaps are reported
        # separately rather than removed from this total.
        reported_error_count = len(fp) + len(fn) + len(quantitative_union)
        summary_rows.append(
            {
                "Software": method,
                "FP": len(fp),
                "FN": len(fn),
                "NA_Difference_Features": len(na),
                "Alignment_Error_Features": len(align),
                "Boundary_Inconsistent_Features": len(iou),
                "Quantitative_Union_Features": len(quantitative_union),
                "Reported_Error_Count": reported_error_count,
                "Total_Features": len(universe),
                "Reported_Total_Error_Rate": reported_error_count / len(universe),
                "Unique_Error_Count_All_Stages": len(all_unique),
                "Unique_Error_Rate_All_Stages": len(all_unique) / len(universe),
                "Cross_Stage_Overlap_Count": reported_error_count - len(all_unique),
            }
        )

        for component in sorted(all_unique):
            assignment_rows.append(
                {
                    "Software": method,
                    "Component": component,
                    "Detection_FP": component in fp,
                    "Detection_FN": component in fn,
                    "NA_Difference": component in na,
                    "Alignment_Error": component in align,
                    "Boundary_Inconsistent": component in iou,
                }
            )

    return {
        "summary": pd.DataFrame(summary_rows),
        "assignments": pd.DataFrame(assignment_rows),
    }
