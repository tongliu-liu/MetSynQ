# -*- coding: utf-8 -*-
"""
Run xcms peak picking and convert the output to analysis input files.
"""

import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


RT_BOUNDARY_EPS_MIN = 1e-6


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run xcms peak extraction for Untargeted pipeline."
    )
    parser.add_argument(
        "--indir", required=True, help="Project directory containing mzML/.wiff data."
    )
    parser.add_argument(
        "--threads",
        default=16,
        type=int,
        help="Kept for compatibility; xcms runs serially here.",
    )
    parser.add_argument(
        "--ppm", default=15, type=float, help="CentWave ppm and ROI extraction ppm."
    )
    parser.add_argument(
        "--polarity",
        default="positive",
        choices=["positive", "negative"],
        help="Compatibility argument.",
    )
    parser.add_argument(
        "--minWidth",
        default=5.0,
        type=float,
        help="CentWave minimum peak width in seconds.",
    )
    parser.add_argument(
        "--maxWidth",
        default=50.0,
        type=float,
        help="CentWave maximum peak width in seconds.",
    )
    parser.add_argument(
        "--s2n", default=5.0, type=float, help="CentWave signal-to-noise threshold."
    )
    parser.add_argument(
        "--noise", default=100.0, type=float, help="CentWave noise threshold."
    )
    parser.add_argument("--mzDiff", default=0.015, type=float, help="CentWave mzdiff.")
    parser.add_argument(
        "--prefilter", default=3.0, type=float, help="CentWave prefilter scan count."
    )
    parser.add_argument(
        "--prefilterIntensity",
        default=100.0,
        type=float,
        help="CentWave prefilter intensity.",
    )
    parser.add_argument(
        "--rscript",
        default="/etc/anaconda3/envs/R4.1/bin/Rscript",
        help="Rscript executable.",
    )
    parser.add_argument(
        "--groupBw", default=5.0, type=float, help="PeakDensityParam bw in seconds."
    )
    parser.add_argument(
        "--obiwarpBinSize", default=1.0, type=float, help="ObiwarpParam binSize."
    )
    return parser.parse_args()


def require_columns(df, columns, path):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_xcms_outputs(xcms_dir):
    feature_path = xcms_dir / "xcms_feature_definitions.csv"
    peak_path = xcms_dir / "xcms_chrom_peaks.csv"
    assignment_path = xcms_dir / "xcms_peak_assignments.csv"

    features = pd.read_csv(feature_path)
    peaks = pd.read_csv(peak_path)
    assignments = pd.read_csv(assignment_path)

    require_columns(
        features, ["feature_id", "mzmed", "rtmed", "rtmin", "rtmax"], feature_path
    )
    require_columns(
        peaks,
        ["peak_index", "sampleID", "mz", "rt", "rtmin", "rtmax", "into"],
        peak_path,
    )
    require_columns(assignments, ["feature_id", "peak_index"], assignment_path)
    if features.empty or peaks.empty or assignments.empty:
        raise ValueError("xcms output is empty; cannot build analysis inputs.")

    return features, peaks, assignments


def numeric_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def list_mzml_files(mzml_dir):
    if not mzml_dir.is_dir():
        raise NotADirectoryError(f"mzML directory does not exist: {mzml_dir}")
    mzml_files = sorted(
        [
            path
            for path in mzml_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mzml"
        ],
        key=lambda path: str(path).lower(),
    )
    if not mzml_files:
        raise FileNotFoundError(f"No mzML files found in: {mzml_dir}")
    return mzml_files


def scan_mzml_rt_range(mzml_file):
    import pymzml

    rt_min = np.inf
    rt_max = -np.inf
    ms1_scans = 0
    with pymzml.run.Reader(str(mzml_file)) as run:
        for spec in run:
            if spec.ms_level != 1:
                continue
            scan_time = spec.scan_time
            if scan_time is None:
                continue
            rt_value = float(scan_time[0])
            rt_unit = str(scan_time[1]).lower() if len(scan_time) > 1 else "minute"
            if "second" in rt_unit:
                rt_value = rt_value / 60.0
            rt_min = min(rt_min, rt_value)
            rt_max = max(rt_max, rt_value)
            ms1_scans += 1

    if ms1_scans == 0 or not np.isfinite(rt_min) or not np.isfinite(rt_max):
        raise ValueError(f"No MS1 RT range could be read from mzML file: {mzml_file}")

    return {
        "sampleID": mzml_file.stem,
        "file": str(mzml_file),
        "rt_min": rt_min,
        "rt_max": rt_max,
        "ms1_scans": ms1_scans,
    }


def read_mzml_rt_ranges(mzml_dir):
    rt_ranges = [scan_mzml_rt_range(path) for path in list_mzml_files(mzml_dir)]
    return pd.DataFrame(rt_ranges)


def filter_to_common_rt_range(
    peak_list, result_table, rt_ranges, xcms_dir, eps=RT_BOUNDARY_EPS_MIN
):
    common_rt_min = float(rt_ranges["rt_min"].max())
    common_rt_max = float(rt_ranges["rt_max"].min())
    if (
        not np.isfinite(common_rt_min)
        or not np.isfinite(common_rt_max)
        or common_rt_min >= common_rt_max
    ):
        raise ValueError(
            "Cannot determine a valid common mzML RT range for xcms conversion."
        )

    lower_bound = common_rt_min + eps
    upper_bound = common_rt_max - eps
    keep_mask = (peak_list["RT"] > lower_bound) & (peak_list["RT"] < upper_bound)
    filtered_peak_list = peak_list.loc[keep_mask].copy()
    dropped_features = peak_list.loc[~keep_mask].copy()

    dropped_features["common_rt_min"] = common_rt_min
    dropped_features["common_rt_max"] = common_rt_max
    dropped_features["drop_reason"] = "feature_RT_outside_common_mzML_RT_range"
    dropped_features.to_csv(
        xcms_dir / "dropped_rt_out_of_range_features.csv", index=False
    )

    if filtered_peak_list.empty:
        raise ValueError(
            "No xcms features remained inside the common mzML RT range "
            f"({common_rt_min:.12g}, {common_rt_max:.12g}) min."
        )

    kept_ids = set(filtered_peak_list["Compound Name"].astype(str))
    filtered_result_table = result_table[
        result_table["mw ID"].astype(str).isin(kept_ids)
    ].copy()
    if filtered_result_table.empty:
        raise ValueError(
            "No result.csv rows remained after filtering xcms features to the common mzML RT range."
        )

    stats = {
        "features_before_rt_filter": len(peak_list),
        "features_after_rt_filter": len(filtered_peak_list),
        "dropped_rt_out_of_range_features": len(dropped_features),
        "result_rows_before_rt_filter": len(result_table),
        "result_rows_after_rt_filter": len(filtered_result_table),
        "common_rt_min": common_rt_min,
        "common_rt_max": common_rt_max,
        "rt_boundary_epsilon_min": eps,
    }
    return (
        filtered_peak_list.reset_index(drop=True),
        filtered_result_table.reset_index(drop=True),
        stats,
    )


def build_peak_list(features):
    peak_list = pd.DataFrame(
        {
            "Compound Name": features["feature_id"].astype(str),
            "mz": pd.to_numeric(features["mzmed"], errors="coerce"),
            "RT": pd.to_numeric(features["rtmed"], errors="coerce") / 60.0,
        }
    )
    peak_list = peak_list.dropna(subset=["Compound Name", "mz", "RT"])
    if peak_list.empty:
        raise ValueError("No valid xcms features remained after mz/RT cleanup.")
    return peak_list


def build_result_table(features, peaks, assignments):
    feature_lookup = features[["feature_id", "mzmed", "rtmed", "rtmin", "rtmax"]].copy()
    merged = assignments.merge(
        peaks, on="peak_index", how="inner", suffixes=("", "_peak")
    )
    merged = merged.merge(
        feature_lookup, on="feature_id", how="left", suffixes=("", "_feature")
    )
    if merged.empty:
        raise ValueError("No chrom peaks could be assigned to xcms features.")

    area = numeric_series(merged, "into")
    height = numeric_series(merged, "maxo", default=np.nan)
    if height.isna().all():
        height = area.copy()
    height = height.fillna(area)

    rt = numeric_series(merged, "rt") / 60.0
    rtmin = numeric_series(merged, "rtmin") / 60.0
    rtmax = numeric_series(merged, "rtmax") / 60.0
    feature_rt = numeric_series(merged, "rtmed") / 60.0

    result = pd.DataFrame(
        {
            "sampleID": merged["sampleID"].astype(str),
            "mw ID": merged["feature_id"].astype(str),
            "mz": numeric_series(merged, "mz"),
            "rt": rt,
            "rtmin": rtmin,
            "rtmax": rtmax,
            "Width": rtmax - rtmin,
            "int": height,
            "area": area,
            "sn": numeric_series(merged, "sn"),
            "sn_2": 0.0,
            "sn_3": 0.0,
            "sn_5": 0.0,
            "points": 0,
            "signal_points": 0,
            "class": "A",
            "conf": 1.0,
            "baseline": numeric_series(merged, "intb"),
            "min_rt": numeric_series(merged, "rtmin_feature") / 60.0,
            "max_rt": numeric_series(merged, "rtmax_feature") / 60.0,
            "lr_diff": 0.0,
            "rt_theoretic": feature_rt,
        }
    )

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=["sampleID", "mw ID", "mz", "rt", "rtmin", "rtmax", "area", "int"]
    )
    result = result[result["rtmin"] <= result["rtmax"]]
    if result.empty:
        raise ValueError("No valid result rows remained after xcms conversion.")

    result = result.sort_values(
        by=["mw ID", "sampleID", "area", "sn"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    result = result.drop_duplicates(subset=["mw ID", "sampleID"], keep="first")
    return result.reset_index(drop=True)


def write_pipeline_files(indir, peak_list, result_table):
    temp_dir = indir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    peak_list.to_csv(indir / "peak_list.csv", index=False)
    all_ions = peak_list[["RT", "Compound Name"]].rename(
        columns={"RT": "RT (min)", "Compound Name": "mw ID"}
    )
    all_ions.to_excel(indir / "ALL_ions.xlsx", index=False)
    result_table.to_csv(temp_dir / "result.csv", index=False)


def build_dataset(indir, peak_list, ppm):
    try:
        from extract_roi import extract_roi
    except ImportError:
        from utils.extract_roi import extract_roi

    mzml_dir = indir / "mzML"
    out_sample = extract_roi(str(mzml_dir), peak_list, ppm)
    if out_sample is None:
        raise RuntimeError("extract_roi returned no data for xcms peak list.")

    final_dataset = [[[s[0], s[1]], s[2], [x + 50 for x in s[3]]] for s in out_sample]
    with open(indir / "temp" / "dataset_all_samples.pkl", "wb") as f:
        pickle.dump(final_dataset, f)


def run_xcms(args, indir, xcms_dir):
    script_path = Path(__file__).resolve().with_name("xcms_peak_extraction.R")
    cmd = [
        args.rscript,
        str(script_path),
        "--data-dir",
        str(indir / "mzML"),
        "--out-dir",
        str(xcms_dir),
        "--ppm",
        str(args.ppm),
        "--min-width",
        str(args.minWidth),
        "--max-width",
        str(args.maxWidth),
        "--noise",
        str(args.noise),
        "--s2n",
        str(args.s2n),
        "--prefilter",
        str(args.prefilter),
        "--prefilter-intensity",
        str(args.prefilterIntensity),
        "--mz-diff",
        str(args.mzDiff),
        "--group-bw",
        str(args.groupBw),
        "--obiwarp-bin-size",
        str(args.obiwarpBinSize),
    ]
    print("[*] Running xcms peak extraction:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_arguments()
    indir = Path(args.indir).resolve()
    temp_dir = indir / "temp"
    xcms_dir = temp_dir / "xcms"
    temp_dir.mkdir(parents=True, exist_ok=True)
    xcms_dir.mkdir(parents=True, exist_ok=True)

    run_xcms(args, indir, xcms_dir)
    features, peaks, assignments = read_xcms_outputs(xcms_dir)

    peak_list = build_peak_list(features)
    result_table = build_result_table(features, peaks, assignments)
    rt_ranges = read_mzml_rt_ranges(indir / "mzML")
    peak_list, result_table, rt_filter_stats = filter_to_common_rt_range(
        peak_list, result_table, rt_ranges, xcms_dir,
    )
    write_pipeline_files(indir, peak_list, result_table)
    build_dataset(indir, peak_list, args.ppm)

    qc = [
        "xcms conversion QC",
        f"features: {len(peak_list)}",
        f"features_before_rt_filter: {rt_filter_stats['features_before_rt_filter']}",
        f"features_after_rt_filter: {rt_filter_stats['features_after_rt_filter']}",
        f"dropped_rt_out_of_range_features: {rt_filter_stats['dropped_rt_out_of_range_features']}",
        f"result_rows_before_rt_filter: {rt_filter_stats['result_rows_before_rt_filter']}",
        f"result_rows_after_rt_filter: {rt_filter_stats['result_rows_after_rt_filter']}",
        f"common_rt_min: {rt_filter_stats['common_rt_min']}",
        f"common_rt_max: {rt_filter_stats['common_rt_max']}",
        f"rt_boundary_epsilon_min: {rt_filter_stats['rt_boundary_epsilon_min']}",
        f"result_rows: {len(result_table)}",
        f"samples: {result_table['sampleID'].nunique()}",
        f"result_csv: {temp_dir / 'result.csv'}",
        f"peak_list_csv: {indir / 'peak_list.csv'}",
        f"all_ions_xlsx: {indir / 'ALL_ions.xlsx'}",
        f"dataset_pkl: {temp_dir / 'dataset_all_samples.pkl'}",
        f"dropped_rt_out_of_range_features_csv: {xcms_dir / 'dropped_rt_out_of_range_features.csv'}",
        "",
        "sample_rt_ranges:",
        "sampleID\trt_min\trt_max\tms1_scans\tfile",
    ]
    for row in rt_ranges.itertuples(index=False):
        qc.append(
            f"{row.sampleID}\t{row.rt_min}\t{row.rt_max}\t{row.ms1_scans}\t{row.file}"
        )
    (xcms_dir / "conversion_qc.txt").write_text("\n".join(qc) + "\n", encoding="utf-8")
    print("[*] xcms peak extraction converted successfully.")


if __name__ == "__main__":
    main()
