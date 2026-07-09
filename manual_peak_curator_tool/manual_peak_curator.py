#!/usr/bin/env python
"""Manual LC-MS peak area curator for MetSynQ validation.

This tool does not perform feature discovery. It loads mzML files and a feature
list, displays MS1 EICs, lets users adjust RT integration boundaries, and
exports a curated area matrix plus an edit log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import sys
import threading
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


APP_VERSION = "0.1.0"
DEFAULT_PPM = 50.0
DEFAULT_RT_WINDOW = 0.8
DEFAULT_BOUNDARY_HALF_WIDTH = 0.12


def require_science_stack():
    try:
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import pymzml  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing runtime dependency. Use the project environment or install "
            "`numpy pandas pymzml matplotlib` before running this tool.\n"
            f"Original error: {exc}"
        ) from exc


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    alias_norm = {normalize_name(item) for item in aliases}
    for col in columns:
        if normalize_name(col) in alias_norm:
            return col
    return None


def file_sha1(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Feature:
    feature_id: str
    mz: float
    rt: float
    rtmin: Optional[float] = None
    rtmax: Optional[float] = None
    annotation: str = ""


@dataclass
class EIC:
    sample: str
    source_file: str
    mz: float
    ppm: float
    rt: List[float]
    intensity: List[float]


@dataclass
class Curation:
    feature_id: str
    sample: str
    mz: float
    rt_expected: float
    rt_left: float
    rt_right: float
    area: float = 0.0
    height: float = 0.0
    apex_rt: float = math.nan
    points: int = 0
    baseline_mode: str = "linear"
    edited: bool = False


class FeatureTableError(ValueError):
    pass


def read_feature_table(path: Path) -> List[Feature]:
    require_science_stack()
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        sep = "\t" if suffix in {".tsv", ".txt"} else None
        frame = pd.read_csv(path, sep=sep, engine="python")

    if frame.empty:
        raise FeatureTableError(f"Feature table is empty: {path}")

    columns = list(frame.columns)
    id_col = find_column(
        columns,
        [
            "feature_id",
            "feature id",
            "id",
            "name",
            "mw ID",
            "mwID",
            "component name",
            "Component_Name",
            "Row.names",
        ],
    )
    mz_col = find_column(
        columns,
        [
            "mz",
            "m/z",
            "mzmed",
            "mzmean",
            "mz_mean",
            "mz median",
            "observed m/z",
            "mass",
            "q1",
        ],
    )
    rt_col = find_column(
        columns,
        [
            "rt",
            "RT",
            "RT (min)",
            "retention time",
            "Retention_Time",
            "rtmed",
            "rtmean",
            "rt_mean",
            "rt median",
        ],
    )
    rtmin_col = find_column(columns, ["rtmin", "rt_min", "min_rt", "rt start", "left rt"])
    rtmax_col = find_column(columns, ["rtmax", "rt_max", "max_rt", "rt end", "right rt"])
    annotation_col = find_column(
        columns,
        ["annotation", "compound", "compound name", "metabolite", "name"],
    )

    if mz_col is None:
        raise FeatureTableError(
            "Feature table needs an m/z column. Accepted names include mz, m/z, mzmed, mzmean, mass, q1."
        )
    if rt_col is None and not (rtmin_col and rtmax_col):
        raise FeatureTableError(
            "Feature table needs an RT column or both rtmin and rtmax columns."
        )

    features: List[Feature] = []
    for idx, row in frame.iterrows():
        try:
            mz = float(row[mz_col])
            if rt_col is not None:
                rt = float(row[rt_col])
            else:
                rt = (float(row[rtmin_col]) + float(row[rtmax_col])) / 2.0
        except Exception:
            continue
        if not math.isfinite(mz) or not math.isfinite(rt):
            continue

        fid = str(row[id_col]).strip() if id_col else f"F{idx + 1}"
        if fid == "" or fid.lower() == "nan":
            fid = f"F{idx + 1}"

        rtmin = _float_or_none(row[rtmin_col]) if rtmin_col else None
        rtmax = _float_or_none(row[rtmax_col]) if rtmax_col else None
        annotation = ""
        if annotation_col:
            annotation = str(row[annotation_col]).strip()
            if annotation.lower() == "nan":
                annotation = ""
        features.append(Feature(fid, mz, rt, rtmin, rtmax, annotation))

    if not features:
        raise FeatureTableError(f"No valid feature rows were read from {path}")
    return features


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def mzml_files_from_dir(path: Path) -> List[Path]:
    files = sorted(path.glob("*.mzML")) + sorted(path.glob("*.mzml"))
    return sorted(set(files), key=lambda item: item.name.lower())


def sample_name_from_path(path: Path) -> str:
    name = path.name
    for suffix in [".mzML", ".mzml"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def spectrum_rt_minutes(spectrum: Any) -> Optional[float]:
    for attr in ["scan_time_in_minutes", "scanTimeInMinutes"]:
        method = getattr(spectrum, attr, None)
        if callable(method):
            try:
                return float(method())
            except Exception:
                pass
    value = getattr(spectrum, "scan_time", None)
    try:
        if callable(value):
            value = value()
        if isinstance(value, (tuple, list)):
            time = float(value[0])
            unit = str(value[1]).lower() if len(value) > 1 else "minute"
            if "second" in unit or unit == "s":
                return time / 60.0
            return time
        if value is not None:
            return float(value)
    except Exception:
        return None
    return None


def spectrum_peaks(spectrum: Any):
    for mode in ["centroided", "raw"]:
        try:
            peaks = spectrum.peaks(mode)
            if peaks is not None:
                return peaks
        except Exception:
            pass
    return []


def extract_eic(
    mzml_path: Path,
    mz: float,
    ppm: float,
    rt_center: Optional[float] = None,
    rt_window: Optional[float] = None,
    intensity_mode: str = "sum",
) -> EIC:
    require_science_stack()
    import numpy as np
    import pymzml

    mz_low = mz * (1.0 - ppm * 1e-6)
    mz_high = mz * (1.0 + ppm * 1e-6)
    rt_low = None
    rt_high = None
    if rt_center is not None and rt_window is not None:
        rt_low = rt_center - rt_window / 2.0
        rt_high = rt_center + rt_window / 2.0

    rts: List[float] = []
    intensities: List[float] = []
    # Sequential iteration does not need pymzml's offset index. Building it
    # forces a full extra XML scan per file, which is what made loading large
    # cohorts feel like a freeze.
    try:
        reader = pymzml.run.Reader(str(mzml_path))
    except Exception:
        reader = pymzml.run.Reader(str(mzml_path), build_index_from_scratch=True)
    for spectrum in reader:
        try:
            if int(spectrum.ms_level) != 1:
                continue
        except Exception:
            continue
        rt = spectrum_rt_minutes(spectrum)
        if rt is None:
            continue
        if rt_low is not None and (rt < rt_low or rt > rt_high):
            continue

        peaks = np.asarray(spectrum_peaks(spectrum))
        signal = 0.0
        if peaks.size:
            peaks = peaks.reshape((-1, 2))
            mask = (peaks[:, 0] >= mz_low) & (peaks[:, 0] <= mz_high)
            if mask.any():
                values = peaks[mask, 1]
                signal = float(np.max(values) if intensity_mode == "max" else np.sum(values))
        rts.append(float(rt))
        intensities.append(signal)

    order = np.argsort(rts)
    rt_sorted = [float(rts[i]) for i in order]
    int_sorted = [float(intensities[i]) for i in order]
    return EIC(sample_name_from_path(mzml_path), str(mzml_path), mz, ppm, rt_sorted, int_sorted)


def extract_eics_multi(
    mzml_path: Path,
    features: List[Feature],
    ppm: float,
    rt_window: float,
    intensity_mode: str = "sum",
) -> Dict[str, EIC]:
    """Extract EICs for many features from one mzML file in a single pass.

    The serial code path opens and re-parses every mzML once per feature, which
    is catastrophic when the cohort has many large files: parsing dominates,
    and we redo it N_features times. This routine pays the parse cost once and
    fans out the per-spectrum filtering across all features whose RT window
    intersects the spectrum, so total work is roughly the same as a single
    feature pass plus a small per-feature mask cost.
    """
    require_science_stack()
    import numpy as np
    import pymzml

    if not features:
        return {}

    sample = sample_name_from_path(mzml_path)
    source = str(mzml_path)
    half_window = rt_window / 2.0

    # Vectorised feature bounds make the per-spectrum relevance check cheap.
    mzs = np.array([f.mz for f in features], dtype=float)
    mz_lows = mzs * (1.0 - ppm * 1e-6)
    mz_highs = mzs * (1.0 + ppm * 1e-6)
    rt_lows = np.array([f.rt - half_window for f in features], dtype=float)
    rt_highs = np.array([f.rt + half_window for f in features], dtype=float)
    global_rt_low = float(rt_lows.min())
    global_rt_high = float(rt_highs.max())
    # mzML spectra are normally ordered by acquisition time, so once RT
    # exceeds the latest feature window we can stop reading. The 1-minute
    # margin guards against minor non-monotonicity at run boundaries.
    rt_break = global_rt_high + 1.0

    rts: List[List[float]] = [[] for _ in features]
    ints: List[List[float]] = [[] for _ in features]

    try:
        reader = pymzml.run.Reader(source)
    except Exception:
        reader = pymzml.run.Reader(source, build_index_from_scratch=True)

    use_max = intensity_mode == "max"
    for spectrum in reader:
        try:
            if int(spectrum.ms_level) != 1:
                continue
        except Exception:
            continue
        rt = spectrum_rt_minutes(spectrum)
        if rt is None:
            continue
        if rt > rt_break:
            break
        if rt < global_rt_low or rt > global_rt_high:
            continue

        relevant = np.where((rt_lows <= rt) & (rt <= rt_highs))[0]
        if relevant.size == 0:
            continue

        peaks = np.asarray(spectrum_peaks(spectrum))
        if peaks.size:
            peaks = peaks.reshape((-1, 2))
            peak_mz = peaks[:, 0]
            peak_int = peaks[:, 1]
            for i in relevant:
                mask = (peak_mz >= mz_lows[i]) & (peak_mz <= mz_highs[i])
                if mask.any():
                    vals = peak_int[mask]
                    signal = float(np.max(vals) if use_max else np.sum(vals))
                else:
                    signal = 0.0
                rts[i].append(float(rt))
                ints[i].append(signal)
        else:
            for i in relevant:
                rts[i].append(float(rt))
                ints[i].append(0.0)

    eics: Dict[str, EIC] = {}
    for i, f in enumerate(features):
        rt_arr = rts[i]
        int_arr = ints[i]
        if rt_arr:
            order = np.argsort(rt_arr)
            rt_sorted = [float(rt_arr[k]) for k in order]
            int_sorted = [float(int_arr[k]) for k in order]
        else:
            rt_sorted = []
            int_sorted = []
        eics[f.feature_id] = EIC(sample, source, f.mz, ppm, rt_sorted, int_sorted)
    return eics


def _write_edit_log_csv(path: Path, edit_log: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestamp",
        "action",
        "feature_id",
        "sample",
        "before_rt_left",
        "before_rt_right",
        "before_area",
        "after_rt_left",
        "after_rt_right",
        "after_area",
        "after_height",
        "after_apex_rt",
        "baseline_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in edit_log:
            before = item.get("before", {})
            after = item.get("after", {})
            writer.writerow(
                {
                    "timestamp": item.get("timestamp", ""),
                    "action": item.get("action", ""),
                    "feature_id": item.get("feature_id", ""),
                    "sample": item.get("sample", ""),
                    "before_rt_left": before.get("rt_left", ""),
                    "before_rt_right": before.get("rt_right", ""),
                    "before_area": before.get("area", ""),
                    "after_rt_left": after.get("rt_left", ""),
                    "after_rt_right": after.get("rt_right", ""),
                    "after_area": after.get("area", ""),
                    "after_height": after.get("height", ""),
                    "after_apex_rt": after.get("apex_rt", ""),
                    "baseline_mode": after.get("baseline_mode", ""),
                }
            )


def _do_export(
    outdir: Path,
    curations: List[Dict[str, Any]],
    edit_log: List[Dict[str, Any]],
    mzml_files: List[Path],
    run_state: Dict[str, Any],
) -> List[Path]:
    """Run the actual file writes off the Tk thread.

    Importantly: the previous implementation called ``file_sha1`` on every
    mzML file to embed a hash in the metadata. Each MTBLS-style mzML is
    several hundred MB, so SHA-1 of the whole cohort meant re-reading 100+ GB
    just to write metadata, which froze the GUI for 30+ minutes on HDD. We
    drop the hash and rely on (size, mtime) for cheap reproducibility.
    """
    import pandas as pd

    long_path = outdir / "manual_curated_peak_areas_long.csv"
    matrix_path = outdir / "manual_curated_peak_areas_matrix.csv"
    edit_path = outdir / "manual_curation_edits.csv"
    meta_path = outdir / "manual_curation_metadata.json"

    long_df = pd.DataFrame(curations).sort_values(["feature_id", "sample"])
    matrix_df = long_df.pivot_table(
        index="feature_id",
        columns="sample",
        values="area",
        aggfunc="first",
    )
    long_df.to_csv(long_path, index=False)
    matrix_df.to_csv(matrix_path)
    _write_edit_log_csv(edit_path, edit_log)

    file_entries = []
    for path in mzml_files:
        entry: Dict[str, Any] = {
            "sample": sample_name_from_path(path),
            "path": str(path),
        }
        try:
            stat = path.stat()
            entry["size_bytes"] = int(stat.st_size)
            entry["mtime"] = float(stat.st_mtime)
        except OSError:
            pass
        file_entries.append(entry)

    metadata = {
        "software": "MetSynQ Manual Peak Curator",
        "version": APP_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **run_state,
        "mzml_files": file_entries,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [long_path, matrix_path, edit_path, meta_path]


def integrate_eic(
    eic: EIC,
    rt_left: float,
    rt_right: float,
    baseline_mode: str = "linear",
) -> Tuple[float, float, float, int]:
    require_science_stack()
    import numpy as np

    if rt_right < rt_left:
        rt_left, rt_right = rt_right, rt_left
    rt = np.asarray(eic.rt, dtype=float)
    intensity = np.asarray(eic.intensity, dtype=float)
    mask = (rt >= rt_left) & (rt <= rt_right)
    if not mask.any():
        return 0.0, 0.0, math.nan, 0

    x = rt[mask]
    y = intensity[mask]
    if len(x) == 1:
        corrected = y.copy()
    elif baseline_mode == "none":
        corrected = y.copy()
    elif baseline_mode == "edge_min":
        corrected = y - min(float(y[0]), float(y[-1]))
    else:
        baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
        corrected = y - baseline
    corrected = np.maximum(corrected, 0.0)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(trapezoid(corrected, x)) if len(x) > 1 else 0.0
    height = float(np.max(corrected)) if len(corrected) else 0.0
    apex_rt = float(x[int(np.argmax(corrected))]) if len(corrected) else math.nan
    return area, height, apex_rt, int(len(x))


def auto_pick_first_signal(mzml_dir: Path) -> Tuple[float, float]:
    require_science_stack()
    import numpy as np
    import pymzml

    files = mzml_files_from_dir(mzml_dir)
    if not files:
        raise SystemExit(f"No mzML files found in {mzml_dir}")
    try:
        reader = pymzml.run.Reader(str(files[0]))
    except Exception:
        reader = pymzml.run.Reader(str(files[0]), build_index_from_scratch=True)
    for spectrum in reader:
        try:
            if int(spectrum.ms_level) != 1:
                continue
        except Exception:
            continue
        peaks = np.asarray(spectrum_peaks(spectrum))
        if not peaks.size:
            continue
        peaks = peaks.reshape((-1, 2))
        idx = int(np.argmax(peaks[:, 1]))
        rt = spectrum_rt_minutes(spectrum)
        if rt is not None:
            return float(peaks[idx, 0]), float(rt)
    raise SystemExit("Could not find an MS1 signal in the first mzML file.")


def run_smoke_test(args: argparse.Namespace) -> int:
    mzml_dir = Path(args.mzml_dir).expanduser().resolve()
    mz = args.mz
    rt = args.rt
    if mz is None or rt is None:
        mz, rt = auto_pick_first_signal(mzml_dir)
    files = mzml_files_from_dir(mzml_dir)
    if not files:
        raise SystemExit(f"No mzML files found in {mzml_dir}")
    eic = extract_eic(
        files[0],
        mz=float(mz),
        ppm=float(args.ppm),
        rt_center=float(rt),
        rt_window=float(args.rt_window),
        intensity_mode=args.intensity_mode,
    )
    left = float(rt) - DEFAULT_BOUNDARY_HALF_WIDTH
    right = float(rt) + DEFAULT_BOUNDARY_HALF_WIDTH
    area, height, apex_rt, points = integrate_eic(eic, left, right, args.baseline)
    print(json.dumps(
        {
            "file": str(files[0]),
            "sample": eic.sample,
            "mz": mz,
            "rt": rt,
            "eic_points": len(eic.rt),
            "boundary": [left, right],
            "area": area,
            "height": height,
            "apex_rt": apex_rt,
            "points_integrated": points,
        },
        indent=2,
    ))
    return 0


class ManualPeakCuratorApp:
    def __init__(self, root: Any):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.root.title("MetSynQ Manual Peak Curator")
        self.root.geometry("1280x820")

        self.tk = tk
        self.ttk = ttk
        self.mzml_dir: Optional[Path] = None
        self.feature_table_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.mzml_files: List[Path] = []
        self.features: List[Feature] = []
        self.feature_index = 0
        self.selected_sample: Optional[str] = None
        self._refreshing_result_tree = False
        self.eic_cache: Dict[Tuple[str, str, float, float, str], EIC] = {}
        self.curations: Dict[Tuple[str, str], Curation] = {}
        self.edit_log: List[Dict[str, Any]] = []
        self.span_selectors: List[Any] = []

        # Async EIC extraction: prevents UI freeze when the cohort has many
        # mzML files. _loading is a re-entry guard for refresh_current_feature;
        # _extraction_cancel lets a new feature selection abort the current
        # batch; _pending_feature_index is the feature to jump to once the
        # current batch ends.
        self._loading = False
        self._extraction_cancel: Optional[threading.Event] = None
        self._extract_pool: Optional[ThreadPoolExecutor] = None
        self._pending_feature_index: Optional[int] = None
        # Export runs on a worker thread so disk I/O doesn't freeze the UI.
        self._exporting = False

        self.ppm_var = tk.DoubleVar(value=DEFAULT_PPM)
        self.rt_window_var = tk.DoubleVar(value=DEFAULT_RT_WINDOW)
        self.baseline_var = tk.StringVar(value="linear")
        self.intensity_mode_var = tk.StringVar(value="sum")
        self.left_var = tk.StringVar(value="")
        self.right_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Load mzML files and a feature table to begin.")

        self._build_layout()

    def _build_layout(self) -> None:
        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from matplotlib.widgets import SpanSelector
        self.SpanSelector = SpanSelector

        tk = self.tk
        ttk = self.ttk

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=330, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=0)
        main.add(right, weight=1)

        file_box = ttk.LabelFrame(left, text="Inputs", padding=8)
        file_box.pack(fill=tk.X)
        ttk.Button(file_box, text="Load mzML Folder", command=self.load_mzml_folder).pack(fill=tk.X, pady=2)
        ttk.Button(file_box, text="Load Feature Table", command=self.load_feature_table).pack(fill=tk.X, pady=2)
        ttk.Button(file_box, text="Set Output Folder", command=self.set_output_folder).pack(fill=tk.X, pady=2)

        param_box = ttk.LabelFrame(left, text="Extraction", padding=8)
        param_box.pack(fill=tk.X, pady=8)
        self._labeled_entry(param_box, "m/z ppm", self.ppm_var)
        self._labeled_entry(param_box, "RT window (min)", self.rt_window_var)
        ttk.Label(param_box, text="Baseline").pack(anchor=tk.W)
        ttk.Combobox(
            param_box,
            textvariable=self.baseline_var,
            values=["linear", "edge_min", "none"],
            state="readonly",
        ).pack(fill=tk.X, pady=2)
        ttk.Label(param_box, text="EIC intensity").pack(anchor=tk.W)
        ttk.Combobox(
            param_box,
            textvariable=self.intensity_mode_var,
            values=["sum", "max"],
            state="readonly",
        ).pack(fill=tk.X, pady=2)
        ttk.Button(param_box, text="Refresh EIC", command=self.refresh_current_feature).pack(fill=tk.X, pady=4)

        feature_box = ttk.LabelFrame(left, text="Features", padding=8)
        feature_box.pack(fill=tk.BOTH, expand=True)
        feature_table_frame = ttk.Frame(feature_box)
        feature_table_frame.pack(fill=tk.BOTH, expand=True)
        self.feature_tree = ttk.Treeview(
            feature_table_frame,
            columns=("feature_id", "rt", "ms1_mz"),
            show="headings",
            height=14,
        )
        feature_scrollbar = ttk.Scrollbar(
            feature_table_frame,
            orient=tk.VERTICAL,
            command=self.feature_tree.yview,
        )
        self.feature_tree.configure(yscrollcommand=feature_scrollbar.set)
        self.feature_tree.heading("feature_id", text="Feature")
        self.feature_tree.heading("rt", text="RT")
        self.feature_tree.heading("ms1_mz", text="MS1 m/z")
        self.feature_tree.column("feature_id", width=92, anchor=tk.W)
        self.feature_tree.column("rt", width=70, anchor=tk.E)
        self.feature_tree.column("ms1_mz", width=98, anchor=tk.E)
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_tree.bind("<<TreeviewSelect>>", self.on_feature_selected)

        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=6)
        ttk.Button(nav, text="Prev", command=lambda: self.step_feature(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next", command=lambda: self.step_feature(1)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        edit_box = ttk.LabelFrame(left, text="Boundary Edit", padding=8)
        edit_box.pack(fill=tk.X, pady=8)
        ttk.Label(edit_box, text="Left RT").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(edit_box, textvariable=self.left_var, width=12).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(edit_box, text="Right RT").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(edit_box, textvariable=self.right_var, width=12).grid(row=1, column=1, sticky=tk.EW)
        edit_box.columnconfigure(1, weight=1)
        ttk.Button(edit_box, text="Apply To All Samples", command=self.apply_to_all_samples).grid(
            row=2, column=0, columnspan=2, sticky=tk.EW, pady=4
        )
        ttk.Button(edit_box, text="Apply To Selected Sample", command=self.apply_to_selected_sample).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW
        )
        ttk.Button(edit_box, text="Show All Samples", command=self.show_all_samples).grid(
            row=4, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0)
        )
        ttk.Button(edit_box, text="Export Curated Results", command=self.export_results).grid(
            row=5, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0)
        )

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(right)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.plot_scrollbar = ttk.Scrollbar(plot_frame, orient=tk.VERTICAL)
        self.plot_scrollbar.grid(row=0, column=1, sticky="ns")
        self.plot_scroll_canvas = tk.Canvas(
            plot_frame,
            highlightthickness=0,
            yscrollcommand=self.plot_scrollbar.set,
        )
        self.plot_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.plot_scrollbar.configure(command=self.plot_scroll_canvas.yview)

        self.figure = Figure(figsize=(9, 5.7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_scroll_canvas)
        self.plot_window = self.plot_scroll_canvas.create_window(
            (0, 0),
            window=self.canvas.get_tk_widget(),
            anchor="nw",
        )
        self.canvas.get_tk_widget().bind("<Configure>", self._update_plot_scrollregion)
        self.canvas.get_tk_widget().bind("<MouseWheel>", self._on_plot_mousewheel)
        self.canvas.get_tk_widget().bind("<Button-4>", self._on_plot_mousewheel)
        self.canvas.get_tk_widget().bind("<Button-5>", self._on_plot_mousewheel)
        self.plot_scroll_canvas.bind("<Configure>", self._resize_plot_window)
        self.plot_scroll_canvas.bind("<MouseWheel>", self._on_plot_mousewheel)
        self.plot_scroll_canvas.bind("<Button-4>", self._on_plot_mousewheel)
        self.plot_scroll_canvas.bind("<Button-5>", self._on_plot_mousewheel)
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

        bottom = ttk.Frame(right)
        bottom.grid(row=2, column=0, sticky="ew", pady=4)
        self.status = ttk.Label(bottom, textvariable=self.status_var, anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.result_tree = ttk.Treeview(
            right,
            columns=("sample", "left", "right", "area", "height", "points"),
            show="headings",
            height=7,
        )
        for col, label, width in [
            ("sample", "Sample", 160),
            ("left", "Left RT", 90),
            ("right", "Right RT", 90),
            ("area", "Area", 120),
            ("height", "Height", 120),
            ("points", "Points", 60),
        ]:
            self.result_tree.heading(col, text=label)
            self.result_tree.column(col, width=width, anchor=tk.E if col != "sample" else tk.W)
        self.result_tree.grid(row=3, column=0, sticky="ew")
        self.result_tree.bind("<<TreeviewSelect>>", self.on_sample_selected)

    def _update_plot_scrollregion(self, _event: Any = None) -> None:
        self.plot_scroll_canvas.configure(scrollregion=self.plot_scroll_canvas.bbox("all"))

    def _resize_canvas_pixels(self, width: int, height: int) -> None:
        """Resize the matplotlib FigureCanvasTk drawing surface to ``width``x``height`` pixels.

        ``widget.configure(width=, height=)`` alone does not propagate to
        FigureCanvasTk._tkphoto when the widget lives inside a Canvas
        ``create_window``: Tk does not deliver a ``<Configure>`` event in that
        case, so matplotlib never recreates its photo at the new size and the
        figure stays clipped to the photo's old dimensions. Calling the
        backend's ``resize`` directly forces both the figure inches and the
        photo to follow our intent.
        """
        if width <= 0 or height <= 0:
            return
        widget = self.canvas.get_tk_widget()
        widget.configure(width=int(width), height=int(height))
        evt = types.SimpleNamespace(width=int(width), height=int(height))
        try:
            self.canvas.resize(evt)
        except Exception:
            pass

    def _resize_plot_window(self, event: Any) -> None:
        self.plot_scroll_canvas.itemconfigure(self.plot_window, width=event.width)
        widget = self.canvas.get_tk_widget()
        target_h = widget.winfo_reqheight() or widget.winfo_height()
        self._resize_canvas_pixels(event.width, target_h)
        self._update_plot_scrollregion()

    def _ensure_extract_pool(self) -> ThreadPoolExecutor:
        if self._extract_pool is None:
            cpu = os.cpu_count() or 4
            # Capped at 4: pymzml is GIL-bound (XML parsing) and large mzML
            # files thrash spinning disks if too many are read in parallel.
            n_workers = max(2, min(4, cpu))
            self._extract_pool = ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="eic-extract",
            )
        return self._extract_pool

    def shutdown(self) -> None:
        """Release the background extraction pool. Safe to call multiple times."""
        if self._extraction_cancel is not None:
            self._extraction_cancel.set()
        pool = self._extract_pool
        self._extract_pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)

    def _prefetch_eics(self, paths: List[Path]) -> bool:
        """Extract EICs for ``paths`` × ALL loaded features in parallel.

        Critical perf win: each mzML file is parsed ONCE per call, and
        ``extract_eics_multi`` extracts EICs for every feature whose RT window
        overlaps each spectrum. The serial-per-feature path used to re-parse
        every file once per feature, which on cohorts with hundreds of large
        Orbitrap mzML files turns into hours of work and a fully frozen UI.

        Tk event pump via ``root.update()`` keeps the window responsive and
        lets the user switch to another feature mid-load (handled by
        ``_extraction_cancel``).

        Returns False if cancelled (caller should abort the current refresh
        and let the next one take over), True otherwise.
        """
        if not self.features or not paths:
            return True

        pool = self._ensure_extract_pool()
        self._extraction_cancel = threading.Event()
        cancel = self._extraction_cancel

        ppm = float(self.ppm_var.get())
        rt_window = float(self.rt_window_var.get())
        intensity_mode = self.intensity_mode_var.get()
        features_snapshot = list(self.features)

        futures: Dict[Any, Path] = {}
        for path in paths:
            fut = pool.submit(
                extract_eics_multi,
                path,
                features_snapshot,
                ppm,
                rt_window,
                intensity_mode,
            )
            futures[fut] = path

        pending = set(futures.keys())
        total = len(futures)
        completed = 0
        self.status_var.set(
            f"Parsing mzML cohort: 0/{total} files (one-time cost; later switches reuse cache)"
        )
        try:
            self.root.update()
        except Exception:
            cancel.set()
            return False

        # Map feature_id -> Feature so we can rebuild cache keys for current params.
        feature_by_id = {f.feature_id: f for f in features_snapshot}

        while pending:
            if cancel.is_set():
                for fut in list(pending):
                    fut.cancel()
                return False
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for fut in done:
                path = futures[fut]
                try:
                    eics_by_id = fut.result()
                    for fid, eic in eics_by_id.items():
                        feat = feature_by_id.get(fid)
                        if feat is not None:
                            self.eic_cache[self.eic_cache_key(feat, path)] = eic
                except Exception:
                    # Surface as "EIC load failed" placeholder during plotting.
                    pass
                completed += 1
            if done:
                self.status_var.set(
                    f"Parsing mzML cohort: {completed}/{total} files"
                )
            try:
                self.root.update()
            except Exception:
                cancel.set()
                return False
        return True

    def _on_plot_mousewheel(self, event: Any) -> None:
        if getattr(event, "num", None) == 4:
            delta = -3
        elif getattr(event, "num", None) == 5:
            delta = 3
        else:
            delta = -1 * int(getattr(event, "delta", 0) / 120)
        if delta:
            self.plot_scroll_canvas.yview_scroll(delta, "units")

    def _labeled_entry(self, parent: Any, label: str, variable: Any) -> None:
        row = self.ttk.Frame(parent)
        row.pack(fill=self.tk.X, pady=2)
        self.ttk.Label(row, text=label).pack(side=self.tk.LEFT)
        self.ttk.Entry(row, textvariable=variable, width=12).pack(side=self.tk.RIGHT)

    def load_mzml_folder(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askdirectory(title="Select mzML folder")
        if not selected:
            return
        path = Path(selected)
        files = mzml_files_from_dir(path)
        if not files:
            messagebox.showerror("No mzML", f"No .mzML files found in {path}")
            return
        self.mzml_dir = path
        self.output_dir = path.parent / "manual_peak_curation"
        self.mzml_files = files
        self.status_var.set(f"Loaded {len(files)} mzML files from {path}")
        self.refresh_current_feature()

    def load_feature_table(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askopenfilename(
            title="Select feature table",
            filetypes=[
                ("Feature tables", "*.csv *.tsv *.txt *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            features = read_feature_table(Path(selected))
        except Exception as exc:
            messagebox.showerror("Feature table error", str(exc))
            return
        self.feature_table_path = Path(selected)
        self.features = features
        self.feature_index = 0
        self._populate_features()
        self.status_var.set(f"Loaded {len(features)} features from {selected}")
        self.refresh_current_feature()

    def set_output_folder(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_dir = Path(selected)
            self.status_var.set(f"Output folder set to {self.output_dir}")

    def _populate_features(self) -> None:
        self.feature_tree.delete(*self.feature_tree.get_children())
        for idx, feature in enumerate(self.features):
            self.feature_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(feature.feature_id, f"{feature.rt:.4f}", f"{feature.mz:.5f}"),
            )
        if self.features:
            self.feature_tree.selection_set("0")
            self.feature_tree.focus("0")

    def on_feature_selected(self, _event: Any = None) -> None:
        selected = self.feature_tree.selection()
        if not selected:
            return
        new_idx = int(selected[0])
        if self._loading:
            # A new selection while a previous batch is still extracting:
            # remember the target and ask the in-flight batch to stop. The
            # finally clause in refresh_current_feature will jump to it.
            if new_idx != self.feature_index:
                self._pending_feature_index = new_idx
                if self._extraction_cancel is not None:
                    self._extraction_cancel.set()
            return
        self.feature_index = new_idx
        self.refresh_current_feature()

    def on_sample_selected(self, _event: Any = None) -> None:
        if self._refreshing_result_tree or self._loading:
            return
        selected = self.result_tree.selection()
        if not selected:
            return
        sample = str(selected[0])
        if sample == self.selected_sample:
            return
        self.selected_sample = sample
        feature = self.current_feature()
        if feature is not None:
            curation = self.get_curation(feature, self.selected_sample)
            self.left_var.set(f"{curation.rt_left:.5f}")
            self.right_var.set(f"{curation.rt_right:.5f}")
        self.refresh_current_feature()

    def show_all_samples(self) -> None:
        if self._loading:
            return
        self.selected_sample = None
        self.result_tree.selection_remove(self.result_tree.selection())
        self.refresh_current_feature()

    def step_feature(self, delta: int) -> None:
        if not self.features or self._loading:
            return
        self.feature_index = max(0, min(len(self.features) - 1, self.feature_index + delta))
        self.feature_tree.selection_set(str(self.feature_index))
        self.feature_tree.focus(str(self.feature_index))
        self.feature_tree.see(str(self.feature_index))
        self.refresh_current_feature()

    def current_feature(self) -> Optional[Feature]:
        if not self.features:
            return None
        return self.features[self.feature_index]

    def default_bounds(self, feature: Feature) -> Tuple[float, float]:
        if feature.rtmin is not None and feature.rtmax is not None and feature.rtmax > feature.rtmin:
            return feature.rtmin, feature.rtmax
        return feature.rt - DEFAULT_BOUNDARY_HALF_WIDTH, feature.rt + DEFAULT_BOUNDARY_HALF_WIDTH

    def display_rt_bounds(self, feature: Feature) -> Tuple[float, float]:
        left, right = self.default_bounds(feature)
        if right < left:
            left, right = right, left
        width = max(right - left, 0.02)
        pad = width * 0.5
        return max(0.0, left - pad), right + pad

    def get_curation(self, feature: Feature, sample: str) -> Curation:
        key = (feature.feature_id, sample)
        if key not in self.curations:
            left, right = self.default_bounds(feature)
            self.curations[key] = Curation(
                feature_id=feature.feature_id,
                sample=sample,
                mz=feature.mz,
                rt_expected=feature.rt,
                rt_left=left,
                rt_right=right,
                baseline_mode=self.baseline_var.get(),
            )
        return self.curations[key]

    def eic_cache_key(self, feature: Feature, mzml_path: Path) -> Tuple[str, str, float, float, str]:
        return (
            feature.feature_id,
            str(mzml_path),
            round(float(self.ppm_var.get()), 6),
            round(float(self.rt_window_var.get()), 6),
            self.intensity_mode_var.get(),
        )

    def load_eic_cached(self, feature: Feature, mzml_path: Path) -> EIC:
        key = self.eic_cache_key(feature, mzml_path)
        if key not in self.eic_cache:
            self.eic_cache[key] = extract_eic(
                mzml_path,
                feature.mz,
                float(self.ppm_var.get()),
                rt_center=feature.rt,
                rt_window=float(self.rt_window_var.get()),
                intensity_mode=self.intensity_mode_var.get(),
            )
        return self.eic_cache[key]

    def refresh_current_feature(self) -> None:
        feature = self.current_feature()
        if feature is None or not self.mzml_files:
            return
        if self._loading:
            # Re-entry from inside root.update(); ignore — the in-flight
            # refresh will pick up _pending_feature_index when it returns.
            return
        self._loading = True
        self._pending_feature_index = None
        try:
            self._do_refresh_current_feature(feature)
        finally:
            self._loading = False
            pending = self._pending_feature_index
            self._pending_feature_index = None
            if pending is not None and pending != self.feature_index:
                # User picked a different feature while we were busy. Honour it.
                self.feature_index = pending
                try:
                    self.feature_tree.selection_set(str(pending))
                    self.feature_tree.focus(str(pending))
                    self.feature_tree.see(str(pending))
                except Exception:
                    pass
                self.root.after(0, self.refresh_current_feature)

    def _do_refresh_current_feature(self, feature: Feature) -> None:
        previous_scroll = self.plot_scroll_canvas.yview()[0]

        # Pre-fetch any uncached EICs in parallel so the UI never blocks on
        # serial mzML parsing. We bulk-extract EICs for *all* features per
        # file in one pass, so subsequent feature switches hit the cache.
        # If the user switches features mid-load this returns False and we
        # bail out; the outer finally handles the jump.
        missing = [
            p for p in self.mzml_files
            if self.eic_cache_key(feature, p) not in self.eic_cache
        ]
        if missing:
            if not self._prefetch_eics(missing):
                return

        self.figure.clear()
        self.span_selectors.clear()
        self._refreshing_result_tree = True
        self.result_tree.delete(*self.result_tree.get_children())
        self._refreshing_result_tree = False
        left, right = self.default_bounds(feature)
        self.left_var.set(f"{left:.5f}")
        self.right_var.set(f"{right:.5f}")

        n_samples = len(self.mzml_files)
        n_cols = 3
        n_rows = max(1, math.ceil(n_samples / n_cols))
        figure_height = max(5.7, n_rows * 1.9)
        scroll_width_px = self.plot_scroll_canvas.winfo_width()
        target_width_px = max(scroll_width_px, int(9.2 * self.figure.dpi))
        target_height_px = int(figure_height * self.figure.dpi)
        self._resize_canvas_pixels(target_width_px, target_height_px)
        axes_grid = self.figure.subplots(n_rows, n_cols, squeeze=False)
        axes = [ax for row in axes_grid for ax in row]
        display_rt_left, display_rt_right = self.display_rt_bounds(feature)

        loaded = 0
        selected_exists = False
        for ax, mzml_path in zip(axes, self.mzml_files):
            try:
                eic = self.load_eic_cached(feature, mzml_path)
            except Exception as exc:
                self.status_var.set(f"Failed to extract {mzml_path.name}: {exc}")
                ax.set_title(sample_name_from_path(mzml_path))
                ax.text(0.5, 0.5, "EIC load failed", ha="center", va="center", transform=ax.transAxes)
                continue
            curation = self.get_curation(feature, eic.sample)
            curation.baseline_mode = self.baseline_var.get()
            self._recalculate_one(feature, eic, curation)
            is_selected = self.selected_sample == eic.sample
            selected_exists = selected_exists or is_selected
            if eic.rt:
                ax.plot(eic.rt, eic.intensity, lw=1.1, alpha=0.9, color="#2b6f9f")
                loaded += 1
            ax.axvline(feature.rt, color="#444444", ls="--", lw=0.8)
            ax.axvspan(curation.rt_left, curation.rt_right, color="#cc5533", alpha=0.18)
            ax.set_xlim(display_rt_left, display_rt_right)
            ax.set_title(
                eic.sample,
                fontsize=9,
                color="#b23b22" if is_selected else "#222222",
                fontweight="bold" if is_selected else "normal",
            )
            ax.tick_params(axis="both", labelsize=7)
            ax.set_xlabel("RT (min)", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            selector = self.SpanSelector(
                ax,
                lambda xmin, xmax, sample=eic.sample: self.on_sample_span_selected(sample, xmin, xmax),
                "horizontal",
                useblit=True,
                props=dict(alpha=0.18, facecolor="#cc5533"),
                interactive=True,
            )
            self.span_selectors.append(selector)
            self._add_result_row(curation)

        for ax in axes[n_samples:]:
            ax.set_visible(False)

        current_left, current_right = self.default_bounds(feature)
        if self.selected_sample is not None and selected_exists:
            c = self.get_curation(feature, self.selected_sample)
            current_left, current_right = c.rt_left, c.rt_right
        elif self.mzml_files:
            first_sample = sample_name_from_path(self.mzml_files[0])
            c = self.get_curation(feature, first_sample)
            current_left, current_right = c.rt_left, c.rt_right
        self.left_var.set(f"{current_left:.5f}")
        self.right_var.set(f"{current_right:.5f}")
        self.figure.suptitle(
            f"{feature.feature_id}  MS1 m/z {feature.mz:.5f}  RT {feature.rt:.4f}",
            fontsize=11,
        )
        # tight_layout becomes prohibitively slow with many subplots and may
        # iterate for seconds on cohorts with hundreds of mzML files. Use
        # fixed margins past a small threshold.
        if n_samples > 24:
            self.figure.subplots_adjust(
                left=0.05,
                right=0.99,
                top=1 - min(0.04, 0.6 / max(figure_height, 1)),
                bottom=min(0.04, 0.6 / max(figure_height, 1)),
                wspace=0.32,
                hspace=0.65,
            )
        else:
            self.figure.tight_layout()
        self.canvas.draw()
        self.canvas.get_tk_widget().update_idletasks()
        self._update_plot_scrollregion()
        self.plot_scroll_canvas.yview_moveto(previous_scroll)
        if self.selected_sample is not None and selected_exists:
            self._refreshing_result_tree = True
            self.result_tree.selection_set(self.selected_sample)
            self.result_tree.focus(self.selected_sample)
            self._refreshing_result_tree = False
            self.status_var.set(f"Displayed {loaded}/{len(self.mzml_files)} samples for {feature.feature_id}; selected {self.selected_sample}")
        else:
            self.selected_sample = None
            self.status_var.set(f"Displayed {loaded}/{len(self.mzml_files)} samples for {feature.feature_id}")

    def _add_result_row(self, curation: Curation) -> None:
        self.result_tree.insert(
            "",
            "end",
            iid=curation.sample,
            values=(
                curation.sample,
                f"{curation.rt_left:.5f}",
                f"{curation.rt_right:.5f}",
                f"{curation.area:.6g}",
                f"{curation.height:.6g}",
                curation.points,
            ),
        )

    def _recalculate_one(self, feature: Feature, eic: EIC, curation: Curation) -> None:
        area, height, apex_rt, points = integrate_eic(
            eic,
            curation.rt_left,
            curation.rt_right,
            self.baseline_var.get(),
        )
        curation.area = area
        curation.height = height
        curation.apex_rt = apex_rt
        curation.points = points
        curation.baseline_mode = self.baseline_var.get()
        curation.mz = feature.mz
        curation.rt_expected = feature.rt

    def on_span_selected(self, xmin: float, xmax: float) -> None:
        self.left_var.set(f"{min(xmin, xmax):.5f}")
        self.right_var.set(f"{max(xmin, xmax):.5f}")
        if self.selected_sample is None:
            self.apply_to_all_samples()
        else:
            self.apply_to_selected_sample()

    def on_sample_span_selected(self, sample: str, xmin: float, xmax: float) -> None:
        self.selected_sample = sample
        self.left_var.set(f"{min(xmin, xmax):.5f}")
        self.right_var.set(f"{max(xmin, xmax):.5f}")
        self.apply_to_selected_sample()

    def apply_to_all_samples(self) -> None:
        from tkinter import messagebox

        if self._loading:
            return
        feature = self.current_feature()
        if feature is None or not self.mzml_files:
            return
        try:
            left = float(self.left_var.get())
            right = float(self.right_var.get())
        except Exception:
            messagebox.showerror("Invalid RT", "Left and right RT boundaries must be numeric.")
            return
        if right < left:
            left, right = right, left

        timestamp = datetime.now().isoformat(timespec="seconds")
        for mzml_path in self.mzml_files:
            sample = sample_name_from_path(mzml_path)
            curation = self.get_curation(feature, sample)
            before = asdict(curation)
            curation.rt_left = left
            curation.rt_right = right
            curation.edited = True
            eic = self.load_eic_cached(feature, mzml_path)
            self._recalculate_one(feature, eic, curation)
            after = asdict(curation)
            self.edit_log.append(
                {
                    "timestamp": timestamp,
                    "action": "apply_to_all_samples",
                    "feature_id": feature.feature_id,
                    "sample": sample,
                    "before": before,
                    "after": after,
                }
        )
        self.selected_sample = None
        self.refresh_current_feature()

    def apply_to_selected_sample(self) -> None:
        from tkinter import messagebox

        if self._loading:
            return
        feature = self.current_feature()
        if feature is None or not self.mzml_files:
            return
        selected = self.result_tree.selection()
        sample = self.selected_sample or (str(selected[0]) if selected else "")
        if not sample:
            messagebox.showerror("No sample selected", "Select a sample row in the results table first.")
            return
        try:
            left = float(self.left_var.get())
            right = float(self.right_var.get())
        except Exception:
            messagebox.showerror("Invalid RT", "Left and right RT boundaries must be numeric.")
            return
        if right < left:
            left, right = right, left

        mzml_path = next((path for path in self.mzml_files if sample_name_from_path(path) == sample), None)
        if mzml_path is None:
            messagebox.showerror("Sample not found", f"Could not find mzML file for sample {sample}.")
            return

        curation = self.get_curation(feature, sample)
        before = asdict(curation)
        curation.rt_left = left
        curation.rt_right = right
        curation.edited = True
        eic = self.load_eic_cached(feature, mzml_path)
        self._recalculate_one(feature, eic, curation)
        self.edit_log.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "action": "apply_to_selected_sample",
                "feature_id": feature.feature_id,
                "sample": sample,
                "before": before,
                "after": asdict(curation),
            }
        )
        self.refresh_current_feature()
        self.result_tree.selection_set(sample)

    def export_results(self) -> None:
        from tkinter import messagebox

        if self._exporting:
            return
        if self._loading:
            messagebox.showinfo(
                "Busy",
                "EIC extraction is in progress. Wait for it to finish before exporting.",
            )
            return
        if not self.curations:
            messagebox.showerror("No results", "No curated peaks to export.")
            return

        outdir = self.output_dir
        if outdir is None:
            if self.mzml_dir:
                outdir = self.mzml_dir.parent / "manual_peak_curation"
            else:
                outdir = Path.cwd() / "manual_peak_curation"
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Output folder error", f"Cannot create {outdir}: {exc}")
            return

        # Snapshot everything the worker needs so the main thread is free to
        # keep editing while we write. Also avoids races on self.curations,
        # self.edit_log, and the Tk variables.
        curations_snapshot = [asdict(c) for c in self.curations.values()]
        edit_log_snapshot = list(self.edit_log)
        mzml_files_snapshot = list(self.mzml_files)
        run_state = {
            "mzml_dir": str(self.mzml_dir) if self.mzml_dir else "",
            "feature_table": str(self.feature_table_path) if self.feature_table_path else "",
            "ppm": float(self.ppm_var.get()),
            "rt_window": float(self.rt_window_var.get()),
            "baseline_mode": self.baseline_var.get(),
            "intensity_mode": self.intensity_mode_var.get(),
        }

        self._exporting = True
        self._export_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.status_var.set(f"Exporting curated results to {outdir}...")

        def worker() -> None:
            try:
                paths = _do_export(
                    outdir,
                    curations_snapshot,
                    edit_log_snapshot,
                    mzml_files_snapshot,
                    run_state,
                )
                self._export_queue.put(("ok", paths))
            except Exception as exc:
                self._export_queue.put(("err", exc))

        threading.Thread(target=worker, daemon=True, name="export").start()
        # Cross-thread Tk calls are unsafe; the main thread polls the queue.
        self.root.after(80, self._poll_export)

    def _poll_export(self) -> None:
        try:
            kind, payload = self._export_queue.get_nowait()
        except queue.Empty:
            if self._exporting:
                self.root.after(80, self._poll_export)
            return
        if kind == "ok":
            self._on_export_complete(payload)
        else:
            self._on_export_error(payload)

    def _on_export_complete(self, paths: List[Path]) -> None:
        from tkinter import messagebox

        self._exporting = False
        outdir = paths[0].parent if paths else "?"
        self.status_var.set(f"Export complete: {len(paths)} files written to {outdir}")
        messagebox.showinfo(
            "Export complete",
            "Exported:\n" + "\n".join(str(p) for p in paths),
        )

    def _on_export_error(self, exc: Exception) -> None:
        from tkinter import messagebox

        self._exporting = False
        self.status_var.set(f"Export failed: {exc}")
        messagebox.showerror("Export failed", str(exc))


def launch_gui() -> int:
    try:
        import tkinter as tk
    except Exception as exc:
        raise SystemExit(
            "Tkinter is not available in this Python environment. On this machine, "
            "use Windows Python, for example: python tools\\manual_peak_curator.py"
        ) from exc
    require_science_stack()
    root = tk.Tk()
    app = ManualPeakCuratorApp(root)

    def _on_close() -> None:
        try:
            app.shutdown()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    try:
        root.mainloop()
    finally:
        app.shutdown()
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual peak area curation GUI for LC-MS mzML data.")
    parser.add_argument("--smoke-test", action="store_true", help="Extract one EIC without opening the GUI.")
    parser.add_argument("--mzml-dir", default="demo_data/untargeted_demo/mzML", help="mzML folder for smoke test.")
    parser.add_argument("--mz", type=float, default=None, help="m/z for smoke test. If omitted, use the strongest first MS1 signal.")
    parser.add_argument("--rt", type=float, default=None, help="RT in minutes for smoke test. If omitted, use the strongest first MS1 signal.")
    parser.add_argument("--ppm", type=float, default=DEFAULT_PPM, help="m/z extraction tolerance in ppm.")
    parser.add_argument("--rt-window", type=float, default=DEFAULT_RT_WINDOW, help="RT extraction window in minutes.")
    parser.add_argument("--baseline", choices=["linear", "edge_min", "none"], default="linear")
    parser.add_argument("--intensity-mode", choices=["sum", "max"], default="sum")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        return run_smoke_test(args)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
