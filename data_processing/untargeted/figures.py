"""Publication-style figures for the unified untargeted benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import read_tsv


COLORS = {
    "MetSynQ": (186 / 255, 147 / 255, 142 / 255),
    "PeakOnly": (109 / 255, 159 / 255, 176 / 255),
    "Quanformer": (177 / 255, 192 / 255, 153 / 255),
    "XCMS": (207 / 255, 192 / 255, 166 / 255),
}
DIFF_COLORS = {
    "identified": (177 / 255, 192 / 255, 153 / 255),
    "false_positive": (186 / 255, 147 / 255, 142 / 255),
    "missed": (109 / 255, 159 / 255, 176 / 255),
}
CV_COLORS = {
    "MetSynQ": (177 / 255, 192 / 255, 153 / 255),
    "PeakOnly": (186 / 255, 147 / 255, 142 / 255),
    "XCMS": (109 / 255, 159 / 255, 176 / 255),
    "Quanformer": (207 / 255, 192 / 255, 166 / 255),
}


def configure() -> str:
    selected = "DejaVu Sans"
    for candidate in ["Arial", "Liberation Sans", "DejaVu Sans"]:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            selected = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.size": 8,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "path",
            "svg.hashsalt": "metsynq-untargeted",
            "pdf.fonttype": 42,
        }
    )
    return selected


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)


def save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def qualitative_figure(
    dataset: str, methods: list[str], table_dir: Path, figure_dir: Path
) -> None:
    summary = read_tsv(table_dir / "qualitative_summary.tsv").set_index("method")
    x = np.arange(3, dtype=float)
    width = min(0.18, 0.72 / max(len(methods), 1))
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    center = (len(methods) - 1) / 2
    for index, method in enumerate(methods):
        means = [
            float(summary.loc[method, "precision_mean"]),
            float(summary.loc[method, "recall_mean"]),
            float(summary.loc[method, "f1_mean"]),
        ]
        stds = [
            float(summary.loc[method, "precision_std"]),
            float(summary.loc[method, "recall_std"]),
            float(summary.loc[method, "f1_std"]),
        ]
        ax.bar(
            x + (index - center) * width,
            means,
            width * 0.88,
            yerr=stds,
            color=COLORS[method],
            edgecolor="none",
            label=method,
            error_kw={"elinewidth": 1, "capsize": 3, "capthick": 1},
        )
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylabel("Score")
    ax.set_ylim(0.70, 1.01)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=len(methods), frameon=False
    )
    finish_axes(ax)
    fig.tight_layout()
    save(fig, figure_dir / f"{dataset}_qualitative_precision_recall_f1")


def cv_figure(
    dataset: str, methods: list[str], table_dir: Path, figure_dir: Path
) -> None:
    values = read_tsv(table_dir / "cv_feature_values.tsv")
    positions: list[float] = []
    distributions: list[np.ndarray] = []
    colors: list[tuple[float, float, float]] = []
    labels: list[str] = []
    centers: list[float] = []
    for method_index, method in enumerate(methods):
        base = method_index * 2.0
        centers.append(base + 0.3)
        for offset, group in [(0.0, "A"), (0.6, "B")]:
            positions.append(base + offset)
            distributions.append(
                values.loc[
                    (values["method"] == method) & (values["sample_group"] == group),
                    "cv_percent",
                ].to_numpy(dtype=float)
            )
            colors.append(CV_COLORS[method])
            labels.append(f"S{group}")
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    boxes = ax.boxplot(
        distributions,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 2,
            "alpha": 0.4,
        },
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
    finite_max = max(float(np.nanmax(values)) for values in distributions)
    top = min(max(100.0, finite_max * 1.02), 180.0)
    for position, label in zip(positions, labels):
        ax.text(position, top * 0.975, label, ha="center", va="bottom")
    ax.set_xticks(centers, methods)
    ax.set_ylabel("CV (%)")
    ax.set_ylim(-8, top)
    finish_axes(ax)
    fig.tight_layout()
    save(fig, figure_dir / f"{dataset}_peak_area_cv")


def differential_figure(
    dataset: str, methods: list[str], table_dir: Path, figure_dir: Path
) -> None:
    summary = read_tsv(table_dir / "differential_summary.tsv").set_index("method")
    plot_methods = list(reversed(methods))
    identified = summary.loc[plot_methods, "identified_count"].to_numpy(dtype=float)
    false_positive = summary.loc[plot_methods, "false_positive_count"].to_numpy(
        dtype=float
    )
    missed = summary.loc[plot_methods, "missed_count"].to_numpy(dtype=float)
    y = np.arange(len(plot_methods))
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    ax.barh(y, identified, color=DIFF_COLORS["identified"], label="Identified")
    ax.barh(
        y,
        false_positive,
        left=identified,
        color=DIFF_COLORS["false_positive"],
        label="False Pos",
    )
    ax.barh(
        y,
        missed,
        left=identified + false_positive,
        color=DIFF_COLORS["missed"],
        label="Missed",
    )
    for index in range(len(plot_methods)):
        segments = [identified[index], false_positive[index], missed[index]]
        starts = [0, identified[index], identified[index] + false_positive[index]]
        text_colors = ["white", "black", "white"]
        for width, start, color in zip(segments, starts, text_colors):
            if width:
                ax.text(
                    start + width / 2,
                    index,
                    str(int(width)),
                    ha="center",
                    va="center",
                    color=color,
                )
    ax.set_yticks(y, plot_methods)
    ax.set_xlabel("Count of Differential Metabolites")
    ax.set_xlim(0, float(np.max(identified + false_positive + missed)) * 1.05)
    ax.legend(loc="upper right", frameon=False)
    finish_axes(ax)
    fig.tight_layout()
    save(fig, figure_dir / f"{dataset}_differential_metabolite_recovery")


def fold_error_figure(
    dataset: str, methods: list[str], table_dir: Path, figure_dir: Path
) -> None:
    comparison = read_tsv(table_dir / "quantitative_feature_comparison.tsv")
    data = [
        comparison.loc[
            comparison["method"] == method, "absolute_relative_error"
        ].to_numpy(dtype=float)
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    boxes = ax.boxplot(
        data,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 2,
            "alpha": 0.4,
        },
    )
    for patch, method in zip(boxes["boxes"], methods):
        patch.set_facecolor(COLORS[method])
    ax.set_xticks(np.arange(1, len(methods) + 1), methods)
    ax.set_ylabel("Absolute Relative Error")
    ax.set_ylim(-0.2, 10)
    finish_axes(ax)
    fig.tight_layout()
    save(fig, figure_dir / f"{dataset}_fold_change_relative_error")


def combined_bar(
    results: dict[str, dict[str, pd.DataFrame]],
    configs: dict[str, dict[str, Any]],
    figure_dir: Path,
    key: str,
    value_column: str,
    ylabel: str,
    basename: str,
    ylim: tuple[float, float],
) -> None:
    datasets = list(configs)
    all_methods: list[str] = []
    for spec in configs.values():
        for method in spec["methods"]:
            if method not in all_methods:
                all_methods.append(method)
    x = np.arange(len(datasets), dtype=float)
    width = 0.16
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    center = (len(all_methods) - 1) / 2
    for index, method in enumerate(all_methods):
        values: list[float] = []
        for dataset in datasets:
            table = results[dataset][key]
            row = table.loc[table["method"] == method, value_column]
            numeric = pd.to_numeric(row, errors="coerce")
            values.append(
                float(numeric.iloc[0])
                if len(numeric) and pd.notna(numeric.iloc[0])
                else np.nan
            )
        ax.bar(
            x + (index - center) * width,
            values,
            width * 0.82,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            label=method,
        )
    ax.set_xticks(x, [configs[dataset]["label"] for dataset in datasets])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(4, len(all_methods)),
        frameon=False,
    )
    finish_axes(ax)
    fig.tight_layout()
    save(fig, figure_dir / basename)


def generate_all(
    results: dict[str, dict[str, pd.DataFrame]],
    configs: dict[str, dict[str, Any]],
    result_root: Path,
) -> str:
    font = configure()
    figure_dir = result_root / "figures" / "main"
    for dataset, spec in configs.items():
        table_dir = result_root / "tables" / dataset
        methods = [
            method for method in spec["methods"] if method in spec["analysis_tables"]
        ]
        qualitative_figure(
            dataset,
            [
                method
                for method in spec["methods"]
                if method in spec["qualitative_tables"]
            ],
            table_dir,
            figure_dir,
        )
        cv_figure(dataset, spec["cv_methods"], table_dir, figure_dir)
        differential_figure(dataset, methods, table_dir, figure_dir)
        fold_error_figure(dataset, methods, table_dir, figure_dir)
    combined_bar(
        results,
        configs,
        figure_dir,
        "final_summary",
        "overall_accuracy",
        "Overall quantitative accuracy",
        "combined_qe_tof_overall_accuracy",
        (0.50, 1.00),
    )
    return font
