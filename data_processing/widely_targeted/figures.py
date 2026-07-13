"""Figures for the widely targeted benchmark."""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"Pandas requires version .* of 'numexpr'")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import InputPaths


COLOR_MS = "#ba938e"
COLOR_MQ = "#6d9fb0"
COLOR_GT = "#b1c099"
COLOR_GRAY = "#717070"


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _style_open_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)
    ax.grid(False)


def _boxplot(
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    stem: Path,
    dpi: int,
    figsize: tuple[float, float] = (3.2, 2.4),
    ylim: tuple[float, float] | None = None,
) -> None:
    clean = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in data]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    box = ax.boxplot(
        clean,
        labels=labels,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "black"},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 2,
            "alpha": 0.35,
        },
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style_open_axes(ax)
    fig.tight_layout()
    _save(fig, stem, dpi)


def plot_detection(detection: dict[str, object], figure_dir: Path, dpi: int) -> None:
    summary = detection["summary"].set_index(["Software", "Statistic"])
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=dpi)
    for offset, method, color in [
        (-width / 2, "MetSynQ", COLOR_MS),
        (width / 2, "MultiQuant", COLOR_MQ),
    ]:
        means = summary.loc[(method, "Mean"), metrics].to_numpy(dtype=float)
        stds = summary.loc[(method, "STD"), metrics].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=color,
            label=method,
            linewidth=0,
        )
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.90, 1.00)
    ax.set_ylabel("Score")
    ax.legend(frameon=True, loc="upper right")
    _style_open_axes(ax)
    fig.tight_layout()
    _save(fig, figure_dir / "Fig2F_detection_metrics", dpi)


def _standardized_curves(paths: InputPaths, figure: str) -> list[dict[str, object]]:
    vertices = pd.read_csv(paths.cdf_curve_vertices, sep="\t")
    vertices = vertices.loc[vertices["Figure"].eq(figure)]
    curves = []
    for series, frame in vertices.groupby("Series", sort=False):
        frame = frame.sort_values("PointOrder")
        curves.append(
            {
                "series": series,
                "color": str(frame["Color"].iloc[0]),
                "dashed": str(frame["Dashed"].iloc[0]).lower() in {"true", "1", "yes"},
                "x": frame["X"].to_numpy(dtype=float),
                "y": frame["Y"].to_numpy(dtype=float),
            }
        )
    return curves


def plot_alignment(
    paths: InputPaths, alignment: dict[str, object], figure_dir: Path, dpi: int,
) -> None:
    curves = _standardized_curves(paths, "alignment")
    if len(curves) != 4:
        raise AssertionError(
            f"Expected four alignment CDF curves, observed {len(curves)}"
        )
    label_map = {
        (COLOR_MS, False): "MetSynQ SD",
        (COLOR_MQ, False): "MultiQuant SD",
        (COLOR_MS, True): "MetSynQ MaxΔRT",
        (COLOR_MQ, True): "MultiQuant MaxΔRT",
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=dpi)
    for curve in curves:
        key = (curve["color"], curve["dashed"])
        ax.plot(
            curve["x"],
            curve["y"],
            color=curve["color"],
            linestyle="--" if curve["dashed"] else "-",
            linewidth=1.5,
            label=label_map[key],
        )
    ax.axvline(0.05, color=COLOR_GRAY, linestyle="--", linewidth=1)
    ax.axvline(0.10, color=COLOR_GRAY, linestyle=":", linewidth=1)
    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, figure_dir / "Fig2G_alignment_cdf", dpi)


def plot_boundary_iou(
    paths: InputPaths, boundary: dict[str, object], figure_dir: Path, dpi: int,
) -> None:
    curves = _standardized_curves(paths, "boundary_iou")
    if len(curves) != 2:
        raise AssertionError(
            f"Expected two signal-IoU CDF curves, observed {len(curves)}"
        )
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=dpi)
    label_by_color = {COLOR_MS: "MetSynQ", COLOR_MQ: "MultiQuant"}
    for curve in curves:
        ax.plot(
            curve["x"],
            curve["y"],
            color=curve["color"],
            linewidth=1.5,
            label=label_by_color[curve["color"]],
        )
    ax.axvline(0.8, color=COLOR_GRAY, linestyle="--", linewidth=1)
    summary = boundary["summary"].set_index("Software")
    ms_bad = float(summary.loc["MetSynQ", "IOU_lt_0.8_Rate"] * 100)
    mq_bad = float(summary.loc["MultiQuant", "IOU_lt_0.8_Rate"] * 100)
    ax.text(0.72, ms_bad + 2.0, f"{ms_bad:.1f}%", color=COLOR_MS, fontsize=8)
    ax.text(0.70, mq_bad + 2.0, f"{mq_bad:.1f}%", color=COLOR_MQ, fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Peak Region IoU")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, figure_dir / "Fig2H_peak_boundary_iou_cdf", dpi)


def plot_integration(
    integration: dict[str, object], figure_dir: Path, dpi: int
) -> None:
    cv = integration["cv_values"]
    _boxplot(
        [cv["Ground Truth"], cv["MetSynQ"], cv["MultiQuant"]],
        ["Ground Truth", "MetSynQ", "MultiQuant"],
        [COLOR_GT, COLOR_MS, COLOR_MQ],
        "CV (%)",
        figure_dir / "Fig2I_peak_area_cv",
        dpi,
    )

    errors = integration["relative_errors"]
    _boxplot(
        [errors["MetSynQ"], errors["MultiQuant"]],
        ["MetSynQ", "MultiQuant"],
        [COLOR_MS, COLOR_MQ],
        "Absolute Relative Error",
        figure_dir / "Fig2J_area_relative_error",
        dpi,
        figsize=(3.2, 2.4),
        ylim=(-0.05, 1.05),
    )

    regression = integration["regression_arrays"]
    _boxplot(
        [regression["MetSynQ"]["R2"], regression["MultiQuant"]["R2"]],
        ["MetSynQ", "MultiQuant"],
        [COLOR_MS, COLOR_MQ],
        "R²",
        figure_dir / "FigS9B_area_r2",
        dpi,
        figsize=(3.0, 2.4),
        ylim=(-0.05, 1.05),
    )
    _boxplot(
        [
            np.abs(1 - regression["MetSynQ"]["Slope"]),
            np.abs(1 - regression["MultiQuant"]["Slope"]),
        ],
        ["MetSynQ", "MultiQuant"],
        [COLOR_MS, COLOR_MQ],
        r"$|1 - slope (a)|$",
        figure_dir / "FigS9C_slope_deviation",
        dpi,
        figsize=(3.0, 2.4),
    )


def generate_all(
    paths: InputPaths,
    detection: dict[str, object],
    alignment: dict[str, object],
    integration: dict[str, object],
    boundary: dict[str, object],
    figure_dir: Path,
    dpi: int = 300,
) -> pd.DataFrame:
    _configure_matplotlib()
    plot_detection(detection, figure_dir, dpi)
    plot_alignment(paths, alignment, figure_dir, dpi)
    plot_boundary_iou(paths, boundary, figure_dir, dpi)
    plot_integration(integration, figure_dir, dpi)
    return pd.DataFrame(
        [
            {
                "Figure": "Fig2F_detection_metrics",
                "Question": "How accurately are peaks detected per sample?",
                "Family": "Comparison & Ranking",
                "Variant": "Grouped bars with sample SD",
                "Source": "Recalculated detection table",
            },
            {
                "Figure": "Fig2G_alignment_cdf",
                "Question": "How concentrated are RT SD and maximum delta RT values?",
                "Family": "Distribution",
                "Variant": "Four-series empirical CDF",
                "Source": "Alignment summary and standardized CDF vertices",
            },
            {
                "Figure": "Fig2H_peak_boundary_iou_cdf",
                "Question": "How closely do automatic peak regions overlap manual regions?",
                "Family": "Distribution",
                "Variant": "Two-series empirical CDF with IoU=0.8 threshold",
                "Source": "Boundary universe, IoU error rows, and standardized CDF vertices",
            },
            {
                "Figure": "Fig2I_peak_area_cv",
                "Question": "How stable are peak areas across nine mixtures?",
                "Family": "Distribution",
                "Variant": "Three-group box plot",
                "Source": "806 shared features",
            },
            {
                "Figure": "Fig2J_area_relative_error",
                "Question": "How far are automatic areas from manual areas?",
                "Family": "Distribution",
                "Variant": "Two-group box plot",
                "Source": "Pairwise shared features; relative errors <= 1",
            },
            {
                "Figure": "FigS9B_area_r2",
                "Question": "How well do automatic and manual areas correlate by feature?",
                "Family": "Distribution",
                "Variant": "R² box plot",
                "Source": "806 shared features",
            },
            {
                "Figure": "FigS9C_slope_deviation",
                "Question": "How far is each fitted slope from one?",
                "Family": "Distribution",
                "Variant": "Absolute slope-deviation box plot",
                "Source": "806 shared features",
            },
        ]
    )
