"""Plot publication-ready TripleTOF 6600 ablation comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from common import read_table


VARIANT_ORDER = ["FullPipeline", "ReplaceAlignment", "ReplacePeakPicking"]
DISPLAY_LABELS = {
    "FullPipeline": "Full pipeline",
    "ReplaceAlignment": "Replace\nalignment",
    "ReplacePeakPicking": "Replace\npeak picking",
}
VARIANT_COLORS = {
    "FullPipeline": (186 / 255, 147 / 255, 142 / 255),
    "ReplaceAlignment": (109 / 255, 159 / 255, 176 / 255),
    "ReplacePeakPicking": (177 / 255, 192 / 255, 153 / 255),
}


def configure_matplotlib() -> str:
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
        }
    )
    return selected


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None},
    )
    plt.close(fig)


def plot_total_error_rate(summary, figure_dir: Path) -> None:
    indexed = summary.set_index("variant").loc[VARIANT_ORDER]
    rates = indexed["total_error_rate_percent"].to_numpy(dtype=float)
    x = np.arange(len(VARIANT_ORDER))
    colors = [VARIANT_COLORS[variant] for variant in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    bars = ax.bar(x, rates, width=0.58, color=colors)
    for bar, value in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(x, [DISPLAY_LABELS[variant] for variant in VARIANT_ORDER])
    ax.set_ylabel("Total error rate (%)")
    ax.set_ylim(0, max(rates) * 1.18)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ablation_total_error_rate")


def plot_overall_accuracy(summary, figure_dir: Path) -> None:
    indexed = summary.set_index("variant").loc[VARIANT_ORDER]
    accuracy = 1.0 - indexed["total_error_rate"].to_numpy(dtype=float)
    width = 0.16
    positions = np.array([-width, 0.0, width])
    legend_labels = ["Full pipeline", "Replace alignment", "Replace peak picking"]

    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    for position, value, variant, label in zip(
        positions, accuracy, VARIANT_ORDER, legend_labels
    ):
        ax.bar(
            position,
            value,
            width=width * 0.82,
            color=VARIANT_COLORS[variant],
            edgecolor="black",
            linewidth=0.6,
            label=label,
        )
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(0.50, 1.00)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks([0.0], ["TOF"])
    ax.set_ylabel("Overall accuracy")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=1.5,
        columnspacing=1.0,
    )
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ablation_overall_accuracy")


def plot_error_components(summary, figure_dir: Path) -> None:
    indexed = summary.set_index("variant").loc[VARIANT_ORDER]
    x = np.arange(len(VARIANT_ORDER), dtype=float)
    width = 0.22
    components = [
        ("qualitative_error_count", "Qualitative", (186 / 255, 147 / 255, 142 / 255)),
        ("quantitative_error_count", "Quantitative", (109 / 255, 159 / 255, 176 / 255)),
        ("unique_error_count", "Unique total", (177 / 255, 192 / 255, 153 / 255)),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    for component_index, (column, label, color) in enumerate(components):
        values = indexed[column].to_numpy(dtype=float)
        offset = (component_index - 1) * width
        bars = ax.bar(x + offset, values, width=width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 14,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, [DISPLAY_LABELS[variant] for variant in VARIANT_ORDER])
    ax.set_ylabel("Error feature count")
    ax.set_ylim(0, float(indexed["unique_error_count"].max()) * 1.16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ablation_error_components")


def generate_ablation_figures(summary_path: Path, figure_dir: Path) -> str:
    font_family = configure_matplotlib()
    summary = read_table(summary_path)
    plot_overall_accuracy(summary, figure_dir)
    plot_total_error_rate(summary, figure_dir)
    plot_error_components(summary, figure_dir)
    return font_family
