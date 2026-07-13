"""Plot combined QE and TOF ablation overall accuracy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import read_table
from plot_ablation_comparison import (
    VARIANT_COLORS,
    VARIANT_ORDER,
    configure_matplotlib,
    finish_axes,
    save_figure,
)


DISPLAY_NAMES = {
    "FullPipeline": "Full pipeline",
    "ReplaceAlignment": "Replace alignment",
    "ReplacePeakPicking": "Replace peak picking",
}


def plot_combined_accuracy(data_path: Path, output_base: Path) -> str:
    font_family = configure_matplotlib()
    data = read_table(data_path)
    datasets = ["QE", "TOF"]
    x = np.arange(len(datasets), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)
    for variant_index, variant in enumerate(VARIANT_ORDER):
        values = [
            float(
                data.loc[
                    (data["dataset"] == dataset) & (data["variant"] == variant),
                    "overall_accuracy",
                ].iloc[0]
            )
            for dataset in datasets
        ]
        offset = (variant_index - 1) * width
        ax.bar(
            x + offset,
            values,
            width=width * 0.82,
            color=VARIANT_COLORS[variant],
            edgecolor="black",
            linewidth=0.6,
            label=DISPLAY_NAMES[variant],
        )
    ax.set_ylim(0.50, 1.00)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks(x, datasets)
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
    save_figure(fig, output_base)
    return font_family
