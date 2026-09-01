from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DPI = 300
FIGURE_SIZE = (16, 9)


COLORS = {
    "ink": "#152235",
    "muted": "#5B6878",
    "line": "#9BA8B5",
    "signal": "#1F5E8C",
    "signal_fill": "#EAF3F8",
    "noise": "#B55C24",
    "noise_fill": "#FBEFE7",
    "data": "#347C6C",
    "data_fill": "#E9F5F1",
    "model": "#5B4B8A",
    "model_fill": "#F0EDF8",
    "eval": "#486F4A",
    "eval_fill": "#EDF5EC",
}


def add_box(ax, xy, width, height, title, body, color, fill):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=color,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.67,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color=COLORS["line"], style="-|>", width=1.6):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=width,
        color=color,
        connectionstyle="arc3,rad=0.0",
        clip_on=False,
    )
    ax.add_patch(arrow)


def add_lane_label(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
    )


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    output_dir = scripts_dir.parent / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overall_architecture_sci.png"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "End-to-End Framework for Population Hyperparameter Inference",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.918,
        "from noisy multi-event gravitational-wave time series",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    ax.plot([0.04, 0.96], [0.875, 0.875], transform=ax.transAxes, color="#D8DEE6", linewidth=1)
    add_lane_label(ax, 0.055, 0.835, "A. Population-level signal simulation", COLORS["signal"])
    add_lane_label(ax, 0.055, 0.575, "B. Noise modeling and signal-preserving augmentation", COLORS["noise"])
    add_lane_label(ax, 0.055, 0.315, "C. Amortized posterior inference and validation", COLORS["model"])

    # Signal-simulation branch.
    add_box(
        ax,
        (0.07, 0.655),
        0.18,
        0.125,
        "Population hyperparameters theta",
        "Prior or Latin-hypercube sampling\n(redshift and mass-population parameters)",
        COLORS["signal"],
        COLORS["signal_fill"],
    )
    add_box(
        ax,
        (0.34, 0.655),
        0.18,
        0.125,
        "CBC population simulation",
        "Event rates, intrinsic parameters,\ncoalescence times and source geometry",
        COLORS["signal"],
        COLORS["signal_fill"],
    )
    add_box(
        ax,
        (0.61, 0.655),
        0.18,
        0.125,
        "Multi-event GW injection",
        "Coherent H1/L1 responses\nfor a 2048 s time-series frame",
        COLORS["signal"],
        COLORS["signal_fill"],
    )
    add_arrow(ax, (0.25, 0.718), (0.34, 0.718), COLORS["signal"])
    add_arrow(ax, (0.52, 0.718), (0.61, 0.718), COLORS["signal"])

    # Noise and augmentation branch.
    add_box(
        ax,
        (0.07, 0.405),
        0.18,
        0.125,
        "O4-like detector spectrum",
        "Target ASD/PSD for H1 and L1\nnoise realizations",
        COLORS["noise"],
        COLORS["noise_fill"],
    )
    add_box(
        ax,
        (0.34, 0.405),
        0.18,
        0.125,
        "Independent noise generation",
        "Seeded H1/L1 noise frames\nwith controllable noise amplitude",
        COLORS["noise"],
        COLORS["noise_fill"],
    )
    add_box(
        ax,
        (0.61, 0.405),
        0.18,
        0.125,
        "Multi-noise augmentation",
        "One signal frame paired with K\nindependent noise realizations",
        COLORS["data"],
        COLORS["data_fill"],
    )
    add_arrow(ax, (0.25, 0.468), (0.34, 0.468), COLORS["noise"])
    add_arrow(ax, (0.52, 0.468), (0.61, 0.468), COLORS["noise"])
    add_arrow(ax, (0.70, 0.655), (0.70, 0.53), COLORS["signal"], width=1.8)

    add_box(
        ax,
        (0.84, 0.485),
        0.115,
        0.16,
        "Noisy two-detector data",
        "d_H1, d_L1\nSignal + noise\npaired labels theta",
        COLORS["data"],
        COLORS["data_fill"],
    )
    add_arrow(ax, (0.79, 0.718), (0.84, 0.585), COLORS["data"], width=1.8)
    add_arrow(ax, (0.79, 0.468), (0.84, 0.545), COLORS["data"], width=1.8)

    # Inference branch.
    add_box(
        ax,
        (0.07, 0.145),
        0.18,
        0.125,
        "Preprocessing",
        "Segmentation, H1/L1 pairing,\ntraining-set normalization",
        COLORS["data"],
        COLORS["data_fill"],
    )
    add_box(
        ax,
        (0.34, 0.145),
        0.18,
        0.125,
        "Conditional normalizing flow",
        "Two-detector context encoder\nand conditional density estimator",
        COLORS["model"],
        COLORS["model_fill"],
    )
    add_box(
        ax,
        (0.61, 0.145),
        0.18,
        0.125,
        "Joint posterior",
        "p(theta | d_H1, d_L1)\nMarginal and joint constraints",
        COLORS["model"],
        COLORS["model_fill"],
    )
    add_box(
        ax,
        (0.84, 0.145),
        0.115,
        0.125,
        "Validation",
        "MAE, NLL, coverage,\ncredible intervals, robustness",
        COLORS["eval"],
        COLORS["eval_fill"],
    )
    add_arrow(ax, (0.895, 0.485), (0.16, 0.27), COLORS["data"], width=1.8)
    add_arrow(ax, (0.25, 0.208), (0.34, 0.208), COLORS["model"])
    add_arrow(ax, (0.52, 0.208), (0.61, 0.208), COLORS["model"])
    add_arrow(ax, (0.79, 0.208), (0.84, 0.208), COLORS["eval"])

    ax.text(
        0.5,
        0.055,
        "Key design: signal-preserving multi-noise augmentation improves robustness to noise realizations\n"
        "while preserving the population-level label theta for each simulated signal.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
        linespacing=1.4,
    )

    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
