from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# ---------------------------- User controls ----------------------------
VERSION = "v100"
EXPECTED_SAMPLE_COUNT = 300_000

# Only the eight independently sampled parameters enter the PCA. Parameters
# fixed during LHS generation (m_max, m_min, lambda_peak) are intentionally
# excluded because their variance is zero.
PCA_PARAMETER_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "delta_m",
    "mu_m",
    "sigma_m",
    "beta_q",
)

# Expected v100 LHS ranges. They are used for validation and console reporting;
# the PCA itself always uses the values actually loaded from the NPZ file.
EXPECTED_PARAMETER_RANGES = {
    "zp": (0.85, 2.85),
    "alpha_z": (1.0, 3.0),
    "beta_z": (1.0, 5.0),
    "alpha_m": (-3.0, 7.0),
    "delta_m": (0.0, 10.0),
    "mu_m": (25.0, 50.0),
    "sigma_m": (0.01, 5.0),
    "beta_q": (-5.0, 5.0),
}
RANGE_TOLERANCE_FRACTION = 1.0e-3

HEXBIN_GRID_SIZE = 105
HEXBIN_MIN_COUNT = 1
MARGINAL_HISTOGRAM_BINS = 50
OUTPUT_DPI = 600
FIGURE_SIZE_INCHES = (10.0, 4.35)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    SCRIPT_DIR
    / "joint_hyperparams_train"
    / VERSION
    / f"joint_hyperparams_train_{VERSION}.npz"
)
OUTPUT_DIR = SCRIPT_DIR / "paper_figures"
OUTPUT_PATH = OUTPUT_DIR / f"LHS_PCA_{VERSION}.pdf"


def load_lhs_samples(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"LHS sample file not found: {path}\n"
            "Expected the v100 NPZ file at the path above."
        )

    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in PCA_PARAMETER_NAMES if name not in data.files]
        if missing:
            raise KeyError(
                f"Missing PCA parameters in {path.name}: {missing}. "
                f"Available keys: {data.files}"
            )

        columns = []
        sample_count: int | None = None
        for name in PCA_PARAMETER_NAMES:
            values = np.asarray(data[name], dtype=np.float64).reshape(-1)
            if sample_count is None:
                sample_count = values.size
            elif values.size != sample_count:
                raise ValueError(
                    f"Parameter {name} has {values.size} samples, expected "
                    f"{sample_count}."
                )
            if not np.all(np.isfinite(values)):
                invalid_count = int(np.count_nonzero(~np.isfinite(values)))
                raise ValueError(
                    f"Parameter {name} contains {invalid_count} non-finite values."
                )
            columns.append(values)

    samples = np.column_stack(columns)
    if samples.shape[0] != EXPECTED_SAMPLE_COUNT:
        print(
            f"Warning: loaded {samples.shape[0]} samples; "
            f"EXPECTED_SAMPLE_COUNT={EXPECTED_SAMPLE_COUNT}."
        )
    return samples


def validate_and_report_ranges(samples: np.ndarray) -> None:
    print("Loaded parameter ranges:")
    for column_index, name in enumerate(PCA_PARAMETER_NAMES):
        values = samples[:, column_index]
        actual_min = float(np.min(values))
        actual_max = float(np.max(values))
        expected_min, expected_max = EXPECTED_PARAMETER_RANGES[name]
        width = expected_max - expected_min
        tolerance = max(np.finfo(np.float64).eps, width * RANGE_TOLERANCE_FRACTION)
        within_expected = (
            actual_min >= expected_min - tolerance
            and actual_max <= expected_max + tolerance
        )
        print(
            f"  {name:<10s} actual=[{actual_min:.9g}, {actual_max:.9g}] "
            f"expected=[{expected_min:.9g}, {expected_max:.9g}] "
            f"within_expected={within_expected}"
        )
        if not within_expected:
            raise ValueError(
                f"Observed range of {name} is outside the expected v100 LHS range."
            )


def compute_standardized_pca(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(samples, axis=0)
    standard_deviations = np.std(samples, axis=0, ddof=1)
    if np.any(standard_deviations <= 0.0):
        constant_names = [
            PCA_PARAMETER_NAMES[index]
            for index in np.flatnonzero(standard_deviations <= 0.0)
        ]
        raise ValueError(f"Zero-variance PCA parameters: {constant_names}")

    standardized = (samples - means) / standard_deviations
    correlation_matrix = np.cov(standardized, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fix arbitrary eigenvector signs for reproducible figures across platforms.
    for component_index in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, component_index]
        largest_loading_index = int(np.argmax(np.abs(vector)))
        if vector[largest_loading_index] < 0.0:
            eigenvectors[:, component_index] *= -1.0

    scores = standardized @ eigenvectors
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return scores, explained_variance_ratio, eigenvectors


def plot_pca(
    samples: np.ndarray,
    scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (density_axis, marginal_axis) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_INCHES,
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.32},
    )

    density = density_axis.hexbin(
        scores[:, 0],
        scores[:, 1],
        gridsize=HEXBIN_GRID_SIZE,
        mincnt=HEXBIN_MIN_COUNT,
        norm=LogNorm(),
        cmap="magma_r",
        linewidths=0.0,
        rasterized=True,
    )
    density_axis.set_xlabel(
        f"PC1 ({100.0 * explained_variance_ratio[0]:.2f}% variance)"
    )
    density_axis.set_ylabel(
        f"PC2 ({100.0 * explained_variance_ratio[1]:.2f}% variance)"
    )
    density_axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    density_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    colorbar = fig.colorbar(density, ax=density_axis, pad=0.025, fraction=0.052)
    colorbar.set_label("Samples per hexagonal bin", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=8.8, length=3.0)

    # Compare all marginals on the same [0, 1] coordinate while retaining a
    # separate labelled ridge for every physical parameter.
    bin_edges = np.linspace(0.0, 1.0, MARGINAL_HISTOGRAM_BINS + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    marginal_densities: list[np.ndarray] = []
    for parameter_index, name in enumerate(PCA_PARAMETER_NAMES):
        lower, upper = EXPECTED_PARAMETER_RANGES[name]
        normalized = (samples[:, parameter_index] - lower) / (upper - lower)
        density_values, _ = np.histogram(
            normalized,
            bins=bin_edges,
            density=True,
        )
        marginal_densities.append(density_values)

    maximum_density = max(float(np.max(values)) for values in marginal_densities)
    ridge_scale = 0.58 / max(1.0, maximum_density)
    ridge_colors = (
        "#4C78A8",
        "#D06B5F",
        "#6A9F72",
        "#8A6FA8",
        "#C58A3A",
        "#4F9A9A",
        "#B56586",
        "#7B8174",
    )
    parameter_labels = {
        "zp": r"$z_p$",
        "alpha_z": r"$\alpha_z$",
        "beta_z": r"$\beta_z$",
        "alpha_m": r"$\alpha_m$",
        "delta_m": r"$\delta_m$",
        "mu_m": r"$\mu_m$",
        "sigma_m": r"$\sigma_m$",
        "beta_q": r"$\beta_q$",
    }
    tick_positions = []
    tick_labels = []
    for parameter_index, (name, density_values) in enumerate(
        zip(PCA_PARAMETER_NAMES, marginal_densities)
    ):
        baseline = float(len(PCA_PARAMETER_NAMES) - 1 - parameter_index)
        curve = baseline + ridge_scale * density_values
        color = ridge_colors[parameter_index]
        marginal_axis.fill_between(
            bin_centres,
            baseline,
            curve,
            step="mid",
            color=color,
            alpha=0.20,
            linewidth=0.0,
            zorder=2,
        )
        marginal_axis.plot(
            bin_centres,
            curve,
            drawstyle="steps-mid",
            color=color,
            linewidth=1.15,
            zorder=3,
        )
        marginal_axis.hlines(
            baseline + ridge_scale,
            0.0,
            1.0,
            color="#777777",
            linestyle=(0, (2.2, 2.2)),
            linewidth=0.72,
            alpha=0.82,
            zorder=4,
        )
        tick_positions.append(baseline + 0.5 * ridge_scale)
        tick_labels.append(parameter_labels[name])

    marginal_axis.set_xlim(0.0, 1.0)
    marginal_axis.set_ylim(-0.08, len(PCA_PARAMETER_NAMES) - 0.20)
    marginal_axis.set_xlabel("Normalized parameter value")
    marginal_axis.set_ylabel("Parameter")
    marginal_axis.set_yticks(tick_positions, labels=tick_labels)
    marginal_axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    marginal_axis.legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", linewidth=1.3, label="LHS marginal"),
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=0.8,
                linestyle=(0, (2.2, 2.2)),
                label="Uniform reference",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )

    for axis in (density_axis, marginal_axis):
        axis.grid(
            which="major",
            color="#C6C8CD",
            linestyle="--",
            linewidth=0.58,
            alpha=0.48,
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(colors="#3F3F3F", length=3.8, width=0.75)

    marginal_axis.grid(axis="y", visible=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.925, bottom=0.15, top=0.965)
    fig.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def print_pca_summary(
    explained_variance_ratio: np.ndarray,
    eigenvectors: np.ndarray,
) -> None:
    print("Explained variance ratio:")
    for index, ratio in enumerate(explained_variance_ratio, start=1):
        print(f"  PC{index}: {ratio:.8f} ({100.0 * ratio:.4f}%)")

    print("PC1 and PC2 loadings:")
    print(f"  {'parameter':<12s} {'PC1':>12s} {'PC2':>12s}")
    for parameter_index, name in enumerate(PCA_PARAMETER_NAMES):
        print(
            f"  {name:<12s} "
            f"{eigenvectors[parameter_index, 0]:+12.7f} "
            f"{eigenvectors[parameter_index, 1]:+12.7f}"
        )


def main() -> None:
    samples = load_lhs_samples(INPUT_PATH)
    print(f"Input file: {INPUT_PATH}")
    print(f"Samples: {samples.shape[0]}")
    print(f"PCA dimensions: {samples.shape[1]}")
    validate_and_report_ranges(samples)

    scores, explained_variance_ratio, eigenvectors = compute_standardized_pca(samples)
    print_pca_summary(explained_variance_ratio, eigenvectors)
    plot_pca(samples, scores, explained_variance_ratio)
    print(f"Saved PCA paper figure: {OUTPUT_PATH}")
    print(f"PDF raster elements: {OUTPUT_DPI} dpi")


if __name__ == "__main__":
    main()
