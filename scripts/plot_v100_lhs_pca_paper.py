from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

# False reproduces the raw-parameter PCA style used by PCA_test.png. In this
# mode, parameters with wider physical ranges contribute more strongly to the
# first principal components. Set True for equal-weight standardized PCA.
STANDARDIZE_PARAMETERS_BEFORE_PCA = False

# alpha_m, delta_m, and beta_q have the same prior width and therefore nearly
# equal variance. Retain alpha_m as the representative member of this group so
# that PC2 is unique instead of an arbitrary rotation of three tied directions.
# The PCA below is a genuine PCA of these six explicitly selected parameters.
PCA_INCLUDED_PARAMETER_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "mu_m",
    "sigma_m",
)

# PCA is fitted with all available samples, while a deterministic subset is
# displayed. Larger opaque markers reproduce the dense appearance of the
# reference figure without putting all 300,000 points into the PDF raster.
PCA_SCATTER_SAMPLE_LIMIT: int | None = 25_000
PCA_SCATTER_RANDOM_SEED = 2026
PCA_SCATTER_POINT_SIZE = 1.25
PCA_SCATTER_ALPHA = 1.0

# Try the publication-quality setting first, then reduce only the rasterized
# scatter layer if necessary to keep the final PDF below the requested limit.
OUTPUT_DPI_CANDIDATES = (600, 500, 450, 400, 350, 300, 250, 200)
MAX_OUTPUT_FILE_SIZE_BYTES = 1_000_000
FIGURE_SIZE_INCHES = (5.7, 4.65)

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


def compute_pca(
    samples: np.ndarray,
    parameter_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameter_indices = [PCA_PARAMETER_NAMES.index(name) for name in parameter_names]
    pca_samples = samples[:, parameter_indices]
    means = np.mean(pca_samples, axis=0)
    standard_deviations = np.std(pca_samples, axis=0, ddof=1)
    if np.any(standard_deviations <= 0.0):
        constant_names = [
            parameter_names[index]
            for index in np.flatnonzero(standard_deviations <= 0.0)
        ]
        raise ValueError(f"Zero-variance PCA parameters: {constant_names}")

    centred = pca_samples - means
    if STANDARDIZE_PARAMETERS_BEFORE_PCA:
        transformed = centred / standard_deviations
    else:
        transformed = centred

    covariance_matrix = np.cov(transformed, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fix arbitrary eigenvector signs for reproducible figures across platforms.
    for component_index in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, component_index]
        largest_loading_index = int(np.argmax(np.abs(vector)))
        if vector[largest_loading_index] < 0.0:
            eigenvectors[:, component_index] *= -1.0

    scores = transformed @ eigenvectors
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return scores, explained_variance_ratio, eigenvectors


def plot_projection(
    scores: np.ndarray,
    x_axis_label: str,
    y_axis_label: str,
) -> tuple[int, int]:
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
            "pdf.compression": 9,
            "ps.fonttype": 42,
        }
    )

    fig, pca_axis = plt.subplots(
        figsize=FIGURE_SIZE_INCHES,
    )

    if (
        PCA_SCATTER_SAMPLE_LIMIT is None
        or PCA_SCATTER_SAMPLE_LIMIT >= scores.shape[0]
    ):
        displayed_scores = scores
    else:
        random_generator = np.random.default_rng(PCA_SCATTER_RANDOM_SEED)
        displayed_indices = random_generator.choice(
            scores.shape[0],
            size=PCA_SCATTER_SAMPLE_LIMIT,
            replace=False,
        )
        displayed_scores = scores[displayed_indices]

    pca_axis.scatter(
        displayed_scores[:, 0],
        displayed_scores[:, 1],
        s=PCA_SCATTER_POINT_SIZE,
        c="#161616",
        alpha=PCA_SCATTER_ALPHA,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )
    pca_axis.set_xlabel(x_axis_label)
    pca_axis.set_ylabel(y_axis_label)
    pca_axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    pca_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    pca_axis.grid(
        which="major",
        color="#C6C8CD",
        linestyle="--",
        linewidth=0.58,
        alpha=0.48,
        zorder=0,
    )
    pca_axis.spines["top"].set_visible(False)
    pca_axis.spines["right"].set_visible(False)
    pca_axis.spines["left"].set_color("#555555")
    pca_axis.spines["bottom"].set_color("#555555")
    pca_axis.tick_params(colors="#3F3F3F", length=3.8, width=0.75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.14, top=0.975)
    used_dpi = OUTPUT_DPI_CANDIDATES[-1]
    output_size_bytes = 0
    for candidate_dpi in OUTPUT_DPI_CANDIDATES:
        fig.savefig(
            OUTPUT_PATH,
            dpi=candidate_dpi,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        output_size_bytes = OUTPUT_PATH.stat().st_size
        used_dpi = candidate_dpi
        if output_size_bytes <= MAX_OUTPUT_FILE_SIZE_BYTES:
            break
    plt.close(fig)
    if output_size_bytes > MAX_OUTPUT_FILE_SIZE_BYTES:
        raise RuntimeError(
            f"PDF size is {output_size_bytes} bytes after trying "
            f"{OUTPUT_DPI_CANDIDATES}; requested maximum is "
            f"{MAX_OUTPUT_FILE_SIZE_BYTES} bytes."
        )
    print(
        "Displayed scatter samples: "
        f"{displayed_scores.shape[0]} / {scores.shape[0]} "
        "(projection statistics use all samples)"
    )
    return used_dpi, output_size_bytes


def print_pca_summary(
    explained_variance_ratio: np.ndarray,
    eigenvectors: np.ndarray,
    parameter_names: tuple[str, ...],
) -> None:
    print("Explained variance ratio:")
    for index, ratio in enumerate(explained_variance_ratio, start=1):
        print(f"  PC{index}: {ratio:.8f} ({100.0 * ratio:.4f}%)")

    print("PC1 and PC2 loadings:")
    print(f"  {'parameter':<12s} {'PC1':>12s} {'PC2':>12s}")
    for parameter_index, name in enumerate(parameter_names):
        print(
            f"  {name:<12s} "
            f"{eigenvectors[parameter_index, 0]:+12.7f} "
            f"{eigenvectors[parameter_index, 1]:+12.7f}"
        )


def main() -> None:
    samples = load_lhs_samples(INPUT_PATH)
    print(f"Input file: {INPUT_PATH}")
    print(f"Samples: {samples.shape[0]}")
    validate_and_report_ranges(samples)

    preprocessing = (
        "standardized parameters (equal parameter weighting)"
        if STANDARDIZE_PARAMETERS_BEFORE_PCA
        else "raw centred physical parameters (range-weighted PCA)"
    )
    excluded_parameters = [
        name
        for name in PCA_PARAMETER_NAMES
        if name not in PCA_INCLUDED_PARAMETER_NAMES
    ]
    print(f"PCA dimensions: {len(PCA_INCLUDED_PARAMETER_NAMES)}")
    print(f"PCA included parameters: {list(PCA_INCLUDED_PARAMETER_NAMES)}")
    print(f"PCA excluded tied parameters: {excluded_parameters}")
    print(f"PCA preprocessing: {preprocessing}")
    scores, explained_variance_ratio, eigenvectors = compute_pca(
        samples,
        PCA_INCLUDED_PARAMETER_NAMES,
    )
    print_pca_summary(
        explained_variance_ratio,
        eigenvectors,
        PCA_INCLUDED_PARAMETER_NAMES,
    )
    x_axis_label = f"PC1 ({100.0 * explained_variance_ratio[0]:.2f}% variance)"
    y_axis_label = f"PC2 ({100.0 * explained_variance_ratio[1]:.2f}% variance)"

    used_dpi, output_size_bytes = plot_projection(
        scores,
        x_axis_label,
        y_axis_label,
    )
    print(f"Saved LHS projection figure: {OUTPUT_PATH}")
    print(f"PDF scatter raster: {used_dpi} dpi")
    print(
        f"PDF size: {output_size_bytes} bytes "
        f"({output_size_bytes / 1_000_000.0:.3f} MB)"
    )


if __name__ == "__main__":
    main()
