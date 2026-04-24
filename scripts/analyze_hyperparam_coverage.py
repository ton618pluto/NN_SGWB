from __future__ import annotations

from pathlib import Path

import numpy as np


PARAMETER_ORDER = [
    "zp",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
]

PARAMETER_RANGES = {
    "zp": (1.4, 2.4),
    "alpha_m": (3.0, 3.5),
    "m_max": (10.0, 100.0),
    "delta_m": (0.0, 10.0),
    "m_min": (2.0, 10.0),
    "lambda_peak": (0.02, 0.06),
    "mu_m": (33.2, 35.4),
    "sigma_m": (3.52, 3.60),
    "beta_q": (-5.0, 5.0),
}

DEFAULT_SAMPLE_COUNTS = [100, 300, 1000]
MONTE_CARLO_TRIALS = 8
RNG_SEED = 42


def normalize_points(points: np.ndarray) -> np.ndarray:
    normalized = np.empty_like(points, dtype=np.float64)
    for index, name in enumerate(PARAMETER_ORDER):
        low, high = PARAMETER_RANGES[name]
        normalized[:, index] = (points[:, index] - low) / (high - low)
    return normalized


def nearest_neighbor_stats(points: np.ndarray) -> dict[str, float]:
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("points must have shape [n_samples, n_dims] with n_samples >= 2")

    diff = points[:, None, :] - points[None, :, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=2))
    chebyshev = np.max(np.abs(diff), axis=2)

    np.fill_diagonal(euclidean, np.inf)
    np.fill_diagonal(chebyshev, np.inf)

    nn_euclidean = np.min(euclidean, axis=1)
    nn_chebyshev = np.min(chebyshev, axis=1)

    return {
        "nn_euclidean_mean": float(nn_euclidean.mean()),
        "nn_euclidean_median": float(np.median(nn_euclidean)),
        "nn_chebyshev_mean": float(nn_chebyshev.mean()),
        "nn_chebyshev_median": float(np.median(nn_chebyshev)),
    }


def simulate_uniform_coverage(n_samples: int, n_dims: int, trials: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    aggregate = []
    for _ in range(trials):
        synthetic = rng.random((n_samples, n_dims), dtype=np.float64)
        aggregate.append(nearest_neighbor_stats(synthetic))

    keys = aggregate[0].keys()
    return {key: float(np.mean([item[key] for item in aggregate])) for key in keys}


def load_actual_v3_samples(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        columns = [np.asarray(data[name], dtype=np.float64) for name in PARAMETER_ORDER]
    return np.stack(columns, axis=1)


def print_parameter_ranges() -> None:
    print("Tracked parameter ranges (aligned with current v6-style training labels):")
    for name in PARAMETER_ORDER:
        low, high = PARAMETER_RANGES[name]
        print(f"  {name:<12} [{low:.6g}, {high:.6g}]   width={high - low:.6g}")


def print_sample_count_summary(sample_counts: list[int]) -> None:
    n_dims = len(PARAMETER_ORDER)
    print("\nCoverage estimates in normalized unit hypercube:")
    print(
        f"{'n_samples':<10}"
        f"{'grid/dim':>12}"
        f"{'cell_width':>12}"
        f"{'nn_l2_mean':>14}"
        f"{'nn_linf_mean':>16}"
    )
    print("-" * 64)

    for n_samples in sample_counts:
        grid_per_dim = n_samples ** (1.0 / n_dims)
        cell_width = 1.0 / grid_per_dim
        summary = simulate_uniform_coverage(
            n_samples=n_samples,
            n_dims=n_dims,
            trials=MONTE_CARLO_TRIALS,
            seed=RNG_SEED + n_samples,
        )
        print(
            f"{n_samples:<10}"
            f"{grid_per_dim:>12.3f}"
            f"{cell_width:>12.3f}"
            f"{summary['nn_euclidean_mean']:>14.3f}"
            f"{summary['nn_chebyshev_mean']:>16.3f}"
        )

    print("\nInterpretation:")
    print("  grid/dim  : if points were spread like a regular grid, this is the effective points per dimension.")
    print("  cell_width: normalized spacing per dimension in [0, 1]. Smaller means denser coverage.")
    print("  nn_l2_mean: average nearest-neighbor Euclidean distance in normalized 9D space.")
    print("  nn_linf_mean: average nearest-neighbor max-axis distance in normalized 9D space.")


def print_actual_sample_summary(actual_points: np.ndarray, source_path: Path) -> None:
    normalized = normalize_points(actual_points)
    stats = nearest_neighbor_stats(normalized)

    print(f"\nActual v3 sample file: {source_path}")
    print(f"  n_samples: {len(actual_points)}")
    print(f"  parameter_dims: {actual_points.shape[1]}")
    print(f"  normalized nearest-neighbor mean (L2):   {stats['nn_euclidean_mean']:.3f}")
    print(f"  normalized nearest-neighbor median (L2): {stats['nn_euclidean_median']:.3f}")
    print(f"  normalized nearest-neighbor mean (Linf): {stats['nn_chebyshev_mean']:.3f}")
    print(f"  normalized nearest-neighbor median (Linf): {stats['nn_chebyshev_median']:.3f}")

    print("\nActual marginal coverage against configured ranges:")
    print(
        f"{'param':<12}"
        f"{'sample_min':>14}"
        f"{'sample_max':>14}"
        f"{'range_min':>14}"
        f"{'range_max':>14}"
    )
    print("-" * 68)
    for index, name in enumerate(PARAMETER_ORDER):
        sample_values = actual_points[:, index]
        range_min, range_max = PARAMETER_RANGES[name]
        print(
            f"{name:<12}"
            f"{sample_values.min():>14.6f}"
            f"{sample_values.max():>14.6f}"
            f"{range_min:>14.6f}"
            f"{range_max:>14.6f}"
        )


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    actual_v3_path = scripts_dir / "joint_hyperparams_train" / "v3" / "joint_hyperparams_train_v3.npz"

    print("Hyperparameter coverage analyzer")
    print_parameter_ranges()
    print_sample_count_summary(DEFAULT_SAMPLE_COUNTS)

    if actual_v3_path.exists():
        actual_points = load_actual_v3_samples(actual_v3_path)
        print_actual_sample_summary(actual_points, actual_v3_path)
    else:
        print(f"\nActual sample file not found: {actual_v3_path}")


if __name__ == "__main__":
    main()
