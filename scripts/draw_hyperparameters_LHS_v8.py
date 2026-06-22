from pathlib import Path

import numpy as np


PARAMETER_NAMES = (
    "alpha_z",
    "beta_z",
    "zp",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)


def madau_dickinson_sfr(z, alpha, beta, zp_target, psi_0=0.015):
    alpha = np.maximum(1e-6, alpha)
    beta = np.maximum(1e-6, beta)

    pivot = (1 + zp_target) * (beta / alpha) ** (1 / (alpha + beta))
    numerator = (1 + z) ** alpha
    denominator = 1 + ((1 + z) / pivot) ** (alpha + beta)
    return psi_0 * (numerator / denominator)


def latin_hypercube_unit(n_samples: int, n_dims: int, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, n_dims), dtype=np.float64)

    for dim in range(n_dims):
        permutation = rng.permutation(n_samples)
        jitter = rng.random(n_samples)
        samples[:, dim] = (permutation + jitter) / n_samples

    return samples


def scale_unit_to_range(unit_values: np.ndarray, low: float, high: float) -> np.ndarray:
    if low == high:
        return np.full(unit_values.shape, low, dtype=np.float64)
    return low + (high - low) * unit_values


def sample_joint_hyperparameters_separate(n_samples=10000, seed=None):
    # Independent LHS dimensions:
    # zp, alpha_m, delta_m, mu_m, sigma_m, beta_q
    lhs_unit = latin_hypercube_unit(n_samples=n_samples, n_dims=6, seed=seed)

    zp = scale_unit_to_range(lhs_unit[:, 0], 0.85, 2.85)

    # Keep the same negative slope as v4, but force the line to pass
    # through the fiducial point (zp=1.9, alpha_z=2.7).
    slope = (2.1 - 3.6) / (2.4 - 1.4)
    intercept = 2.7 - slope * 1.9
    alpha_z = slope * zp + intercept
    beta_z = np.maximum(1e-6, (2.7 + 2.9) - alpha_z)

    alpha_m = scale_unit_to_range(lhs_unit[:, 1], -3.0, 7.0)
    m_max = np.full(n_samples, 100.0, dtype=np.float64)
    delta_m = scale_unit_to_range(lhs_unit[:, 2], 0.0, 10.0)
    m_min = np.full(n_samples, 2.0, dtype=np.float64)
    lambda_peak = np.full(n_samples, 0.04, dtype=np.float64)
    mu_m = scale_unit_to_range(lhs_unit[:, 3], 25.0, 100.0)
    sigma_m = scale_unit_to_range(lhs_unit[:, 4], 0.01, 5.0)
    beta_q = scale_unit_to_range(lhs_unit[:, 5], -5.0, 5.0)

    return (
        alpha_z,
        beta_z,
        zp,
        alpha_m,
        m_max,
        delta_m,
        m_min,
        lambda_peak,
        mu_m,
        sigma_m,
        beta_q,
    )


def sample_joint_hyperparameters_dict(n_samples=10000, seed=None):
    return dict(zip(PARAMETER_NAMES, sample_joint_hyperparameters_separate(n_samples, seed)))


def save_joint_hyperparameters(output_path, n_samples=100, seed=42):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = sample_joint_hyperparameters_dict(n_samples=n_samples, seed=seed)
    np.savez(output_path, **samples)
    return output_path


if __name__ == "__main__":
    n_samples = 200000
    seed = 42
    version = "v80"
    output_path = Path(__file__).parent / "joint_hyperparams_train" / version / f"joint_hyperparams_train_{version}.npz"

    saved_path = save_joint_hyperparameters(output_path, n_samples=n_samples, seed=seed)
    print(f"Saved samples to: {saved_path}")
