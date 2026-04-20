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


def calculate_beta_to_match_peak(zp, alpha, target_peak_height, psi_0=0.015):
    alpha = np.maximum(1e-6, alpha)
    numerator_at_peak = psi_0 * (1 + zp) ** alpha
    ratio = numerator_at_peak / target_peak_height

    beta = np.where(
        ratio > 1.0,
        alpha / (ratio - 1.0),
        10.0,
    )
    return np.maximum(1e-6, beta)


def _build_rngs(seed=None):
    stream_names = (
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
    base_sequence = np.random.SeedSequence(seed)
    child_sequences = base_sequence.spawn(len(stream_names))
    return {
        name: np.random.default_rng(child_sequence)
        for name, child_sequence in zip(stream_names, child_sequences)
    }


def sample_joint_hyperparameters_separate(n_samples=10000, seed=None):
    rngs = _build_rngs(seed)

    zp = rngs["zp"].uniform(1.4, 2.4, n_samples)
    slope = (2.1 - 3.6) / (2.4 - 1.4)
    intercept = 3.6 - slope * 1.4
    alpha_z = slope * zp + intercept

    zp_fid, a_fid, b_fid = 1.9, 2.7, 2.9
    psi_0 = 0.015
    target_peak_height = madau_dickinson_sfr(zp_fid, a_fid, b_fid, zp_fid, psi_0)
    beta_z = calculate_beta_to_match_peak(zp, alpha_z, target_peak_height, psi_0)

    alpha_m = rngs["alpha_m"].uniform(3.0, 3.5, n_samples)
    m_max = rngs["m_max"].uniform(10.0, 100.0, n_samples)
    delta_m = rngs["delta_m"].uniform(0.0, 10.0, n_samples)
    m_min = rngs["m_min"].uniform(2.0, 10.0, n_samples)
    lambda_peak = rngs["lambda_peak"].uniform(0.02, 0.06, n_samples)
    mu_m = rngs["mu_m"].uniform(33.2, 35.4, n_samples)
    sigma_m = rngs["sigma_m"].uniform(3.52, 3.60, n_samples)
    beta_q = rngs["beta_q"].uniform(-5.0, 5.0, n_samples)

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


def check_prefix_consistency(prefix_samples=100, total_samples=200, seed=42):
    small = sample_joint_hyperparameters_dict(n_samples=prefix_samples, seed=seed)
    large = sample_joint_hyperparameters_dict(n_samples=total_samples, seed=seed)

    return {
        name: np.array_equal(small[name], large[name][:prefix_samples])
        for name in PARAMETER_NAMES
    }


if __name__ == "__main__":
    n_samples = 100
    seed = 42
    output_path = Path(__file__).parent / "joint_hyperparams_train" / "v3" / "joint_hyperparams_train_v3.npz"

    saved_path = save_joint_hyperparameters(output_path, n_samples=n_samples, seed=seed)
    print(f"Saved samples to: {saved_path}")

    prefix_check = check_prefix_consistency(prefix_samples=100, total_samples=200, seed=seed)
    print("Prefix consistency check:")
    for name, is_consistent in prefix_check.items():
        print(f"  {name}: {is_consistent}")
