from pathlib import Path

import numpy as np


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


def build_fixed_hyperparameters():
    zp = 2.1

    # Keep the same corrected relation as draw_hyperparameters_LHS_v2.py:
    # alpha_z = -1.5 * zp + 5.55, so that zp=1.9 -> alpha_z=2.7.
    alpha_z = -1.5 * zp + 5.55

    zp_fid, a_fid, b_fid = 1.9, 2.7, 2.9
    psi_0 = 0.015
    target_peak_height = madau_dickinson_sfr(zp_fid, a_fid, b_fid, zp_fid, psi_0)
    beta_z = float(calculate_beta_to_match_peak(np.array([zp]), np.array([alpha_z]), target_peak_height, psi_0)[0])

    return {
        "alpha_z": alpha_z,
        "beta_z": beta_z,
        "zp": zp,
        "alpha_m": 3.4,
        "m_max": 100.0,
        "delta_m": 4.83,
        "m_min": 5.08,
        "lambda_peak": 0.04,
        "mu_m": 33.73,
        "sigma_m": 3.56,
        "beta_q": 1.08,
    }


def sample_fixed_hyperparameters_dict(n_samples=1):
    fixed_values = build_fixed_hyperparameters()
    return {
        name: np.full(n_samples, value, dtype=np.float64)
        for name, value in fixed_values.items()
    }


def save_fixed_hyperparameters(output_path, n_samples=1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = sample_fixed_hyperparameters_dict(n_samples=n_samples)
    np.savez(output_path, **samples)
    return output_path


if __name__ == "__main__":
    n_samples = 1
    version = "v0000"
    output_path = (
        Path(__file__).parent
        / "joint_hyperparams_train"
        / version
        / f"joint_hyperparams_train_{version}.npz"
    )

    saved_path = save_fixed_hyperparameters(output_path, n_samples=n_samples)
    print(f"Saved fixed hyperparameters to: {saved_path}")
    print("Fixed values:")
    for key, value in build_fixed_hyperparameters().items():
        print(f"  {key}: {value}")
