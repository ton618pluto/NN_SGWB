from pathlib import Path

import numpy as np


HYPERPARAMETERS = {
    "alpha_z": 2.7,
    "beta_z": 2.9,
    "zp": 1.9,
    "alpha_m": 4.4,
    "beta_q": 1.08,
    "m_min": 5.08,
    "m_max": 86.85,
    "lambda_peak": 1.04,
    "delta_m": 4.83,
    "mu_m": 34.73,
    "sigma_m": 4.56,
}


def sample_fixed_hyperparameters_dict(n_samples=1):
    return {
        name: np.full(n_samples, value, dtype=np.float64)
        for name, value in HYPERPARAMETERS.items()
    }


def save_fixed_hyperparameters(output_path, n_samples=1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = sample_fixed_hyperparameters_dict(n_samples=n_samples)
    np.savez(output_path, **samples)
    return output_path


if __name__ == "__main__":
    n_samples = 1
    version = "v00000"
    output_path = (
        Path(__file__).parent
        / "joint_hyperparams_train"
        / version
        / f"joint_hyperparams_train_{version}.npz"
    )

    saved_path = save_fixed_hyperparameters(output_path, n_samples=n_samples)
    print(f"Saved fixed hyperparameters to: {saved_path}")
