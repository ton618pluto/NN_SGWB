from __future__ import annotations

from pathlib import Path

import numpy as np


VERSION = "v70"
GROUP_INDEX = 199999

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


def main() -> None:
    file_path = (
        Path(__file__).resolve().parent
        / "joint_hyperparams_train"
        / VERSION
        / f"joint_hyperparams_train_{VERSION}.npz"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path)

    first_name = PARAMETER_NAMES[0]
    total_groups = int(np.asarray(data[first_name]).shape[0])
    if GROUP_INDEX < 0 or GROUP_INDEX >= total_groups:
        raise IndexError(
            f"GROUP_INDEX out of range: {GROUP_INDEX}. "
            f"Available indices: 0 to {total_groups - 1}"
        )

    print(f"File: {file_path}")
    print(f"Version: {VERSION}")
    print(f"Group index: {GROUP_INDEX}")
    print()

    for name in PARAMETER_NAMES:
        if name not in data:
            print(f"{name}: <missing>")
            continue
        values = np.asarray(data[name])
        print(f"{name}: {float(values[GROUP_INDEX])}")


if __name__ == "__main__":
    main()
