from pathlib import Path

import numpy as np


VERSION = "v64"
FILE_INDEX_1 = 0
FILE_INDEX_2 = 100000

HYPERPARAMETER_NAMES = (
    "R0",
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


def load_parameter_file(version: str, file_index: int) -> tuple[Path, dict[str, float]]:
    base_dir = Path(__file__).resolve().parent / "parameter_sampling_train" / version
    file_path = base_dir / f"CBC_params_example{file_index:05d}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    npz = np.load(file_path, allow_pickle=True)
    values: dict[str, float] = {}
    for name in HYPERPARAMETER_NAMES:
        true_key = f"true_{name}"
        if true_key in npz:
            values[name] = float(npz[true_key])
        elif name in npz:
            values[name] = float(npz[name])
        else:
            values[name] = float("nan")
    return file_path, values


def main() -> None:
    file_path_1, values_1 = load_parameter_file(VERSION, FILE_INDEX_1)
    file_path_2, values_2 = load_parameter_file(VERSION, FILE_INDEX_2)

    print(f"Version: {VERSION}")
    print(f"File 1: {file_path_1}")
    print(f"File 2: {file_path_2}")
    print()
    print(f"{'parameter':<15} {'file1':>18} {'file2':>18}")
    print("-" * 55)
    for name in HYPERPARAMETER_NAMES:
        print(f"{name:<15} {values_1[name]:>18.6f} {values_2[name]:>18.6f}")


if __name__ == "__main__":
    main()
