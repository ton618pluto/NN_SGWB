from pathlib import Path

import numpy as np


SOURCE_VERSION = "v61"
DETECTOR = "H1"
SAMPLE_INDEX = 0
OUTPUT_VERSION = "fixed_one_test"
N_SAMPLES = 1

PARAMETER_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)


def get_split_dir(base_dir: Path, detector: str) -> Path:
    detector = detector.upper()
    split_dir = base_dir / f"{detector}_splits"
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return split_dir


def load_test_sample_hyperparameters(
    source_version: str = SOURCE_VERSION,
    sample_index: int = SAMPLE_INDEX,
    detector: str = DETECTOR,
) -> tuple[dict[str, float], Path]:
    base_dir = Path(__file__).parent / "processed_data_train" / source_version
    split_dir = get_split_dir(base_dir, detector)
    npy_files = sorted(split_dir.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {split_dir}")
    if sample_index < 0 or sample_index >= len(npy_files):
        raise IndexError(
            f"SAMPLE_INDEX out of range: {sample_index}. "
            f"Available files: 0 to {len(npy_files) - 1}"
        )

    file_path = npy_files[sample_index]
    sample = np.load(file_path, allow_pickle=True).item()
    label = sample["label"]

    if isinstance(label, dict):
        hyperparameters = {name: float(label[name]) for name in PARAMETER_NAMES}
    else:
        label_array = np.asarray(label, dtype=np.float64).reshape(-1)
        if label_array.size != len(PARAMETER_NAMES):
            raise ValueError(
                f"Label length mismatch for {file_path.name}: "
                f"expected {len(PARAMETER_NAMES)}, got {label_array.size}"
            )
        hyperparameters = {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, label_array)
        }

    return hyperparameters, file_path


def sample_fixed_hyperparameters_dict(
    fixed_values: dict[str, float],
    n_samples: int = N_SAMPLES,
) -> dict[str, np.ndarray]:
    return {
        name: np.full(n_samples, value, dtype=np.float64)
        for name, value in fixed_values.items()
    }


def save_fixed_hyperparameters(
    output_path: Path,
    fixed_values: dict[str, float],
    n_samples: int = N_SAMPLES,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = sample_fixed_hyperparameters_dict(fixed_values, n_samples=n_samples)
    np.savez(output_path, **samples)
    return output_path


if __name__ == "__main__":
    fixed_values, source_file = load_test_sample_hyperparameters()
    output_version_name = f"{OUTPUT_VERSION}_{SOURCE_VERSION}"
    output_path = (
        Path(__file__).parent
        / "joint_hyperparams_test"
        / output_version_name
        / f"joint_hyperparams_test_{output_version_name}.npz"
    )

    saved_path = save_fixed_hyperparameters(output_path, fixed_values, n_samples=N_SAMPLES)
    print(f"Loaded source file: {source_file}")
    print(f"Saved fixed hyperparameters to: {saved_path}")
    print("Fixed values:")
    for key, value in fixed_values.items():
        print(f"  {key}: {value}")
