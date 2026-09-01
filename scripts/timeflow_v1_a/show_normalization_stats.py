from pathlib import Path

import numpy as np


DATA_VERSION = "v100_random_two_noise_100"
USE_MODEL2 = False


def build_output_dir() -> Path:
    scripts_dir = Path(__file__).resolve().parent
    output_name = f"outputs_dataset_{DATA_VERSION}"
    if USE_MODEL2:
        output_name += "_model2"
    return scripts_dir / output_name


def main() -> None:
    output_dir = build_output_dir()
    stats_path = output_dir / "normalization_stats_v1_a.npz"

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with np.load(stats_path, allow_pickle=False) as stats:
        print(f"Stats file: {stats_path}")
        print()

        for key in ["label_mean", "label_std", "feature_mean", "feature_std"]:
            if key not in stats:
                print(f"{key}: <missing>")
                print()
                continue

            value = np.asarray(stats[key])
            print(f"{key}:")
            print(f"  shape = {value.shape}")
            print(f"  dtype = {value.dtype}")
            print(f"  values = {value.tolist()}")
            print()


if __name__ == "__main__":
    main()
