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
    label_only_path = output_dir / "label_stats_only_v1_a.npz"

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with np.load(stats_path, allow_pickle=False) as stats:
        if "label_mean" not in stats or "label_std" not in stats:
            raise KeyError(f"label_mean/label_std not found in: {stats_path}")

        np.savez(
            label_only_path,
            label_mean=stats["label_mean"],
            label_std=stats["label_std"],
        )

    print(f"Source stats: {stats_path}")
    print(f"Saved label-only stats: {label_only_path}")


if __name__ == "__main__":
    main()
