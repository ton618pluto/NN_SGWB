from pathlib import Path

import numpy as np


VERSION = "v62"
MAX_FILES = 6
MAX_VALUES_PER_KEY = 6


def print_npz_file(npz_path: Path) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        print(f"File: {npz_path}")
        print(f"Keys: {data.files}")

        with np.printoptions(precision=6, suppress=True, linewidth=160):
            for key in data.files:
                values = data[key]
                preview = values[:MAX_VALUES_PER_KEY] if values.ndim > 0 else values
                print(f"  {key} | shape={values.shape} | dtype={values.dtype}")
                print(f"    first_{MAX_VALUES_PER_KEY}: {preview}")
        print()


def main() -> None:
    version_dir = Path(__file__).parent / "joint_hyperparams_train" / VERSION
    npz_files = sorted(version_dir.glob("*.npz"))

    print(f"Version: {VERSION}")
    print(f"Directory: {version_dir}")

    if not version_dir.exists():
        print("Directory does not exist.")
        return

    if not npz_files:
        print("No .npz files found.")
        return

    print(f"Found {len(npz_files)} .npz file(s).")
    print()

    for npz_path in npz_files[:MAX_FILES]:
        print_npz_file(npz_path)


if __name__ == "__main__":
    main()
