from pathlib import Path

import numpy as np


VERSION = "v63"
FILE_INDEX = 49588


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "parameter_sampling_train" / VERSION
    file_path = base_dir / f"CBC_params_example{FILE_INDEX:05d}.npz"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)

    print(f"Version: {VERSION}")
    print(f"File index: {FILE_INDEX}")
    print(f"File path: {file_path}")
    print()

    for key in sorted(data.files):
        value = data[key]
        if np.isscalar(value) or value.shape == ():
            print(f"{key}: {value.item()}")
        else:
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            preview = value[:5] if value.ndim > 0 else value
            print(f"  first values: {preview}")


if __name__ == "__main__":
    main()
