from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
FOLDER = "processed_data_train"
VERSION = "v30_ns"
TARGET_DIRS = {
    "H1": SCRIPTS_DIR / FOLDER / VERSION / "H1_splits",
    "L1": SCRIPTS_DIR / FOLDER / VERSION / "L1_splits",
}


def first_npy_file(target_dir: Path) -> Path | None:
    for path in sorted(target_dir.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix == ".npy":
            return path
    return None


def main() -> None:
    print(f"folder: {FOLDER}")
    print(f"version: {VERSION}")

    for detector, target_dir in TARGET_DIRS.items():
        print(f"\n[{detector}] directory: {target_dir}")
        if not target_dir.exists():
            print("  Directory not found.")
            continue

        sample_path = first_npy_file(target_dir)
        if sample_path is None:
            print("  No valid .npy files found.")
            continue

        sample = np.load(sample_path, allow_pickle=True).item()
        data = np.asarray(sample["data"])
        print(f"  sample file: {sample_path.name}")
        print(f"  data shape: {tuple(data.shape)}")


if __name__ == "__main__":
    main()
