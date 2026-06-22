from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
# folder = "processed_data_superimposed"
folder = "processed_data_train"
#version = "v23"
#version = "v30_s"
#version = "v30_ns"
#version = "v45"
#version = "v61"
#version = "v62"
version = "v63"

TARGET_DIRS = {
    "H1": SCRIPTS_DIR / folder / version / "H1_splits",
    "L1": SCRIPTS_DIR / folder / version / "L1_splits",
}


def count_npy_files(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    return sum(1 for path in target_dir.iterdir() if path.is_file() and path.suffix == ".npy")


def main() -> None:
    total = 0
    print(f"folder: {folder}")
    print(f"version: {version}")
    for detector, target_dir in TARGET_DIRS.items():
        count = count_npy_files(target_dir)
        total += count
        print(f"{detector} .npy file count: {count}")
    print(f"Total .npy file count: {total}")


if __name__ == "__main__":
    main()
