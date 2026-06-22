from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
FOLDER = "processed_data_freq_features"
VERSION = "v21"
FS = 64
NPERSEG = 64


def build_target_dir() -> Path:
    return SCRIPTS_DIR / FOLDER / VERSION / f"fs{FS}_nperseg{NPERSEG}"


def count_files(target_dir: Path) -> tuple[int, int]:
    total = 0
    empty = 0
    for path in target_dir.iterdir():
        if not path.is_file() or path.suffix != ".npy":
            continue
        total += 1
        if path.stat().st_size == 0:
            empty += 1
    return total, empty


def main() -> None:
    target_dir = build_target_dir()
    print(f"folder: {FOLDER}")
    print(f"version: {VERSION}")
    print(f"fs: {FS}")
    print(f"nperseg: {NPERSEG}")
    print(f"target_dir: {target_dir}")

    if not target_dir.exists():
        print("Directory not found.")
        return

    total, empty = count_files(target_dir)
    print(f"Total .npy files: {total}")
    print(f"Empty .npy files: {empty}")


if __name__ == "__main__":
    main()
