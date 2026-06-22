from pathlib import Path


VERSION = "v63"
BASE_DIR = Path(__file__).resolve().parent / "processed_data_train" / VERSION


def list_npy_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix == ".npy")


def find_empty_files(files: list[Path]) -> list[Path]:
    return [path for path in files if path.stat().st_size == 0]


def delete_files(files: list[Path]) -> None:
    for path in files:
        path.unlink(missing_ok=True)


def report_and_delete(detector_name: str, directory: Path) -> None:
    files = list_npy_files(directory)
    empty_files = find_empty_files(files)

    print(f"{detector_name} directory: {directory}")
    print(f"{detector_name} total .npy files: {len(files)}")

    if not empty_files:
        print(f"{detector_name}: no empty .npy files found.")
        return

    print(f"{detector_name} empty .npy files ({len(empty_files)}):")
    for path in empty_files:
        print(f"  {path.name}")

    delete_files(empty_files)
    print(f"{detector_name}: deleted {len(empty_files)} empty .npy files.")


def main() -> None:
    base_dir = BASE_DIR.resolve()
    h1_dir = base_dir / "H1_splits"
    l1_dir = base_dir / "L1_splits"

    print(f"Version: {VERSION}")
    print(f"Base directory: {base_dir}")
    report_and_delete("H1", h1_dir)
    report_and_delete("L1", l1_dir)


if __name__ == "__main__":
    main()
