from pathlib import Path


VERSION = "v8"
BASE_DIR = Path(__file__).resolve().parent / "training_set_superimposed" / VERSION


def list_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(item for item in directory.iterdir() if item.is_file())


def find_empty_files(files: list[Path]) -> list[Path]:
    return [path for path in files if path.stat().st_size == 0]


def report_detector(detector_name: str, directory: Path) -> None:
    files = list_files(directory)
    empty_files = find_empty_files(files)

    print(f"{detector_name} directory: {directory}")
    print(f"{detector_name} total files: {len(files)}")
    print(f"{detector_name} empty files: {len(empty_files)}")


def main() -> None:
    base_dir = BASE_DIR.resolve()
    h1_dir = base_dir / "H1"
    l1_dir = base_dir / "L1"

    print(f"Version: {VERSION}")
    print(f"Base directory: {base_dir}")
    report_detector("H1", h1_dir)
    report_detector("L1", l1_dir)


if __name__ == "__main__":
    main()
