from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent / "training_set5"


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sum(1 for item in directory.iterdir() if item.is_file())


def main() -> None:
    base_dir = BASE_DIR.resolve()
    h1_dir = base_dir / "H1"
    l1_dir = base_dir / "L1"

    h1_count = count_files(h1_dir)
    l1_count = count_files(l1_dir)

    print(f"Base directory: {base_dir}")
    print(f"H1: {h1_count}")
    print(f"L1: {l1_count}")


if __name__ == "__main__":
    main()
