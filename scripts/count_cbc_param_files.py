from pathlib import Path


VERSION = "v40"
BASE_DIR = Path(__file__).resolve().parent / "parameter_sampling_train" / VERSION


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {BASE_DIR}")
    if not BASE_DIR.is_dir():
        raise NotADirectoryError(f"Not a directory: {BASE_DIR}")

    files = sorted(BASE_DIR.glob("CBC_params_example*.npz"))
    print(f"Version: {VERSION}")
    print(f"Directory: {BASE_DIR}")
    print(f"Number of generated CBC parameter files: {len(files)}")


if __name__ == "__main__":
    main()
