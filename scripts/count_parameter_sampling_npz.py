from pathlib import Path


VERSION = "v64"
DELETE_EMPTY_FILES = True


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "parameter_sampling_train" / VERSION
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    npz_files = sorted(base_dir.glob("*.npz"))
    empty_files = [path for path in npz_files if path.stat().st_size == 0]
    deleted_files = []

    if DELETE_EMPTY_FILES:
        for path in empty_files:
            try:
                path.unlink()
                deleted_files.append(path)
            except OSError as exc:
                print(f"Failed to delete empty file: {path} | error: {exc}")

    print(f"Directory: {base_dir}")
    print(f"Version: {VERSION}")
    print(f"Total .npz files: {len(npz_files)}")
    print(f"Empty .npz files: {len(empty_files)}")
    if DELETE_EMPTY_FILES:
        print(f"Deleted empty .npz files: {len(deleted_files)}")
        for path in deleted_files:
            print(f"Deleted: {path.name}")


if __name__ == "__main__":
    main()
