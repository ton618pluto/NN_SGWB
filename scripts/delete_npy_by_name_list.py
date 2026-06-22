from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
folder = "processed_data_train"
version = "v43"
TARGET_DIRS = [
    SCRIPTS_DIR / folder / version / "H1_splits",
    SCRIPTS_DIR / folder / version / "L1_splits",
]

# Fill this list with exact file names to delete.
FILES_TO_DELETE = [
    "L1-pop09638_sample00000_p0.npy",
    "L1-pop16992_sample00000_p0.npy ",
    "L1-pop27930_sample00000_p0.npy "
]


def main() -> None:
    print(f"Folder: {folder}")
    print(f"Version: {version}")
    print(f"Files requested for deletion: {len(FILES_TO_DELETE)}")

    if not FILES_TO_DELETE:
        print("No file names provided in FILES_TO_DELETE.")
        return

    total_deleted = 0
    total_missing = 0

    for target_dir in TARGET_DIRS:
        print(f"\nChecking directory: {target_dir}")
        if not target_dir.exists():
            print("  Directory does not exist.")
            continue

        deleted_here = 0
        missing_here = 0

        for file_name in FILES_TO_DELETE:
            target_path = target_dir / file_name
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                print(f"  Deleted: {target_path.name}")
                deleted_here += 1
            else:
                missing_here += 1

        print(f"  Deleted in this directory: {deleted_here}")
        print(f"  Not found in this directory: {missing_here}")
        total_deleted += deleted_here
        total_missing += missing_here

    print(f"\nTotal deleted: {total_deleted}")
    print(f"Total not found: {total_missing}")


if __name__ == "__main__":
    main()
