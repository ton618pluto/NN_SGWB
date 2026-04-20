from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    REPO_ROOT / "scripts" / "processed_data_train" / "v3" / "training_set0" / "H1_splits",
    REPO_ROOT / "scripts" / "processed_data_train" / "v3" / "training_set0" / "L1_splits",
]


def check_npy_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"

    size = path.stat().st_size
    if size == 0:
        return False, "empty_file"

    try:
        sample = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        return False, f"load_error: {type(exc).__name__}: {exc}"

    if not isinstance(sample, dict):
        return False, f"invalid_type: expected dict, got {type(sample).__name__}"

    missing_keys = [key for key in ("data", "label") if key not in sample]
    if missing_keys:
        return False, f"missing_keys: {','.join(missing_keys)}"

    try:
        np.asarray(sample["data"], dtype=np.float32)
        np.asarray(sample["label"], dtype=np.float32)
    except Exception as exc:
        return False, f"invalid_array: {type(exc).__name__}: {exc}"

    return True, "ok"


def scan_directory(directory: Path) -> tuple[int, list[tuple[Path, str]]]:
    if not directory.exists():
        return 0, [(directory, "missing_directory")]

    bad_files: list[tuple[Path, str]] = []
    npy_files = sorted(directory.rglob("*.npy"))

    progress = tqdm(npy_files, desc=f"Scanning {directory.name}", unit="file")
    for path in progress:
        ok, reason = check_npy_file(path)
        if not ok:
            bad_files.append((path, reason))
            progress.set_postfix(bad=len(bad_files))

    return len(npy_files), bad_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Check .npy files for corruption or invalid format.")
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=DEFAULT_DIRS,
        help="Directories to scan. Defaults to training_set0 H1_splits and L1_splits.",
    )
    args = parser.parse_args()

    total_files = 0
    total_bad = 0

    for directory in args.directories:
        resolved_dir = directory if directory.is_absolute() else (REPO_ROOT / directory).resolve()
        file_count, bad_files = scan_directory(resolved_dir)
        total_files += file_count
        total_bad += len(bad_files)

        print(f"\nDirectory: {resolved_dir}")
        print(f"Total .npy files: {file_count}")
        print(f"Bad files: {len(bad_files)}")

        for path, reason in bad_files:
            print(f"{path}\t{reason}")

    print("\nSummary")
    print(f"Scanned files: {total_files}")
    print(f"Bad files: {total_bad}")


if __name__ == "__main__":
    main()
