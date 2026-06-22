from collections import Counter
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
# folder = "processed_data_superimposed"
folder = "processed_data_train"
version = "v63"
TARGET_DIRS = [
    SCRIPTS_DIR / folder / version / "H1_splits",
    SCRIPTS_DIR / folder / version / "L1_splits",
]


def count_npy_files(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    return sum(1 for path in target_dir.iterdir() if path.suffix == ".npy" and path.is_file())


def load_sequence_length(path: Path) -> int | None:
    try:
        sample = np.load(path, allow_pickle=True).item()
        data = sample["data"]
        return int(len(data))
    except Exception as exc:
        print(f"Failed to read {path.name}: {exc}")
        return None


def collect_lengths(target_dir: Path) -> list[tuple[Path, int]]:
    if not target_dir.exists():
        print(f"Directory does not exist: {target_dir}")
        return []

    items: list[tuple[Path, int]] = []
    for path in sorted(target_dir.iterdir()):
        if path.suffix != ".npy" or not path.is_file():
            continue
        seq_len = load_sequence_length(path)
        if seq_len is None:
            continue
        items.append((path, seq_len))
    return items


def main() -> None:
    total_checked = 0
    total_anomalies = 0

    for target_dir in TARGET_DIRS:
        print(f"\nChecking directory: {target_dir}")
        print(f"  .npy file count: {count_npy_files(target_dir)}")

        items = collect_lengths(target_dir)
        total_checked += len(items)

        if not items:
            print("  No readable .npy files found.")
            continue

        length_counter = Counter(seq_len for _, seq_len in items)
        expected_length, expected_count = length_counter.most_common(1)[0]

        print(f"  Most common sequence length: {expected_length} ({expected_count} files)")
        print("  Sequence length distribution:")
        for seq_len, count in sorted(length_counter.items()):
            print(f"    {seq_len}: {count}")

        anomalous = [(path, seq_len) for path, seq_len in items if seq_len != expected_length]
        if not anomalous:
            print("  No files with different sequence length found.")
            continue

        print(f"  Found {len(anomalous)} files with different sequence length:")
        for path, seq_len in anomalous:
            print(f"    {path.name} | length={seq_len}")
        total_anomalies += len(anomalous)

    print(f"\nTotal readable .npy files checked: {total_checked}")
    print(f"Total files with different sequence length: {total_anomalies}")


if __name__ == "__main__":
    main()
