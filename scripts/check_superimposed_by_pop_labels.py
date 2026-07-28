from pathlib import Path
import re

import numpy as np


version = "v61_random_two_noise"  # this will be changed
detector = "H1"
sample_a = 0
sample_b = 1
split_index = 0  # is no changed

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR / "processed_data_superimposed" / version / f"{detector}_splits"
POP_PATTERN = re.compile(r"(pop\d{5})")


def find_first_pop_with_samples(data_dir: Path, sample_indices: tuple[int, int], split_idx: int):
    grouped = {}
    for path in sorted(data_dir.glob("*.npy")):
        match = POP_PATTERN.search(path.name)
        if match is None:
            continue
        pop_id = match.group(1)
        grouped.setdefault(pop_id, []).append(path)

    for pop_id, paths in grouped.items():
        names = {path.name for path in paths}
        needed = {
            f"{detector}-SUPERIMPOSED-{pop_id}_sample{sample_indices[0]:05d}_p{split_idx}.npy",
            f"{detector}-SUPERIMPOSED-{pop_id}_sample{sample_indices[1]:05d}_p{split_idx}.npy",
        }
        if needed.issubset(names):
            return pop_id
    return None


def load_label(path: Path):
    sample = np.load(path, allow_pickle=True).item()
    return np.asarray(sample["label"])


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    pop_id = find_first_pop_with_samples(DATA_DIR, (sample_a, sample_b), split_index)
    if pop_id is None:
        raise RuntimeError(
            f"No pop found with both sample{sample_a:05d} and sample{sample_b:05d} at split p{split_index}."
        )

    path_a = DATA_DIR / f"{detector}-SUPERIMPOSED-{pop_id}_sample{sample_a:05d}_p{split_index}.npy"
    path_b = DATA_DIR / f"{detector}-SUPERIMPOSED-{pop_id}_sample{sample_b:05d}_p{split_index}.npy"

    label_a = load_label(path_a)
    label_b = load_label(path_b)
    same = np.array_equal(label_a, label_b)

    print(f"Data dir: {DATA_DIR}")
    print(f"Detector: {detector}")
    print(f"Pop id: {pop_id}")
    print(f"Sample A path: {path_a}")
    print(f"Sample B path: {path_b}")
    print(f"Label A: {label_a}")
    print(f"Label B: {label_b}")
    print(f"Labels identical: {same}")


if __name__ == "__main__":
    main()
