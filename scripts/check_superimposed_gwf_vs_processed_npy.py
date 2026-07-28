from pathlib import Path
import re

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


SUPERIMPOSED_VERSION = "v61_random_two_noise_1000"  # this will be changed
PROCESSED_VERSION = "v61_random_two_noise_1000"  # this will be changed
DETECTOR = "L1"
POP_NUM = 0
SAMPLE_NUM = 0
SPLIT_INDEX = 0
NUM_SPLITS = 1  # this will be changed
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
RTOL = 1e-6
ATOL = 1e-8

SCRIPTS_DIR = Path(__file__).resolve().parent
POP_PATTERN = re.compile(r"(pop\d{5})")


def load_label_by_pop(train_idx_path: Path):
    df = pd.read_csv(train_idx_path)
    labels_by_pop = {}
    for row in df.itertuples(index=False):
        sample_id = str(row[0])
        match = POP_PATTERN.search(sample_id)
        if match is None:
            continue
        pop_id = match.group(1)
        if pop_id not in labels_by_pop:
            labels_by_pop[pop_id] = np.asarray(list(row[2:]), dtype=np.float32)
    return labels_by_pop


def main() -> None:
    gwf_dir = SCRIPTS_DIR / "training_set_superimposed" / SUPERIMPOSED_VERSION / DETECTOR
    processed_dir = SCRIPTS_DIR / "processed_data_superimposed" / PROCESSED_VERSION / f"{DETECTOR}_splits"
    train_idx_path = SCRIPTS_DIR / "training_set_superimposed" / SUPERIMPOSED_VERSION / "train_idx.csv"

    gwf_path = gwf_dir / f"{DETECTOR}-SUPERIMPOSED-pop{POP_NUM:05d}_sample{SAMPLE_NUM:05d}.gwf"
    npy_path = processed_dir / f"{DETECTOR}-SUPERIMPOSED-pop{POP_NUM:05d}_sample{SAMPLE_NUM:05d}_p{SPLIT_INDEX}.npy"

    if not gwf_path.exists():
        raise FileNotFoundError(f"GWF file not found: {gwf_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Processed npy file not found: {npy_path}")
    if not train_idx_path.exists():
        raise FileNotFoundError(f"train_idx.csv not found: {train_idx_path}")

    strain_channel = f"{DETECTOR}:{STRAIN_CHANNEL_SUFFIX}"
    gwf_data = TimeSeries.read(gwf_path, channel=strain_channel).value

    sample = np.load(npy_path, allow_pickle=True).item()
    npy_data = np.asarray(sample["data"], dtype=np.float64)
    npy_label = np.asarray(sample["label"], dtype=np.float32)

    if NUM_SPLITS <= 0:
        raise ValueError("NUM_SPLITS must be positive.")
    split_length = len(gwf_data) // NUM_SPLITS
    start = SPLIT_INDEX * split_length
    end = (SPLIT_INDEX + 1) * split_length
    gwf_chunk = np.asarray(gwf_data[start:end], dtype=np.float64)

    labels_by_pop = load_label_by_pop(train_idx_path)
    pop_id = f"pop{POP_NUM:05d}"
    expected_label = labels_by_pop.get(pop_id)
    if expected_label is None:
        raise KeyError(f"Label not found for {pop_id} in {train_idx_path}")

    same_shape = gwf_chunk.shape == npy_data.shape
    allclose_data = same_shape and np.allclose(gwf_chunk, npy_data, rtol=RTOL, atol=ATOL)
    max_abs_diff = float(np.max(np.abs(gwf_chunk - npy_data))) if same_shape and gwf_chunk.size > 0 else None
    same_label = np.array_equal(expected_label, npy_label)

    print(f"GWF path: {gwf_path}")
    print(f"Processed npy path: {npy_path}")
    print(f"train_idx path: {train_idx_path}")
    print(f"Detector: {DETECTOR}")
    print(f"Pop: {pop_id}")
    print(f"Sample: sample{SAMPLE_NUM:05d}")
    print(f"Split index: p{SPLIT_INDEX}")
    print(f"NUM_SPLITS: {NUM_SPLITS}")
    print()

    print(f"GWF full length: {len(gwf_data)}")
    print(f"GWF chunk length: {len(gwf_chunk)}")
    print(f"NPY data length: {len(npy_data)}")
    print(f"Data same shape: {same_shape}")
    print(f"Data allclose: {allclose_data}")
    print(f"Data max abs diff: {max_abs_diff}")
    print()

    print(f"Expected label from train_idx (by pop): {expected_label}")
    print(f"Label stored in npy: {npy_label}")
    print(f"Labels identical: {same_label}")


if __name__ == "__main__":
    main()
