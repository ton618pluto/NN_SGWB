from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries


TRAINING_VERSION = "v80"
PROCESSED_VERSION = "v80"
FOLDER_NAME = "training_set0"
POP_INDEX = 79286
SAMPLE_INDEX = 0
SPLIT_INDEX = 0


def build_base_name(pop_index: int, sample_index: int) -> str:
    return f"pop{pop_index:05d}_sample{sample_index:05d}"


def read_gwf_length(file_path: Path, detector: str) -> int:
    ts = TimeSeries.read(file_path, channel=f"{detector}:TEST_INJ")
    return len(ts.value)


def read_npy_length(file_path: Path) -> int:
    sample = np.load(file_path, allow_pickle=True).item()
    return len(sample["data"])


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    base_name = build_base_name(POP_INDEX, SAMPLE_INDEX)

    training_root = scripts_dir / "training_set" / TRAINING_VERSION / FOLDER_NAME
    processed_root = scripts_dir / "processed_data_train" / PROCESSED_VERSION

    print(f"Training version: {TRAINING_VERSION}")
    print(f"Processed version: {PROCESSED_VERSION}")
    print(f"Base sample id: {base_name}")
    print(f"Split index: {SPLIT_INDEX}")
    print()

    for detector in ("H1", "L1"):
        gwf_path = training_root / detector / f"{detector}-{base_name}.gwf"
        npy_path = processed_root / f"{detector}_splits" / f"{detector}-{base_name}_p{SPLIT_INDEX}.npy"

        print(f"{detector}:")
        print(f"  GWF path: {gwf_path}")
        if gwf_path.exists():
            gwf_length = read_gwf_length(gwf_path, detector)
            print(f"  GWF sequence length: {gwf_length}")
        else:
            print("  GWF file not found")

        print(f"  NPY path: {npy_path}")
        if npy_path.exists():
            npy_length = read_npy_length(npy_path)
            print(f"  NPY sequence length: {npy_length}")
        else:
            print("  NPY file not found")
        print()


if __name__ == "__main__":
    main()
