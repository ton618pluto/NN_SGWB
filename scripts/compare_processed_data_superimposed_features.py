from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent / "processed_data_superimposed"
VERSION_A = "v100_random_two_noise_100"
VERSION_B = "v100_random_two_noise_100_float64"

POP_INDEX = 0
SAMPLE_INDEX = 0
SPLIT_INDEX = 0
DETECTORS = ["H1", "L1"]

ATOL = 1e-12
RTOL = 1e-9
PRINT_HEAD = 10


def build_file_path(root_dir: Path, version: str, detector: str) -> Path:
    return (
        root_dir
        / version
        / f"{detector}_splits"
        / f"{detector}-SUPERIMPOSED-pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}_p{SPLIT_INDEX}.npy"
    )


def load_sample(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    sample = np.load(path, allow_pickle=True).item()
    if not isinstance(sample, dict) or "data" not in sample:
        raise ValueError(f"Unexpected sample format in {path}")
    return sample


def compare_feature_arrays(data_a: np.ndarray, data_b: np.ndarray) -> None:
    same_shape = data_a.shape == data_b.shape
    print(f"  shape A: {data_a.shape}")
    print(f"  shape B: {data_b.shape}")
    print(f"  same shape: {'Y' if same_shape else 'N'}")
    print(f"  dtype A: {data_a.dtype}")
    print(f"  dtype B: {data_b.dtype}")
    if not same_shape:
        return

    a64 = np.asarray(data_a, dtype=np.float64)
    b64 = np.asarray(data_b, dtype=np.float64)
    diff = a64 - b64
    abs_diff = np.abs(diff)

    print(f"  exact same numeric values: {'Y' if np.array_equal(a64, b64) else 'N'}")
    print(f"  allclose (rtol={RTOL}, atol={ATOL}): {'Y' if np.allclose(a64, b64, rtol=RTOL, atol=ATOL) else 'N'}")
    print(f"  same after B -> float32 cast: {'Y' if np.array_equal(data_a.astype(np.float32), data_b.astype(np.float32)) else 'N'}")
    print(f"  max abs diff: {float(abs_diff.max()):.18e}")
    print(f"  mean abs diff: {float(abs_diff.mean()):.18e}")

    if np.any(abs_diff > 0):
        idx = int(np.argmax(abs_diff))
        print(f"  first {PRINT_HEAD} values A: {data_a[:PRINT_HEAD]}")
        print(f"  first {PRINT_HEAD} values B: {data_b[:PRINT_HEAD]}")
        print(f"  max diff index: {idx}")
        print(f"  value A @ max diff: {a64[idx]:.18e}")
        print(f"  value B @ max diff: {b64[idx]:.18e}")


def main() -> None:
    print(f"Root dir: {ROOT_DIR}")
    print(f"Version A: {VERSION_A}")
    print(f"Version B: {VERSION_B}")
    print(f"Target sample: pop{POP_INDEX:05d}_sample{SAMPLE_INDEX:05d}_p{SPLIT_INDEX}")
    print()

    for detector in DETECTORS:
        path_a = build_file_path(ROOT_DIR, VERSION_A, detector)
        path_b = build_file_path(ROOT_DIR, VERSION_B, detector)

        print(f"[{detector}]")
        print(f"  file A: {path_a}")
        print(f"  file B: {path_b}")

        sample_a = load_sample(path_a)
        sample_b = load_sample(path_b)

        data_a = np.asarray(sample_a["data"])
        data_b = np.asarray(sample_b["data"])
        compare_feature_arrays(data_a, data_b)
        print()


if __name__ == "__main__":
    main()
