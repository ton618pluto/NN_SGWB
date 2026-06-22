from __future__ import annotations

from pathlib import Path

import numpy as np


TRAIN_VERSION = "v80"
TEST_VERSION = "fixed_one_test_v80"
TRAIN_FILE_INDEX = 3
TEST_FILE_INDEX = 0
ABS_TOL = 0.0
TEST_FIRST_FRAME_ONLY = True
FRAME_DURATION_SECONDS = 2048.0


TRUE_PARAMETER_NAMES = (
    "R0",
    "alpha_z",
    "beta_z",
    "zp",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)

EVENT_ARRAY_NAMES = (
    "tc",
    "m1",
    "m2",
    "dL",
    "z",
    "ra",
    "dec",
    "psi",
    "phi_c",
    "iota",
)

EVENT_STAT_NAMES = (
    "num_events",
    "num_valid_events",
    "num_errored_events",
    "num_strong_events",
)


def build_train_file_path(version: str, file_index: int) -> Path:
    return (
        Path(__file__).resolve().parent
        / "parameter_sampling_train"
        / version
        / f"CBC_params_example{file_index:05d}.npz"
    )


def build_test_file_path(version: str, file_index: int) -> Path:
    return (
        Path(__file__).resolve().parent
        / "parameter_sampling_test"
        / version
        / f"CBC_params_example{file_index:05d}.npz"
    )


def load_true_parameters(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path)
    result = {}
    for name in TRUE_PARAMETER_NAMES:
        key = f"true_{name}"
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}")
        result[name] = float(data[key])
    return result


def load_event_arrays_and_stats(path: Path):
    data = np.load(path)

    arrays = {}
    for name in EVENT_ARRAY_NAMES:
        if name not in data:
            raise KeyError(f"Missing key '{name}' in {path}")
        arrays[name] = np.asarray(data[name])

    stats = {}
    for name in EVENT_STAT_NAMES:
        if name not in data:
            raise KeyError(f"Missing key '{name}' in {path}")
        stats[name] = int(data[name])

    return arrays, stats


def select_first_frame(arrays: dict[str, np.ndarray], frame_duration_seconds: float) -> dict[str, np.ndarray]:
    tc = arrays["tc"].astype(np.float64)
    mask = tc < frame_duration_seconds
    filtered = {}
    for name, arr in arrays.items():
        filtered[name] = np.asarray(arr)[mask]
    return filtered


def main() -> None:
    train_path = build_train_file_path(TRAIN_VERSION, TRAIN_FILE_INDEX)
    test_path = build_test_file_path(TEST_VERSION, TEST_FILE_INDEX)

    train_params = load_true_parameters(train_path)
    test_params = load_true_parameters(test_path)
    train_arrays, train_stats = load_event_arrays_and_stats(train_path)
    test_arrays, test_stats = load_event_arrays_and_stats(test_path)
    if TEST_FIRST_FRAME_ONLY:
        test_arrays = select_first_frame(test_arrays, FRAME_DURATION_SECONDS)
        test_stats = {
            "num_events": int(test_arrays["tc"].size),
            "num_valid_events": int(test_arrays["tc"].size),
            "num_errored_events": 0,
            "num_strong_events": 0,
        }

    print(f"Train file: {train_path}")
    print(f"Test file:  {test_path}")
    if TEST_FIRST_FRAME_ONLY:
        print(f"Test comparison scope: first frame only (tc < {FRAME_DURATION_SECONDS}s)")
    print()
    print(f"{'param':<14} {'train':>18} {'test':>18} {'same':>8} {'abs_diff':>18}")
    print("-" * 82)

    all_same = True
    for name in TRUE_PARAMETER_NAMES:
        train_value = train_params[name]
        test_value = test_params[name]
        abs_diff = abs(train_value - test_value)
        same = abs_diff <= ABS_TOL
        all_same &= same
        same_text = "Y" if same else "N"
        print(
            f"{name:<14} "
            f"{train_value:>18.10f} "
            f"{test_value:>18.10f} "
            f"{same_text:>8} "
            f"{abs_diff:>18.10f}"
        )

    print()
    print(f"All true_* hyperparameters identical: {'YES' if all_same else 'NO'}")
    print()

    print("Event statistics:")
    print(f"{'stat':<22} {'train':>12} {'test':>12} {'same':>8}")
    print("-" * 58)
    all_stats_same = True
    for name in EVENT_STAT_NAMES:
        train_value = train_stats[name]
        test_value = test_stats[name]
        same = train_value == test_value
        all_stats_same &= same
        same_text = "Y" if same else "N"
        print(f"{name:<22} {train_value:>12d} {test_value:>12d} {same_text:>8}")

    print()
    print("Event arrays:")
    print(
        f"{'array':<12} {'train_len':>10} {'test_len':>10} "
        f"{'same_len':>10} {'allclose':>10} {'max_abs_diff':>18}"
    )
    print("-" * 76)
    all_arrays_same = True
    for name in EVENT_ARRAY_NAMES:
        train_arr = train_arrays[name]
        test_arr = test_arrays[name]
        same_len = train_arr.shape == test_arr.shape
        if same_len:
            allclose = np.allclose(train_arr, test_arr, atol=1e-8, rtol=0.0)
            if train_arr.size == 0:
                max_abs_diff = 0.0
            else:
                max_abs_diff = float(np.max(np.abs(train_arr.astype(np.float64) - test_arr.astype(np.float64))))
        else:
            allclose = False
            max_abs_diff = float("nan")

        all_arrays_same &= same_len and allclose
        print(
            f"{name:<12} "
            f"{train_arr.size:>10d} "
            f"{test_arr.size:>10d} "
            f"{('Y' if same_len else 'N'):>10} "
            f"{('Y' if allclose else 'N'):>10} "
            f"{max_abs_diff:>18.10f}"
        )

        if same_len and not allclose and train_arr.size > 0:
            diff_idx = int(np.argmax(np.abs(train_arr.astype(np.float64) - test_arr.astype(np.float64))))
            print(
                f"  first notable diff @ idx={diff_idx}: "
                f"train={float(train_arr[diff_idx]):.10f}, "
                f"test={float(test_arr[diff_idx]):.10f}"
            )

    print()
    print(f"All event stats identical: {'YES' if all_stats_same else 'NO'}")
    print(f"All event arrays identical within atol=1e-8: {'YES' if all_arrays_same else 'NO'}")


if __name__ == "__main__":
    main()
