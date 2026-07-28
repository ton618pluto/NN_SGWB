from pathlib import Path

import numpy as np


TRAIN_VERSION = "v100"
TEST_VERSION = "fixed_one_test_v100"
TRAIN_FILE_INDEX = 3
TEST_FILE_INDEX = 0

TRAIN_DIR = Path("./parameter_sampling_train")
TEST_DIR = Path("./parameter_sampling_test")

HEAD_COUNT = 10
COMPARE_KEYS = [
    "num_events",
    "attempt_index",
    "attempts_used",
    "true_R0",
    "true_alpha_z",
    "true_beta_z",
    "true_zp",
    "true_alpha_m",
    "true_m_max",
    "true_delta_m",
    "true_m_min",
    "true_lambda_peak",
    "true_mu_m",
    "true_sigma_m",
    "true_beta_q",
    "total_duration_seconds",
    "frame_duration_seconds",
    "z_max",
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
]


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=False)


def summarize_value(name: str, value, head_count: int = HEAD_COUNT) -> None:
    arr = np.asarray(value)
    if arr.ndim == 0:
        print(f"  {name}: scalar={arr.item()}")
        return

    head = arr[: min(head_count, arr.shape[0])]
    print(
        f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"head={np.array2string(head, precision=8, separator=', ')}"
    )


def compare_values(name: str, train_value, test_value) -> None:
    train_arr = np.asarray(train_value)
    test_arr = np.asarray(test_value)

    if train_arr.ndim == 0 and test_arr.ndim == 0:
        same = np.isclose(train_arr.item(), test_arr.item())
        print(
            f"  {name}: "
            f"train={train_arr.item()} | test={test_arr.item()} | same={bool(same)}"
        )
        return

    same_shape = train_arr.shape == test_arr.shape
    print(f"  {name}: same_shape={same_shape}, train_shape={train_arr.shape}, test_shape={test_arr.shape}")
    if not same_shape:
        return

    if np.issubdtype(train_arr.dtype, np.number) and np.issubdtype(test_arr.dtype, np.number):
        allclose = np.allclose(train_arr, test_arr, equal_nan=True)
        max_abs_diff = float(np.max(np.abs(train_arr - test_arr))) if train_arr.size > 0 else 0.0
        print(f"    allclose={allclose}, max_abs_diff={max_abs_diff}")
    else:
        equal = np.array_equal(train_arr, test_arr)
        print(f"    equal={equal}")

    train_head = train_arr[: min(HEAD_COUNT, train_arr.shape[0])]
    test_head = test_arr[: min(HEAD_COUNT, test_arr.shape[0])]
    print(f"    train_head={np.array2string(train_head, precision=8, separator=', ')}")
    print(f"    test_head={np.array2string(test_head, precision=8, separator=', ')}")


def main() -> None:
    train_path = TRAIN_DIR / TRAIN_VERSION / f"CBC_params_example{TRAIN_FILE_INDEX:05d}.npz"
    test_path = TEST_DIR / TEST_VERSION / f"CBC_params_example{TEST_FILE_INDEX:05d}.npz"

    train_npz = load_npz(train_path)
    test_npz = load_npz(test_path)

    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    print()

    print("Train file summary:")
    for key in COMPARE_KEYS:
        if key in train_npz.files:
            summarize_value(key, train_npz[key])
    print()

    print("Test file summary:")
    for key in COMPARE_KEYS:
        if key in test_npz.files:
            summarize_value(key, test_npz[key])
    print()

    print("Comparison:")
    all_keys = sorted(set(train_npz.files) | set(test_npz.files))
    for key in all_keys:
        if key not in train_npz.files:
            print(f"  {key}: missing in train, present in test")
            continue
        if key not in test_npz.files:
            print(f"  {key}: present in train, missing in test")
            continue
        compare_values(key, train_npz[key], test_npz[key])


if __name__ == "__main__":
    main()
