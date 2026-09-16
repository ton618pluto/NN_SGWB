from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

import predict_test_sample_cobaya_mcmc_single_frame as signal_noise_script

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    TimeSeries = None


# This script uses the same trained model and MCMC settings as the original
# single-frame script.  Only the model input is changed to
# (processed signal + noise) - pure signal.
SIGNAL_DATA_VERSION = "fixed_one_test_v100_random_two_noise_100"
SIGNAL_FOLDER_NAME = "training_set0"
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"

# Keep the source processed-data version separate from OUTPUT_TEST_DATA_VERSION.
# The latter is used in result filenames so the signal+noise results are not
# overwritten.
SOURCE_PROCESSED_TEST_VERSION = signal_noise_script.TEST_DATA_VERSION
OUTPUT_TEST_DATA_VERSION = f"{SOURCE_PROCESSED_TEST_VERSION}_noise_only"


def _load_signal_chunk(
    signal_path: Path,
    detector: str,
    split_index: int,
    expected_length: int,
) -> np.ndarray:
    if TimeSeries is None:
        raise ImportError(
            "gwpy is required to read the pure-signal GWF files. "
            "Please install `gwpy` before running this script."
        )
    if not signal_path.exists():
        raise FileNotFoundError(f"Pure-signal GWF file not found: {signal_path}")

    signal = np.asarray(
        TimeSeries.read(signal_path, channel=f"{detector}:{STRAIN_CHANNEL_SUFFIX}").value,
        dtype=np.float64,
    )
    if expected_length <= 0 or len(signal) % expected_length != 0:
        raise ValueError(
            f"Cannot align signal and processed frame: signal length={len(signal)}, "
            f"processed length={expected_length}, file={signal_path}"
        )

    num_splits = len(signal) // expected_length
    if split_index < 0 or split_index >= num_splits:
        raise IndexError(
            f"Split index {split_index} is outside 0..{num_splits - 1} for {signal_path}"
        )
    start = split_index * expected_length
    return signal[start : start + expected_length]


def _load_noise_only_single_frame_data(
    processed_test_version: str,
    test_sample_index: int,
    frame_index: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    parameter_names: list[str],
    label_mean_array: np.ndarray,
    label_std_array: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, dict[str, float], Path, Path]:
    del processed_test_version

    scripts_dir = Path(__file__).resolve().parents[1]
    processed_root = (
        scripts_dir
        / signal_noise_script.input_folder
        / SOURCE_PROCESSED_TEST_VERSION
    )
    h1_dir = processed_root / "H1_splits"
    l1_dir = processed_root / "L1_splits"
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 split directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 split directory not found: {l1_dir}")

    prefix = f"H1-SUPERIMPOSED-pop{test_sample_index:05d}_"
    h1_files = sorted(h1_dir.glob(f"{prefix}*.npy"))
    if not h1_files:
        raise FileNotFoundError(
            f"No H1 files found for pop{test_sample_index:05d} in {h1_dir}"
        )
    if frame_index < 0 or frame_index >= len(h1_files):
        raise IndexError(
            f"FRAME_INDEX out of range: {frame_index}. "
            f"Available frames: 0 to {len(h1_files) - 1}"
        )

    h1_path = h1_files[frame_index]
    l1_path = l1_dir / h1_path.name.replace("H1-", "L1-", 1)
    if not l1_path.exists():
        raise FileNotFoundError(f"Matching L1 file not found: {l1_path}")

    match = re.search(r"_sample(?P<sample>\d+)_p(?P<split>\d+)\.npy$", h1_path.name)
    if match is None:
        raise ValueError(f"Unexpected processed filename: {h1_path.name}")
    sample_index = int(match.group("sample"))
    split_index = int(match.group("split"))

    signal_root = (
        scripts_dir
        / "training_set_test"
        / SIGNAL_DATA_VERSION
        / SIGNAL_FOLDER_NAME
    )
    signal_paths = {
        detector: signal_root
        / detector
        / f"{detector}-pop{test_sample_index:05d}_sample{sample_index:05d}.gwf"
        for detector in ("H1", "L1")
    }

    processed_samples = {
        "H1": np.load(h1_path, allow_pickle=True).item(),
        "L1": np.load(l1_path, allow_pickle=True).item(),
    }
    noise_channels = []
    for detector in ("H1", "L1"):
        # Subtract in float64 so cancellation does not unnecessarily discard the
        # much smaller residual, then cast to the model's float32 input below.
        signal_noise = np.asarray(processed_samples[detector]["data"], dtype=np.float64)
        signal = _load_signal_chunk(
            signal_paths[detector], detector, split_index, len(signal_noise)
        )
        noise = signal_noise - signal
        if signal_noise_script.USE_INPUT_SCALE:
            noise = noise * signal_noise_script.INPUT_SCALE
        if signal_noise_script.USE_PER_SAMPLE_DEMEAN:
            noise = noise - noise.mean(dtype=np.float32)
        noise_channels.append(noise)
        print(f"Pure-signal {detector} path: {signal_paths[detector]}")
        print(
            f"Noise-only {detector}: samples={len(noise)}, "
            f"mean={noise.mean():.6e}, std={noise.std():.6e}, "
            f"min={noise.min():.6e}, max={noise.max():.6e}"
        )

    features = np.stack(noise_channels, axis=0)
    features = (features - feature_mean[:, None]) / feature_std[:, None]

    full_hyperparameters = signal_noise_script._label_to_full_hyperparameters(
        processed_samples["H1"]["label"]
    )
    label_true = np.asarray(
        [full_hyperparameters[name] for name in parameter_names], dtype=np.float64
    )
    label_true_norm = (label_true - label_mean_array) / label_std_array
    data_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
    return data_tensor, label_true_norm, full_hyperparameters, h1_path, l1_path


def main() -> None:
    print("Input mode: noise only = (signal + noise) - pure signal")
    print(f"Signal data version: {SIGNAL_DATA_VERSION}")
    print(f"Processed signal+noise version: {SOURCE_PROCESSED_TEST_VERSION}")
    signal_noise_script.TEST_DATA_VERSION = OUTPUT_TEST_DATA_VERSION
    signal_noise_script._load_single_frame_data = _load_noise_only_single_frame_data
    signal_noise_script.main()


if __name__ == "__main__":
    main()
