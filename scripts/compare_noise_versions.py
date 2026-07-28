from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries


# Configure the two noise versions and the shared noise file number to compare.
VERSION_A = "va"
VERSION_B = "vb"
JOB_NUMBER = 1
DETECTORS = ("H1", "L1")
CHANNEL_SUFFIX = "TEST_INJ"
RTOL = 1e-7
ATOL = 0.0


def noise_path(scripts_dir: Path, version: str, detector: str, job_number: int) -> Path:
    return (
        scripts_dir
        / "noise_waveform"
        / version
        / f"noise_waveform_{detector}"
        / f"{detector}-STRAIN-{job_number:05d}-duration.gwf"
    )


def load_noise(path: Path, detector: str) -> TimeSeries:
    if not path.exists():
        raise FileNotFoundError(f"Noise file not found: {path}")
    return TimeSeries.read(str(path), channel=f"{detector}:{CHANNEL_SUFFIX}")


def compare_detector(scripts_dir: Path, detector: str) -> bool:
    path_a = noise_path(scripts_dir, VERSION_A, detector, JOB_NUMBER)
    path_b = noise_path(scripts_dir, VERSION_B, detector, JOB_NUMBER)
    series_a = load_noise(path_a, detector)
    series_b = load_noise(path_b, detector)
    values_a = np.asarray(series_a.value)
    values_b = np.asarray(series_b.value)

    same_shape = values_a.shape == values_b.shape
    same_rate = series_a.sample_rate == series_b.sample_rate
    exactly_equal = same_shape and np.array_equal(values_a, values_b)
    close_equal = same_shape and np.allclose(values_a, values_b, rtol=RTOL, atol=ATOL)
    max_abs_diff = (
        float(np.max(np.abs(values_a - values_b))) if same_shape and values_a.size else float("nan")
    )

    print(f"[{detector}]")
    print(f"  Version A: {path_a}")
    print(f"  Version B: {path_b}")
    print(f"  A shape/sample rate: {values_a.shape} / {series_a.sample_rate}")
    print(f"  B shape/sample rate: {values_b.shape} / {series_b.sample_rate}")
    print(f"  Same shape: {same_shape}")
    print(f"  Same sample rate: {same_rate}")
    print(f"  Exactly identical: {exactly_equal}")
    print(f"  Allclose (rtol={RTOL}, atol={ATOL}): {close_equal}")
    print(f"  Max absolute difference: {max_abs_diff:.8e}\n")
    return exactly_equal


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    print(f"Noise file number: {JOB_NUMBER:05d}")
    print(f"Version A: {VERSION_A}")
    print(f"Version B: {VERSION_B}\n")

    results = [compare_detector(scripts_dir, detector) for detector in DETECTORS]
    if all(results):
        print("Result: H1 and L1 are exactly identical between the two versions.")
    else:
        print("Result: at least one detector differs between the two versions.")


if __name__ == "__main__":
    main()
