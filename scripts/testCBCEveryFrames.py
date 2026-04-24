from pathlib import Path

import numpy as np


duration = 2048
jobNumber = 4
minimum_frequency = 10
param_dir = Path("./parameter_sampling_train/v4")


def estimation_t0(tc, m1, m2):
    MTSUN_SI = 4.925491025543576e-06
    f0 = minimum_frequency - 1
    mc = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
    c_factor = 5 ** (3.0 / 8) / (8 * np.pi)
    real_t0 = tc - (f0 / c_factor) ** (-8.0 / 3) * (mc * MTSUN_SI) ** (-5.0 / 3)
    return real_t0


def count_events_for_job(npz_path: Path, job_index: int) -> int:
    st = (job_index - 1) * duration
    et = st + duration

    list_cbc = np.load(npz_path)
    tc = list_cbc["tc"]
    t0 = estimation_t0(tc, list_cbc["m1"], list_cbc["m2"])
    list_cbc_mask = (tc > st) & (t0 < et)
    return int(np.count_nonzero(list_cbc_mask))


def main() -> None:
    npz_files = sorted(param_dir.glob("CBC_params_example*.npz"))[:10]
    if not npz_files:
        raise FileNotFoundError(f"No CBC parameter files found in: {param_dir.resolve()}")

    print(f"Parameter directory: {param_dir.resolve()}")
    print(f"Checking first {len(npz_files)} files, jobs 1 to {jobNumber}")

    for npz_path in npz_files:
        print(f"\n{npz_path.name}")
        for job_index in range(1, jobNumber + 1):
            num_events = count_events_for_job(npz_path, job_index)
            print(f"  job {job_index}: {num_events} events")


if __name__ == "__main__":
    main()
