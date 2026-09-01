#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path

import bilby
import gwpy
import gwpy.timeseries as ts
import numpy as np
from bilby.gw.detector import PowerSpectralDensity


BASE_SEED = 42
version = "v100_shifted_asd"
start_job = 1
end_job = 3  # this will be changed

NOISE_SCALE_FACTOR = 1
sampling_frequency = 512.0
duration = 2048
minimum_frequency = 5.0

ASD_SOURCE_PATH = Path(__file__).resolve().parent / "O4_mock.txt"
FREQUENCY_DIVIDE_SCALE = 2.0

SHIFTED_ASD_PATH = (
    Path(__file__).resolve().parent
    / f"O4_mock_freq_divide_by_scale_{FREQUENCY_DIVIDE_SCALE:g}.txt"
)


def build_shifted_asd_file(
    source_path: Path = ASD_SOURCE_PATH,
    output_path: Path = SHIFTED_ASD_PATH,
    divide_scale: float = FREQUENCY_DIVIDE_SCALE,
) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"ASD source file not found: {source_path}")
    if divide_scale == 0:
        raise ValueError("FREQUENCY_DIVIDE_SCALE cannot be 0")

    table = np.loadtxt(source_path, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {source_path}")

    shifted = table.copy()
    shifted[:, 0] = shifted[:, 0] / divide_scale
    np.savetxt(output_path, shifted, fmt="%.16e")
    print(f"ASD shift info: divide_scale={divide_scale:g}")
    return output_path


def generate_noiseframes_for_job(
    job_number: int,
    asd_path: Path,
    base_seed: int = BASE_SEED,
) -> None:
    start_time = 0
    h1_output_dir = f"./noise_waveform/{version}/noise_waveform_H1"
    l1_output_dir = f"./noise_waveform/{version}/noise_waveform_L1"
    os.makedirs(h1_output_dir, exist_ok=True)
    os.makedirs(l1_output_dir, exist_ok=True)

    h1_path = os.path.join(h1_output_dir, f"H1-STRAIN-{job_number:05d}-duration.gwf")
    l1_path = os.path.join(l1_output_dir, f"L1-STRAIN-{job_number:05d}-duration.gwf")
    if os.path.exists(h1_path) and os.path.exists(l1_path):
        print(f"job {job_number:05d} skipped, existing files found.")
        return

    job_seed = base_seed + job_number
    random.seed(job_seed)
    np.random.seed(job_seed)

    psd = PowerSpectralDensity(asd_file=str(asd_path))
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in interferometers:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.power_spectral_density = psd

    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )

    base_channel_name = "TEST_INJ"
    for ifo in interferometers:
        channel_name = f"{ifo.name}:{base_channel_name}"
        scaled_strain = ifo.time_domain_strain * NOISE_SCALE_FACTOR
        inj_channel = gwpy.detector.Channel(channel_name)
        injected_ts = ts.TimeSeries(
            scaled_strain,
            times=ifo.time_array,
            name=channel_name,
            channel=inj_channel,
            dtype=float,
        )

        output_dir = f"./noise_waveform/{version}/noise_waveform_{ifo.name}"
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"{ifo.name}-STRAIN-{job_number:05d}-duration.gwf"
        full_path = os.path.join(output_dir, file_name)
        injected_ts.write(full_path)


if __name__ == "__main__":
    shifted_asd_path = build_shifted_asd_file()
    print(f"Using shifted ASD file: {shifted_asd_path}")

    total_jobs = end_job - start_job + 1
    for index, job_number in enumerate(range(start_job, end_job + 1), start=1):
        generate_noiseframes_for_job(job_number, shifted_asd_path)
        print(f"进度: {index}/{total_jobs} (job {job_number:05d})")

    print(f"\n全部完成: 已生成 job {start_job} 到 job {end_job} 的噪声文件。")
