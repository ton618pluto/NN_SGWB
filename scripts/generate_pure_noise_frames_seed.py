#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random

import bilby
from bilby.gw.detector import PowerSpectralDensity
import gwpy
import gwpy.timeseries as ts
import numpy as np


BASE_SEED = 123456


def generate_noiseframes_for_job(job_number: int, base_seed: int = BASE_SEED) -> None:
    duration = 2048
    start_time = (job_number - 1) * duration
    minimum_frequency = 5.0
    sampling_frequency = 2048.0

    # 固定到 job 级别：同一个 job_number 多次运行结果一致，不同 job_number 仍然不同。
    job_seed = base_seed + job_number
    random.seed(job_seed)
    np.random.seed(job_seed)

    psd = PowerSpectralDensity(asd_file="./O4_mock.txt")
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
        inj_channel = gwpy.detector.Channel(channel_name)
        injected_ts = ts.TimeSeries(
            ifo.time_domain_strain,
            times=ifo.time_array,
            name=channel_name,
            channel=inj_channel,
            dtype=float,
        )

        output_dir = f"./noise_waveform_{ifo.name}"
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"{ifo.name}-STRAIN-{job_number:05d}-duration.gwf"
        full_path = os.path.join(output_dir, file_name)
        injected_ts.write(full_path)


if __name__ == "__main__":
    start_job = 1
    end_job = 23

    total_jobs = end_job - start_job + 1
    for index, job_number in enumerate(range(start_job, end_job + 1), start=1):
        generate_noiseframes_for_job(job_number)
        print(f"进度: {index}/{total_jobs} (job {job_number:05d})")

    print(f"\n全部完成: 已生成 job {start_job} 到 job {end_job} 的噪声文件。")
