#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import gwpy
import bilby
import sys
import os
import gwpy.timeseries
import gwpy
from gwpy.timeseries import TimeSeries
from pygwb.detector import Interferometer
from pygwb.network import Network
from pygwb.baseline import Baseline
from pygwb.simulator import Simulator
from bilby.gw.detector import PowerSpectralDensity

def generate_noise_sgwb(jobNumber):
    def PL_GW(freqs, omegaref, alpha, fref):
        from pygwb.constants import H0
        H_theor = (3 * H0.si.value ** 2) / (10 * np.pi ** 2)

        power = np.zeros_like(freqs)

        power = H_theor * omegaref * freqs ** (alpha - 3) / fref ** (alpha)

        power[0] = power[1]

        return gwpy.frequencyseries.FrequencySeries(power, frequencies=freqs)

    # jobNumber = int(sys.argv[1])
    st = (jobNumber - 1) * 2048
    duration = 4096
    minimum_frequency = 5
    sampling_frequency = 2048

    # 0.定义要注入的SGWB信号的强度
    omegaref = 5e-12  # 引力波能量密度
    fref = 25  # 频率
    alpha = 2/3  # 谱指数

    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo in interferometers:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration

    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=st
    )
    base_channel_name = 'TEST_INJ'
    gps_start_time = ifo.time_array[0]

    net_HL = Network('HL', interferometers)

    frequencies_x = np.linspace(5, 2048, 65537)  # delta_f = 1/32 Hz

    # 1. 计算SGWB PSD
    Intensity_GW_inject = PL_GW(frequencies_x, omegaref, alpha, fref)

    # 2. 注入SGWB信号到探测器中
    net_HL.set_interferometer_data_from_simulator(start_time=st, N_segments=1, GWB_intensity=Intensity_GW_inject, sampling_frequency=sampling_frequency, inject_into_data_flag=True)

    # 3. 提取出对应部分的探测器的时间序列数据
    # 2048*2048是表示数据的长度，因为duration为2048s，且每秒采样2048个点（sampling_frequency为2048）
    H_data = net_HL.interferometers[0].strain_data.to_gwpy_timeseries()[1024 * 2048: 1024 * 2048 + 2048 * 2048]
    L_data = net_HL.interferometers[1].strain_data.to_gwpy_timeseries()[1024 * 2048: 1024 * 2048 + 2048 * 2048]

    H_data.times = np.arange(st, st + 2048 + 1 / 2048, 1 / 2048)
    L_data.times = np.arange(st, st + 2048 + 1 / 2048, 1 / 2048)

    H_data.name = "H1:TEST_INJ"
    L_data.name = "L1:TEST_INJ"

    # 4. 从其他文件中读取噪声
    # H_Noise = TimeSeries.read(f'.../Pure_Noise/H-STRAIN-{st}-2048.gwf', "H1:TEST_INJ", st, st+2048)
    H_Noise = TimeSeries.read(f'./noise_waveform_H1/H1-STRAIN-{jobNumber:05d}-duration.gwf', "H1:TEST_INJ", st, st + 2048)
    # L_Noise = TimeSeries.read(f'.../Pure_Noise/L-STRAIN-{st}-2048.gwf', "L1:TEST_INJ", st, st+2048)
    L_Noise = TimeSeries.read(f'./noise_waveform_L1/L1-STRAIN-{jobNumber:05d}-duration.gwf', "L1:TEST_INJ", st, st + 2048)

    # 5. SGWB+仪器噪声
    H_data = H_data + H_Noise
    L_data = L_data + L_Noise

    # output_dir = './noise_waveform_' + ifo.name
    outdirs=['./Noisy_SGWB_H1','./Noisy_SGWB_L1']
    for dir in outdirs:
        if not os.path.exists(dir):
            # 如果目录不存在，尝试创建它
            os.makedirs(dir)
            print(f"⚠️ 警告: 创建了输出目录: {dir}")

    # H_data.write(f'.../Noisy_SGWB/alpha_0_Omega_5e-12/H-STRAIN-{st}-2048.gwf')
    H_data.write(f'./Noisy_SGWB_H1/H-STRAIN-SGWB-{jobNumber:05d}-2048.gwf')
    # L_data.write(f'.../Noisy_SGWB/alpha_0_Omega_5e-12/L-STRAIN-{st}-2048.gwf')
    L_data.write(f'./Noisy_SGWB_L1/L-STRAIN-SGWB-{jobNumber:05d}-2048.gwf')


if __name__ == '__main__':
    # 定义循环范围
    start_job = 284
    end_job = 15409

    for job_number in range(start_job, end_job + 1):
        generate_noise_sgwb(job_number)
        if job_number%50==0:
            print(f"🚀 进度: {job_number/end_job*100:.0f}% ({job_number} 个SGWB事件已处理)")


    print(f"\n\n=============== ✅ 所有任务 (Job 1 到 Job {end_job}) 已完成 ===============")