#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import bilby
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.detector import utils
import gwpy
import gwpy.timeseries as ts
from gwpy.timeseries import TimeSeries
import warnings
import sys

def generate_Noiseframes_for_job(jobNumber):
    duration = 2048

    st = (jobNumber - 1) * duration
    et = st + duration
    minimum_frequency = 5.
    reference_frequency = 25.
    sampling_frequency = 2048.0
    psd = PowerSpectralDensity(asd_file="./O4_mock.txt")
    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo in interferometers:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.power_spectral_density = psd

    # Realization of the detector's noise
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=st)

    base_channel_name = 'TEST_INJ'
    gps_start_time = ifo.time_array[0]
    for ifo in interferometers:
        new_channel_names = ifo.name + ':' + base_channel_name
        inj_channel = gwpy.detector.Channel(ifo.name + ':' + new_channel_names)
        injected_ts = ts.TimeSeries(ifo.time_domain_strain, times=ifo.time_array,
                                    name=new_channel_names, channel=inj_channel, dtype=float)
        # file_name = ifo.name[0]+'-'+ifo.name+'_'+base_channel_name+'_'+str(st)+'_'+str(et)+'.gwf'

        file_name = f"{ifo.name}-STRAIN-{jobNumber:05d}-duration.gwf"


        output_dir = './noise_waveform_' + ifo.name
        if not os.path.exists(output_dir):
            # 如果目录不存在，尝试创建它
            os.makedirs(output_dir)
            print(f"⚠️ 警告: 创建了输出目录: {output_dir}")
        full_path = os.path.join(output_dir, file_name)
        injected_ts.write(full_path)


if __name__ == '__main__':
    # 定义循环范围
    start_job = 1
    end_job = 24

    for job_number in range(start_job, end_job):
        generate_Noiseframes_for_job(job_number)
        if job_number%1==0:
            print(f"🚀 进度: {job_number/end_job*100:.0f}% ({job_number} 个事件已处理)")


    print(f"\n\n=============== ✅ 所有任务 (Job 1 到 Job {end_job}) 已完成 ===============")
