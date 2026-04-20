# generate_pure_cbc_frames.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import bilby
import logging

bilby.core.utils.setup_logger(log_level=logging.WARNING)
from bilby.gw.detector import PowerSpectralDensity
import gwpy.timeseries as ts
import gwpy
from gwpy.timeseries import TimeSeries
import numpy as np
import sys

"""
The code for generating GW frames using CBC params
"""


# I usually run this script on the cluster...please modify the following two lines accordingly
# jobNumber = int(sys.argv[1])
def generate_frames_for_job(jobNumber):
    paramFile = 'test'
    duration = 2048  # s
    st = (jobNumber - 1) * duration
    et = st + duration

    # Set up the duration, sampling frequency, etc.
    # list_cbc: the file contains all the CBC parameters
    list_cbc = np.load("./CBC_params_example.npz")
    minimum_frequency = 10  # 10 Hz is low enough for LIGO! Using smaller f_min can be problematic!
    reference_frequency = 25.  # 参考频率
    sampling_frequency = 2048  # Can be higher...but I use this resoultion all the time

    def estimation_t0(tc, m1, m2):
        """
        Newtownian apporximation to estimate the time when the frequency of the chirp is equal to f0
        """
        MTSUN_SI = 4.925491025543576e-06
        f0 = minimum_frequency - 1
        mc = (m1 * m2) ** (3. / 5) / (m1 + m2) ** (1. / 5)
        C = 5 ** (3. / 8) / (8 * np.pi)
        real_t0 = tc - (f0 / C) ** (-8. / 3) * (mc * MTSUN_SI) ** (-5. / 3)  #
        return real_t0

    tc = list_cbc['tc']  # 合并时间
    m1 = list_cbc['m1']
    m2 = list_cbc['m2']
    z = list_cbc['z']
    m1, m2 = m1 * (1 + z), m2 * (1 + z)  # detector-frame mass
    t0 = estimation_t0(tc, m1, m2)  # 信号的起始时间
    list_cbc_mask = (tc > st) * (t0 < et)  # 获得在第一个jobnumber内的合并事件，布尔数组

    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])  # 包含H1和L1两个探测器实例
    for ifo in interferometers:  # 为两个探测器定义开始分析信号的最小频率、采样频率和每个数据段的持续时间
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration

    interferometers.set_strain_data_from_zero_noise(  # 初始化引力波探测器的应变数据 (strain data) 为零噪声
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=st)

    # set up the waveform generator
    waveform_arguments = {'waveform_approximant': 'IMRPhenomXAS',
                          'reference_frequency': reference_frequency,
                          'minimum_frequency': minimum_frequency}

    # BNS 我们可以暂时不考虑，暂时只考虑双黑洞
    source_model = bilby.gw.source.lal_binary_black_hole
    pc = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

    # for EVENT in range(list_cbc.shape[0]):
    # --- 修正后的参数提取（替换原脚本中从第 81 行开始的 for 循环内部）---

    # 获取所有符合筛选条件的参数数组
    tc_all = list_cbc['tc'][list_cbc_mask]
    m1_all = list_cbc['m1'][list_cbc_mask]
    m2_all = list_cbc['m2'][list_cbc_mask]
    dL_all = list_cbc['dL'][list_cbc_mask]
    z_all = list_cbc['z'][list_cbc_mask]
    ra_all = list_cbc['ra'][list_cbc_mask]
    dec_all = list_cbc['dec'][list_cbc_mask]
    psi_all = list_cbc['psi'][list_cbc_mask]  # 修正：这里应使用 'psi'
    phi_c_all = list_cbc['phi_c'][list_cbc_mask]
    iota_all = list_cbc['iota'][list_cbc_mask]

    # 修复循环范围：使用筛选后的事件数量
    num_events = len(tc_all)
    report_interval = max(1, int(num_events * 0.10))
    print(f"✅ 找到 {num_events} 个符合时间条件的事件。开始注入信号...")

    for EVENT in range(num_events):
        # 打印进度信息（每 1% 打印一次）
        if num_events > 0 and (EVENT + 1) % report_interval == 0:
            current_percent = (EVENT + 1) / num_events * 100
            print(f"🚀 进度: {current_percent:.0f}% ({EVENT + 1}/{num_events} 个事件已处理)")

        # 从筛选后的数组中，提取当前循环的单个事件参数
        tc = tc_all[EVENT]
        m1 = m1_all[EVENT]  # 源系质量
        m2 = m2_all[EVENT]  # 源系质量
        dL = dL_all[EVENT]
        z = z_all[EVENT]
        ra = ra_all[EVENT]
        dec = dec_all[EVENT]
        psi = psi_all[EVENT]
        phi_c = phi_c_all[EVENT]
        iota = iota_all[EVENT]

        # 转换为探测器系质量（使用单个数值进行计算）
        m1_det, m2_det = m1 * (1 + z), m2 * (1 + z)

        try:

            injection_parameters = dict(mass_1=m1_det, mass_2=m2_det, a_1=0.0, a_2=0.0,
                                        luminosity_distance=dL, psi=psi, phase=phi_c, geocent_time=tc,
                                        ra=ra, dec=dec, theta_jn=iota, tilt_1=0.0, tilt_2=0.0,
                                        phi_12=0.0, phi_jl=0.0)
            # ... 后续的波形生成和注入代码照常执行 ...

            # 获得时域波形（plus和cross极化）
            waveform_generator = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=source_model,
                waveform_arguments=waveform_arguments,
                parameter_conversion=pc)
            tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
            plus, cross = tds['plus'], tds['cross']
        except RuntimeError as e:
            # 关键排查点：打印出当前事件的参数
            print(f"--- 处理事件 {EVENT + 1}/{num_events}出错，将跳过 ---")
            print(f"出错的系统参数为：tc: {tc}, m1_det: {m1_det}, m2_det: {m2_det}, z: {z}")
            continue

        # --------------------UPDATES------------------------
        # roll the data to put ringdown part at the end of the signal
        # move the first 0.25s to the end
        # 因为 bilby 库在生成引力波波形时，默认将并合时间 ($t_c$) 对齐到波形数组的中心附近，
        # 而通过 np.roll 操作，你可以人为地将信号的关键部分bu
        # （高频、并合、尾声）移动到数组的末尾，以适应后续的快速傅里叶变换（FFT）或时间窗口处理。
        plus = np.roll(plus, -512)    # shape=2048*2048
        cross = np.roll(cross, -512)  # shape=2048*2048
        # --------------------UPDATES------------------------

        time = waveform_generator.time_array
        dt = time[1] - time[0]  # 采样间隔

        # 切掉信号中早于数据段起始时间st的部分
        if tc <= et:
            print(tc,et)
            event_dur = int((tc - st) / dt)   # st到tc之间一共有多少个点
            print(2048*2048-event_dur,plus[-event_dur:].shape,event_dur)
            ptemp = np.zeros_like(plus)
            ctemp = np.zeros_like(cross)
            ptemp[-event_dur:] = plus[-event_dur:]
            ctemp[-event_dur:] = cross[-event_dur:]
            plus = np.copy(ptemp)
            cross = np.copy(ctemp)

        # 切掉信号中晚于数据段结束时间et的部分
        if tc >= et:
            idx = int(abs(tc - et) / dt)
            nidx = plus.size - idx
            ptemp = np.zeros_like(plus)
            ctemp = np.zeros_like(cross)
            ptemp[:nidx] = plus[:nidx]
            ctemp[:nidx] = cross[:nidx]
            plus = np.copy(ptemp)
            cross = np.copy(ctemp)

        # print(f'合并时间：{tc},结束时间：{et}')

        # 对裁剪好的时域波形plus（加极化）进行快速傅里叶变换rfft
        fds = dict(plus=np.fft.rfft(plus) * dt, cross=np.fft.rfft(cross) * dt)
        interferometers.inject_signal(injection_parameters, injection_polarizations=fds, raise_error=False)
        #        print(interferometers[0].strain_data.to_pycbc_timeseries())

        # 只保留旋进阶段的信号，除去并合之后的信号
        #        print(interferometers[0].time_domain_strain[:int((tc - st) * sampling_frequency)+1])
        #        print(interferometers[0].time_domain_strain[int((tc - st) * sampling_frequency):])
        #        print()
        #        print(tc,st,int((tc - st) * sampling_frequency))
        interferometers[0].time_domain_strain[int((tc - st) * sampling_frequency):] = 0
        interferometers[1].time_domain_strain[int((tc - st) * sampling_frequency):] = 0

        #        print(interferometers[0].time_domain_strain[:int((tc - st) * sampling_frequency)-1])
        #        print(interferometers[0].time_domain_strain[int((tc - st) * sampling_frequency):])
        # print()

    # save as gwf frames
    print("\n💾 所有信号注入完成，开始写入 GWF 文件...")
    base_channel_name = 'TEST_INJ'
    gps_start_time = ifo.time_array[0]
    for ix, ifo in enumerate(interferometers):
        new_channel_names = ifo.name + ':' + base_channel_name
        inj_channel = gwpy.detector.Channel(ifo.name + ':' + new_channel_names)
        injected_ts = ts.TimeSeries(ifo.time_domain_strain, times=ifo.time_array,
                                    name=new_channel_names, channel=inj_channel, dtype=float)
        # file_name = ifo.name + '-STRAIN-' + str(st) + '-' + str(duration) + '.gwf'
        file_name = f"{ifo.name}-STRAIN-{jobNumber:05d}-duration.gwf"

        output_dir = './cbc_waveform_' + ifo.name
        if not os.path.exists(output_dir):
            # 如果目录不存在，尝试创建它
            os.makedirs(output_dir)
            print(f"⚠️ 警告: 创建了输出目录: {output_dir}")
        full_path = os.path.join(output_dir, file_name)
        # file_name = ifo.name + '-STRAIN-' + str(st) + '-' + str(duration) + '.gwf'
        injected_ts.write(full_path)

    print(f"🎉 文件生成完毕。保存了 {interferometers[0].name} 和 {interferometers[1].name} 的 GWF 文件。")


if __name__ == '__main__':
    # 定义循环范围
    start_job = 1
    # end_job = 15409
    end_job = 10000

    for job_number in range(start_job, end_job + 1):
        print(f'----job_number{job_number}开始分析------')
        generate_frames_for_job(job_number)
        print(f'✅----job_number{job_number}分析完毕------')

    print(f"\n\n=============== ✅ 所有任务 (Job 1 到 Job {end_job}) 已完成 ===============")
