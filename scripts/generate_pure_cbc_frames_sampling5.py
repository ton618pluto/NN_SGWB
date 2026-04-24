import os
import bilby
import logging
import pandas as pd
import gwpy.timeseries as ts
import gwpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 减少日志输出，避免多进程刷屏
bilby.core.utils.setup_logger(log_level=logging.WARNING)
SNR_THRESH = 12.0
#start = 7000
start = 0
temp_idx = 0
input_dir = './parameter_sampling_train/v4'
output_base_dir = './training_set/v4/training_set0'
#num_populations = 8000
num_populations = 100
#all_jobs=44
# all_jobs=24
all_jobs=5


def process_population(pop_idx, input_dir, output_base_dir, config):
    """
    处理单个超参文件的函数，将被多进程调用
    """
    # 提取配置
    duration = config['duration']
    sampling_frequency = config['sampling_frequency']
    minimum_frequency = config['minimum_frequency']
    reference_frequency = config['reference_frequency']
    samples_per_pop = config['samples_per_pop']

    param_file = os.path.join(input_dir, f'CBC_params_example{pop_idx:05d}.npz')
    if not os.path.exists(param_file):
        return []

    results = []
    list_cbc = np.load(param_file)

    # 提取标签
    try:
        pop_labels = {k: float(list_cbc[f'true_{k}']) for k in ['zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max', 'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q']}
    except:
        pop_labels = {k: float(list_cbc[k]) for k in ['zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max', 'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q']}

    # 这里使用你固定的 range(1, 44)
    random_jobs = list(range(1, all_jobs))

    for s_idx, jobNumber in enumerate(random_jobs):
        sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"

        # ====== 断点恢复关键 ======
        h1_path = os.path.join(output_base_dir, 'H1', f"H1-{sample_id_str}.gwf")
        l1_path = os.path.join(output_base_dir, 'L1', f"L1-{sample_id_str}.gwf")
    
        # 如果两个文件都存在，说明已完成
        if os.path.exists(h1_path) and os.path.exists(l1_path):
            #   print(f"{sample_id_str} 已存在，跳过")
            continue
    
        # if pop_idx == start and s_idx < temp_idx: continue
        st = (jobNumber - 1) * duration
        et = st + duration
        sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"

        tc_all_raw = list_cbc['tc']
        m1_raw = list_cbc['m1']
        m2_raw = list_cbc['m2']
        z_raw = list_cbc['z']

        def estimation_t0(tc, m1, m2):
            MTSUN_SI = 4.925491025543576e-06
            f0 = minimum_frequency - 1
            mc = (m1 * m2) ** (3. / 5) / (m1 + m2) ** (1. / 5)
            C = 5 ** (3. / 8) / (8 * np.pi)
            return tc - (f0 / C) ** (-8. / 3) * (mc * MTSUN_SI) ** (-5. / 3)

        m1_det_logic, m2_det_logic = m1_raw * (1 + z_raw), m2_raw * (1 + z_raw)
        t0_all = estimation_t0(tc_all_raw, m1_det_logic, m2_det_logic)
        list_cbc_mask = (tc_all_raw > st) * (t0_all < et)

        # 每次采样重新初始化干涉仪，防止信号叠加残留
        interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
        for ifo in interferometers:
            ifo.minimum_frequency = minimum_frequency
            ifo.sampling_frequency = sampling_frequency
            ifo.duration = duration

        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration, start_time=st)

        waveform_arguments = {'waveform_approximant': 'IMRPhenomXAS',
                              'reference_frequency': reference_frequency,
                              'minimum_frequency': minimum_frequency}

        tc_all = list_cbc['tc'][list_cbc_mask]
        m1_all = list_cbc['m1'][list_cbc_mask]
        m2_all = list_cbc['m2'][list_cbc_mask]
        dL_all = list_cbc['dL'][list_cbc_mask]
        z_all = list_cbc['z'][list_cbc_mask]
        ra_all = list_cbc['ra'][list_cbc_mask]
        dec_all = list_cbc['dec'][list_cbc_mask]
        psi_all = list_cbc['psi'][list_cbc_mask]
        phi_c_all = list_cbc['phi_c'][list_cbc_mask]
        iota_all = list_cbc['iota'][list_cbc_mask]

        num_events = len(tc_all)

        skipped_strong = 0
        injected_weak = 0
        errored= 0
        for EVENT in range(num_events):
            tc, m1, m2 = tc_all[EVENT], m1_all[EVENT], m2_all[EVENT]
            dL, z, ra, dec = dL_all[EVENT], z_all[EVENT], ra_all[EVENT], dec_all[EVENT]
            psi, phi_c, iota = psi_all[EVENT], phi_c_all[EVENT], iota_all[EVENT]
            m1_det, m2_det = m1 * (1 + z), m2 * (1 + z)

            try:
                injection_parameters = dict(mass_1=m1_det, mass_2=m2_det, a_1=0.0, a_2=0.0,
                                            luminosity_distance=dL, psi=psi, phase=phi_c, geocent_time=tc,
                                            ra=ra, dec=dec, theta_jn=iota, tilt_1=0.0, tilt_2=0.0,
                                            phi_12=0.0, phi_jl=0.0)

                waveform_generator = bilby.gw.WaveformGenerator(
                    duration=duration, sampling_frequency=sampling_frequency,
                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    waveform_arguments=waveform_arguments,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

                tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
                plus, cross = tds['plus'], tds['cross']

                plus = np.roll(plus, -512)
                cross = np.roll(cross, -512)
                time = waveform_generator.time_array
                dt = time[1] - time[0]

                if tc <= et:
                    event_dur = int((tc - st) / dt)
                    ptemp, ctemp = np.zeros_like(plus), np.zeros_like(cross)
                    ptemp[-event_dur:] = plus[-event_dur:]
                    ctemp[-event_dur:] = cross[-event_dur:]
                    plus, cross = ptemp, ctemp

                if tc >= et:
                    idx = int(abs(tc - et) / dt)
                    nidx = plus.size - idx
                    ptemp, ctemp = np.zeros_like(plus), np.zeros_like(cross)
                    ptemp[:nidx] = plus[:nidx]
                    ctemp[:nidx] = cross[:nidx]
                    plus, cross = ptemp, ctemp

                fds = dict(plus=np.fft.rfft(plus) * dt, cross=np.fft.rfft(cross) * dt)
                # 去除过于强的信号
                snr2_net = 0.0
                for ifo in interferometers:
                    det_signal_fd = ifo.get_detector_response(fds, injection_parameters)
                    snr2_net += ifo.optimal_snr_squared(det_signal_fd)

                snr_net = float(np.sqrt(snr2_net))
                if snr_net >= SNR_THRESH:
                    skipped_strong += 1
                    continue

                injected_weak += 1
                
                interferometers.inject_signal(injection_parameters, injection_polarizations=fds, raise_error=False)

                # 按原逻辑清零合并后的尾部
                for ifo_obj in interferometers:
                    idx_cut = int((tc - st) * sampling_frequency)
                    if idx_cut < len(ifo_obj.time_domain_strain):
                        ifo_obj.time_domain_strain[idx_cut:] = 0

            except Exception as e:
                errored+=1
                continue

                
        # 保存 GWF 文件
        print(f'总的事件数为：{num_events}，出错的事件数为：{errored}，\n注入事件数为:{injected_weak},跳过的事件数为：{skipped_strong} ')
        for ifo in interferometers:
            new_channel_names = ifo.name + ':TEST_INJ'
            inj_channel = gwpy.detector.Channel(new_channel_names)
            injected_ts = ts.TimeSeries(ifo.time_domain_strain, times=ifo.time_array,
                                        name=new_channel_names, channel=inj_channel, dtype=float)
            full_path = os.path.join(output_base_dir, ifo.name, f"{ifo.name}-{sample_id_str}.gwf")
            injected_ts.write(full_path)

        results.append({'sample_id': sample_id_str, 'st': st, **pop_labels})

    return results


def generate_samples(max_workers=8):

    csv_path = os.path.join(output_base_dir, 'train_idx.csv')

    config = {
        'samples_per_pop': 5,
        'duration': 2048,
        'sampling_frequency': 2048,
        'minimum_frequency': 10,
        'reference_frequency': 25.
    }

    for ifo_name in ['H1', 'L1']:
        os.makedirs(os.path.join(output_base_dir, ifo_name), exist_ok=True)

    # 使用进程池
    print(f"?? 启动多进程处理，最大进程数: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {executor.submit(process_population, i, input_dir, output_base_dir, config): i
                   for i in range(start, num_populations)}

        for future in as_completed(futures):
            pop_idx = futures[future]
            try:
                pop_results = future.result()
                if pop_results:
                    df_tmp = pd.DataFrame(pop_results)
                    # 主进程负责写入文件，确保线程/进程安全
                    if not os.path.exists(csv_path):
                        df_tmp.to_csv(csv_path, index=False, mode='w')
                    else:
                        df_tmp.to_csv(csv_path, index=False, mode='a', header=False)
                    print(f"? 超参组 {pop_idx:05d} 处理完成并写入CSV。")
            except Exception as e:
                print(f"? 超参组 {pop_idx:05d} 运行出错: {e}")


if __name__ == '__main__':
    # 建议 max_workers 设为 CPU 核心数的一半或 2/3，避免内存溢出
    generate_samples(max_workers=22)