# generate_pure_cbc_frames.py
import os
import bilby
import logging
import pandas as pd
import gwpy.timeseries as ts
import gwpy
import numpy as np

bilby.core.utils.setup_logger(log_level=logging.WARNING)
SNR_THRESH = 12.0
temp_idx = 0
start = 0
num_populations = 100  # 采样的超参数组数范围
input_dir = './parameter_sampling_val/v2'
output_base_dir = './val_set/v2/val_set1'


def generate_samples():
    # --- 保持你的参数定义 ---

    # input_dir = './parameter_sampling_val'
    # input_dir = './parameter_sampling_test'

    # output_base_dir = './val_set'
    # output_base_dir = './test_set'

    samples_per_pop = 5
    duration = 2048
    sampling_frequency = 2048
    minimum_frequency = 10
    reference_frequency = 25.
    # np.random.seed(3333)
    np.random.seed(4444)
    # np.random.seed(5555)

    for ifo_name in ['H1', 'L1']:
        os.makedirs(os.path.join(output_base_dir, ifo_name), exist_ok=True)

    all_sample_registry = []

    # 外层循环 200 个文件
    for pop_idx in range(start, num_populations):
        print(f'✅ 开始采样CBC_params_example{pop_idx:05d}.npz')
        param_file = os.path.join(input_dir, f'CBC_params_example{pop_idx:05d}.npz')
        if not os.path.exists(param_file):
            print(f'没有CBC_params_example{pop_idx:05d}.npz这个文件...')
            continue

        list_cbc = np.load(param_file)
        # 临时存储当前文件的 5 个采样结果
        current_pop_registry = []

        # 提取标签
        try:
            pop_labels = {k: float(list_cbc[f'true_{k}']) for k in ['zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max', 'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q']}
        except:
            pop_labels = {k: float(list_cbc[k]) for k in ['zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max', 'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q']}

        # random_jobs = np.random.choice(np.arange(1, 15410), size=samples_per_pop, replace=False)
        random_jobs = list(range(1, 6))
        # print(f'采样的5个jobnumber为{random_jobs}')
        # 内层循环 5 次采样
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
            # 这里的 st, et 变量名完全按你的来
            st = (jobNumber - 1) * duration
            et = st + duration
            sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"

            # --- 以下是你原来的逻辑，变量名一个字不改 ---
            tc = list_cbc['tc']
            m1 = list_cbc['m1']
            m2 = list_cbc['m2']
            z = list_cbc['z']

            def estimation_t0(tc, m1, m2):
                MTSUN_SI = 4.925491025543576e-06
                f0 = minimum_frequency - 1
                mc = (m1 * m2) ** (3. / 5) / (m1 + m2) ** (1. / 5)
                C = 5 ** (3. / 8) / (8 * np.pi)
                real_t0 = tc - (f0 / C) ** (-8. / 3) * (mc * MTSUN_SI) ** (-5. / 3)
                return real_t0

            m1_det_logic, m2_det_logic = m1 * (1 + z), m2 * (1 + z)
            t0 = estimation_t0(tc, m1_det_logic, m2_det_logic)
            list_cbc_mask = (tc > st) * (t0 < et)

            interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
            for ifo in interferometers:
                ifo.minimum_frequency = minimum_frequency
                ifo.sampling_frequency = sampling_frequency
                ifo.duration = duration

            interferometers.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=st)

            waveform_arguments = {'waveform_approximant': 'IMRPhenomXAS',
                                  'reference_frequency': reference_frequency,
                                  'minimum_frequency': minimum_frequency}
            source_model = bilby.gw.source.lal_binary_black_hole
            pc = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

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
            num_events = len(tc_all)
            report_interval = max(1, int(num_events * 0.10))
            print(f"📂 {sample_id_str}: 找到 {num_events} 个事件")

            skipped_strong = 0
            injected_weak = 0
            errored= 0
            for EVENT in range(num_events):
                # print(f"DEBUG: Processing EVENT {EVENT}, tc={tc}, m1={m1}, m2={m2}")
                if num_events > 0 and (EVENT + 1) % report_interval == 0:
                    current_percent = (EVENT + 1) / num_events * 100
                    print(f"🚀 进度: {current_percent:.0f}% ({EVENT + 1}/{num_events} 个事件已处理)")
                # 为了不改你后面的逻辑，这里把单次循环的变量名依然设为 tc, m1 等
                tc = tc_all[EVENT]
                m1 = m1_all[EVENT]
                m2 = m2_all[EVENT]
                dL = dL_all[EVENT]
                z = z_all[EVENT]
                ra = ra_all[EVENT]
                dec = dec_all[EVENT]
                psi = psi_all[EVENT]
                phi_c = phi_c_all[EVENT]
                iota = iota_all[EVENT]
                m1_det, m2_det = m1 * (1 + z), m2 * (1 + z)

                try:
                    injection_parameters = dict(mass_1=m1_det, mass_2=m2_det, a_1=0.0, a_2=0.0,
                                                luminosity_distance=dL, psi=psi, phase=phi_c, geocent_time=tc,
                                                ra=ra, dec=dec, theta_jn=iota, tilt_1=0.0, tilt_2=0.0,
                                                phi_12=0.0, phi_jl=0.0)

                    waveform_generator = bilby.gw.WaveformGenerator(
                        duration=duration, sampling_frequency=sampling_frequency,
                        frequency_domain_source_model=source_model,
                        waveform_arguments=waveform_arguments,
                        parameter_conversion=pc)

                    tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
                    plus, cross = tds['plus'], tds['cross']
                except Exception:
                    errored+=1
                    print(f"--- 处理事件 {EVENT + 1}/{num_events}出错，将跳过 ---")
                    print(f"出错的系统参数为：tc: {tc}, m1_det: {m1_det}, m2_det: {m2_det}, z: {z}")
                    continue

                plus = np.roll(plus, -512)
                cross = np.roll(cross, -512)

                time = waveform_generator.time_array
                dt = time[1] - time[0]
                # print('dt',dt)

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

                # print(fds)
                interferometers.inject_signal(injection_parameters, injection_polarizations=fds, raise_error=False)

                # 这两行变量名完全按你原来的写
                interferometers[0].time_domain_strain[int((tc - st) * sampling_frequency):] = 0
                interferometers[1].time_domain_strain[int((tc - st) * sampling_frequency):] = 0

            print(f'总的事件数为：{num_events}，出错的事件数为：{errored}，\n注入事件数为:{injected_weak},跳过的事件数为：{skipped_strong} ')
            # 保存 GWF
            for ix, ifo in enumerate(interferometers):
                new_channel_names = ifo.name + ':TEST_INJ'
                inj_channel = gwpy.detector.Channel(new_channel_names)
                injected_ts = ts.TimeSeries(ifo.time_domain_strain, times=ifo.time_array,
                                            name=new_channel_names, channel=inj_channel, dtype=float)
                full_path = os.path.join(output_base_dir, ifo.name, f"{ifo.name}-{sample_id_str}.gwf")
                injected_ts.write(full_path)

            # 将当前采样结果存入临时列表
            current_pop_registry.append({'sample_id': sample_id_str, 'st': st, **pop_labels})
            # --- 核心改动：每处理完一个超参文件，立即写入 CSV ---
            csv_path = os.path.join(output_base_dir, 'train_idx.csv')
            df_tmp = pd.DataFrame(current_pop_registry)

            if not os.path.exists(csv_path):
                # 第一次写入，包含表头
                df_tmp.to_csv(csv_path, index=False, mode='w')
            else:
                # 追加写入，不包含表头
                df_tmp.to_csv(csv_path, index=False, mode='a', header=False)
            current_pop_registry = []

        print(f"✅ 超参组 {pop_idx:05d} 的标签已同步写入 CSV。")
        # current_pop_registry=[]
        print()
        print(f"✅ CBC_params_example{pop_idx:05d}.npz采样结束")

    # pd.DataFrame(all_sample_registry).to_csv(os.path.join(output_base_dir, 'train_idx.csv'), index=False)


if __name__ == '__main__':
    generate_samples()