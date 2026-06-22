# generate_pure_cbc_frames.py
import logging
import os

import bilby
import gwpy
import gwpy.timeseries as ts
import numpy as np
import pandas as pd


bilby.core.utils.setup_logger(log_level=logging.WARNING)

SNR_THRESH = 12.0
start = 0
temp_idx = 0
# sampling_frequency = 2048
sampling_frequency = 32
version = 'v21'
num_populations = 1000
input_dir = f'./parameter_sampling_train/{version}'
output_base_dir = f'./training_set/v1001/training_set0'
all_jobs = 2


def generate_samples():
    config = {
        'samples_per_pop': 5,
        'duration': 2048,
        'sampling_frequency': sampling_frequency,
        'minimum_frequency': 10,
        'reference_frequency': 25.
    }

    samples_per_pop = config['samples_per_pop']
    duration = config['duration']
    sampling_frequency_cfg = config['sampling_frequency']
    minimum_frequency = config['minimum_frequency']
    reference_frequency = config['reference_frequency']

    np.random.seed(3333)

    for ifo_name in ['H1', 'L1']:
        os.makedirs(os.path.join(output_base_dir, ifo_name), exist_ok=True)

    for pop_idx in range(start, num_populations):
        print(f'开始采样 CBC_params_example{pop_idx:05d}.npz')
        param_file = os.path.join(input_dir, f'CBC_params_example{pop_idx:05d}.npz')
        if not os.path.exists(param_file):
            print(f'没有 CBC_params_example{pop_idx:05d}.npz 这个文件')
            continue

        list_cbc = np.load(param_file)
        current_pop_registry = []

        try:
            pop_labels = {
                k: float(list_cbc[f'true_{k}'])
                for k in [
                    'zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max',
                    'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q'
                ]
            }
        except Exception:
            pop_labels = {
                k: float(list_cbc[k])
                for k in [
                    'zp', 'alpha_z', 'beta_z', 'alpha_m', 'm_max',
                    'delta_m', 'm_min', 'lambda_peak', 'mu_m', 'sigma_m', 'beta_q'
                ]
            }

        random_jobs = list(range(1, all_jobs))
        for s_idx, job_number in enumerate(random_jobs):
            sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"

            h1_path = os.path.join(output_base_dir, 'H1', f"H1-{sample_id_str}.gwf")
            l1_path = os.path.join(output_base_dir, 'L1', f"L1-{sample_id_str}.gwf")
            if os.path.exists(h1_path) and os.path.exists(l1_path):
                continue

            st = (job_number - 1) * duration
            et = st + duration

            tc = list_cbc['tc']
            m1 = list_cbc['m1']
            m2 = list_cbc['m2']
            z = list_cbc['z']

            def estimation_t0(tc_values, m1_values, m2_values):
                mtsun_si = 4.925491025543576e-06
                f0 = minimum_frequency - 1
                mc = (m1_values * m2_values) ** (3.0 / 5.0) / (m1_values + m2_values) ** (1.0 / 5.0)
                c_const = 5 ** (3.0 / 8.0) / (8 * np.pi)
                return tc_values - (f0 / c_const) ** (-8.0 / 3.0) * (mc * mtsun_si) ** (-5.0 / 3.0)

            m1_det_logic = m1 * (1 + z)
            m2_det_logic = m2 * (1 + z)
            t0 = estimation_t0(tc, m1_det_logic, m2_det_logic)
            list_cbc_mask = (tc > st) * (t0 < et)

            interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
            for ifo in interferometers:
                ifo.minimum_frequency = minimum_frequency
                ifo.sampling_frequency = sampling_frequency_cfg
                ifo.duration = duration

            interferometers.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_frequency_cfg,
                duration=duration,
                start_time=st
            )

            waveform_arguments = {
                'waveform_approximant': 'IMRPhenomXAS',
                'reference_frequency': reference_frequency,
                'minimum_frequency': minimum_frequency,
            }
            source_model = bilby.gw.source.lal_binary_black_hole
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters

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
            report_interval = max(1, int(num_events * 0.10)) if num_events > 0 else 1
            print(f"{sample_id_str}: 找到 {num_events} 个事件")

            skipped_strong = 0
            injected_weak = 0
            errored = 0

            for event_idx in range(num_events):
                if num_events > 0 and (event_idx + 1) % report_interval == 0:
                    current_percent = (event_idx + 1) / num_events * 100
                    print(f"进度: {current_percent:.0f}% ({event_idx + 1}/{num_events})")

                tc = tc_all[event_idx]
                m1 = m1_all[event_idx]
                m2 = m2_all[event_idx]
                dL = dL_all[event_idx]
                z = z_all[event_idx]
                ra = ra_all[event_idx]
                dec = dec_all[event_idx]
                psi = psi_all[event_idx]
                phi_c = phi_c_all[event_idx]
                iota = iota_all[event_idx]
                m1_det = m1 * (1 + z)
                m2_det = m2 * (1 + z)

                try:
                    injection_parameters = dict(
                        mass_1=m1_det,
                        mass_2=m2_det,
                        a_1=0.0,
                        a_2=0.0,
                        luminosity_distance=dL,
                        psi=psi,
                        phase=phi_c,
                        geocent_time=tc,
                        ra=ra,
                        dec=dec,
                        theta_jn=iota,
                        tilt_1=0.0,
                        tilt_2=0.0,
                        phi_12=0.0,
                        phi_jl=0.0,
                    )

                    waveform_generator = bilby.gw.WaveformGenerator(
                        duration=duration,
                        sampling_frequency=sampling_frequency_cfg,
                        frequency_domain_source_model=source_model,
                        waveform_arguments=waveform_arguments,
                        parameter_conversion=parameter_conversion,
                    )

                    tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
                    plus, cross = tds['plus'], tds['cross']
                except Exception:
                    errored += 1
                    print(f"处理事件 {event_idx + 1}/{num_events} 出错，跳过")
                    print(f"出错参数: tc={tc}, m1_det={m1_det}, m2_det={m2_det}, z={z}")
                    continue

                plus = np.roll(plus, -512)
                cross = np.roll(cross, -512)

                time = waveform_generator.time_array
                dt = time[1] - time[0]

                if tc <= et:
                    event_dur = int((tc - st) / dt)
                    ptemp = np.zeros_like(plus)
                    ctemp = np.zeros_like(cross)
                    ptemp[-event_dur:] = plus[-event_dur:]
                    ctemp[-event_dur:] = cross[-event_dur:]
                    plus, cross = ptemp, ctemp

                if tc >= et:
                    idx = int(abs(tc - et) / dt)
                    nidx = plus.size - idx
                    ptemp = np.zeros_like(plus)
                    ctemp = np.zeros_like(cross)
                    ptemp[:nidx] = plus[:nidx]
                    ctemp[:nidx] = cross[:nidx]
                    plus, cross = ptemp, ctemp

                fds = dict(plus=np.fft.rfft(plus) * dt, cross=np.fft.rfft(cross) * dt)

                snr2_net = 0.0
                for ifo in interferometers:
                    det_signal_fd = ifo.get_detector_response(fds, injection_parameters)
                    snr2_net += ifo.optimal_snr_squared(det_signal_fd)

                snr_net = float(np.sqrt(snr2_net))
                if snr_net >= SNR_THRESH:
                    skipped_strong += 1
                    continue

                injected_weak += 1
                interferometers.inject_signal(
                    injection_parameters,
                    injection_polarizations=fds,
                    raise_error=False
                )

                interferometers[0].time_domain_strain[int((tc - st) * sampling_frequency_cfg):] = 0
                interferometers[1].time_domain_strain[int((tc - st) * sampling_frequency_cfg):] = 0

            print(
                f'总事件数={num_events}, 出错事件数={errored}, '
                f'注入事件数={injected_weak}, 跳过强事件数={skipped_strong}'
            )

            for ifo in interferometers:
                new_channel_names = ifo.name + ':TEST_INJ'
                inj_channel = gwpy.detector.Channel(new_channel_names)
                injected_ts = ts.TimeSeries(
                    ifo.time_domain_strain,
                    times=ifo.time_array,
                    name=new_channel_names,
                    channel=inj_channel,
                    dtype=float
                )
                full_path = os.path.join(output_base_dir, ifo.name, f"{ifo.name}-{sample_id_str}.gwf")
                injected_ts.write(full_path)

            current_pop_registry.append({'sample_id': sample_id_str, 'st': st, **pop_labels})
            csv_path = os.path.join(output_base_dir, 'train_idx.csv')
            df_tmp = pd.DataFrame(current_pop_registry)

            if not os.path.exists(csv_path):
                df_tmp.to_csv(csv_path, index=False, mode='w')
            else:
                df_tmp.to_csv(csv_path, index=False, mode='a', header=False)
            current_pop_registry = []

        print(f"超参数组 {pop_idx:05d} 的标签已同步写入 CSV")
        print(f"CBC_params_example{pop_idx:05d}.npz 采样结束")


if __name__ == '__main__':
    generate_samples()
