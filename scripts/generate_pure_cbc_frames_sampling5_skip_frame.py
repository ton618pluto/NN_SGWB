import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import bilby
import gwpy
import gwpy.timeseries as ts
import numpy as np
import pandas as pd


bilby.core.utils.setup_logger(log_level=logging.WARNING)

SNR_THRESH = 12.0
start = 0
sampling_frequency = 64
version = "v21"
num_populations = 100000
all_jobs = 2
input_dir = f"./parameter_sampling_train/{version}"
output_base_dir = f"./training_set/{version}/training_set0"
skip_base_dir = os.path.join(output_base_dir, "processSkip")


def estimation_t0(tc, m1, m2, minimum_frequency):
    mtsun_si = 4.925491025543576e-06
    f0 = minimum_frequency - 1
    mc = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    c_const = 5 ** (3.0 / 8.0) / (8 * np.pi)
    return tc - (f0 / c_const) ** (-8.0 / 3.0) * (mc * mtsun_si) ** (-5.0 / 3.0)


def build_event_polarizations(waveform_generator, injection_parameters, st, et):
    tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
    plus, cross = tds["plus"], tds["cross"]

    plus = np.roll(plus, -512)
    cross = np.roll(cross, -512)
    time = waveform_generator.time_array
    dt = time[1] - time[0]
    tc = injection_parameters["geocent_time"]

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

    return dict(plus=np.fft.rfft(plus) * dt, cross=np.fft.rfft(cross) * dt)


def process_population(pop_idx, input_dir, output_base_dir, config):
    duration = config["duration"]
    sampling_frequency = config["sampling_frequency"]
    minimum_frequency = config["minimum_frequency"]
    reference_frequency = config["reference_frequency"]

    param_file = os.path.join(input_dir, f"CBC_params_example{pop_idx:05d}.npz")
    if not os.path.exists(param_file):
        return {"written_rows": [], "skipped_rows": []}

    list_cbc = np.load(param_file)

    try:
        pop_labels = {
            k: float(list_cbc[f"true_{k}"])
            for k in [
                "zp",
                "alpha_z",
                "beta_z",
                "alpha_m",
                "m_max",
                "delta_m",
                "m_min",
                "lambda_peak",
                "mu_m",
                "sigma_m",
                "beta_q",
            ]
        }
    except Exception:
        pop_labels = {
            k: float(list_cbc[k])
            for k in [
                "zp",
                "alpha_z",
                "beta_z",
                "alpha_m",
                "m_max",
                "delta_m",
                "m_min",
                "lambda_peak",
                "mu_m",
                "sigma_m",
                "beta_q",
            ]
        }

    written_rows = []
    skipped_rows = []
    random_jobs = list(range(1, all_jobs))

    for s_idx, job_number in enumerate(random_jobs):
        sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"
        h1_path = os.path.join(output_base_dir, "H1", f"H1-{sample_id_str}.gwf")
        l1_path = os.path.join(output_base_dir, "L1", f"L1-{sample_id_str}.gwf")

        if os.path.exists(h1_path) and os.path.exists(l1_path):
            continue

        st = (job_number - 1) * duration
        et = st + duration

        tc_all_raw = list_cbc["tc"]
        m1_raw = list_cbc["m1"]
        m2_raw = list_cbc["m2"]
        z_raw = list_cbc["z"]

        m1_det_logic = m1_raw * (1 + z_raw)
        m2_det_logic = m2_raw * (1 + z_raw)
        t0_all = estimation_t0(tc_all_raw, m1_det_logic, m2_det_logic, minimum_frequency)
        list_cbc_mask = (tc_all_raw > st) * (t0_all < et)

        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        for ifo in interferometers:
            ifo.minimum_frequency = minimum_frequency
            ifo.sampling_frequency = sampling_frequency
            ifo.duration = duration

        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration, start_time=st
        )

        waveform_arguments = {
            "waveform_approximant": "IMRPhenomXAS",
            "reference_frequency": reference_frequency,
            "minimum_frequency": minimum_frequency,
        }
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=waveform_arguments,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        )

        tc_all = list_cbc["tc"][list_cbc_mask]
        m1_all = list_cbc["m1"][list_cbc_mask]
        m2_all = list_cbc["m2"][list_cbc_mask]
        dL_all = list_cbc["dL"][list_cbc_mask]
        z_all = list_cbc["z"][list_cbc_mask]
        ra_all = list_cbc["ra"][list_cbc_mask]
        dec_all = list_cbc["dec"][list_cbc_mask]
        psi_all = list_cbc["psi"][list_cbc_mask]
        phi_c_all = list_cbc["phi_c"][list_cbc_mask]
        iota_all = list_cbc["iota"][list_cbc_mask]

        num_events = len(tc_all)
        prepared_events = []
        errored = 0
        strong_event_count = 0

        for event_idx in range(num_events):
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

                fds = build_event_polarizations(waveform_generator, injection_parameters, st, et)

                snr2_net = 0.0
                for ifo in interferometers:
                    det_signal_fd = ifo.get_detector_response(fds, injection_parameters)
                    snr2_net += ifo.optimal_snr_squared(det_signal_fd)

                snr_net = float(np.sqrt(snr2_net))
                if snr_net >= SNR_THRESH:
                    strong_event_count += 1
                    break

                prepared_events.append((tc, injection_parameters, fds))
            except Exception:
                errored += 1
                continue

        if strong_event_count > 0:
            skipped_rows.append(
                {
                    "sample_id": sample_id_str,
                    "h1_file": f"H1-{sample_id_str}.gwf",
                    "l1_file": f"L1-{sample_id_str}.gwf",
                    "num_events": num_events,
                    "errored_events": errored,
                    "strong_events": strong_event_count,
                    "reason": f"contains_event_with_snr>={SNR_THRESH}",
                }
            )
            print(
                f"整段 frame 跳过: {sample_id_str}, 总事件数={num_events}, "
                f"强事件数={strong_event_count}, 出错事件数={errored}"
            )
            continue

        injected_weak = 0
        for tc, injection_parameters, fds in prepared_events:
            injected_weak += 1
            interferometers.inject_signal(
                injection_parameters,
                injection_polarizations=fds,
                raise_error=False,
            )

            for ifo_obj in interferometers:
                idx_cut = int((tc - st) * sampling_frequency)
                if idx_cut < len(ifo_obj.time_domain_strain):
                    ifo_obj.time_domain_strain[idx_cut:] = 0

        for ifo in interferometers:
            new_channel_name = ifo.name + ":TEST_INJ"
            inj_channel = gwpy.detector.Channel(new_channel_name)
            injected_ts = ts.TimeSeries(
                ifo.time_domain_strain,
                times=ifo.time_array,
                name=new_channel_name,
                channel=inj_channel,
                dtype=float,
            )
            full_path = os.path.join(output_base_dir, ifo.name, f"{ifo.name}-{sample_id_str}.gwf")
            injected_ts.write(full_path)

        written_rows.append({"sample_id": sample_id_str, "st": st, **pop_labels})
        print(
            f"sample={sample_id_str}, 总事件数={num_events}, 出错事件数={errored}, "
            f"注入事件数={injected_weak}"
        )

    return {"written_rows": written_rows, "skipped_rows": skipped_rows}


def generate_samples(max_workers=8):
    csv_path = os.path.join(output_base_dir, "train_idx.csv")
    skip_csv_path = os.path.join(skip_base_dir, "skipped_frames.csv")
    skip_txt_path = os.path.join(skip_base_dir, "skipped_h1_files.txt")

    config = {
        "samples_per_pop": 5,
        "duration": 2048,
        "sampling_frequency": sampling_frequency,
        "minimum_frequency": 10,
        "reference_frequency": 25.0,
    }

    for ifo_name in ["H1", "L1"]:
        os.makedirs(os.path.join(output_base_dir, ifo_name), exist_ok=True)
    os.makedirs(skip_base_dir, exist_ok=True)

    print(f"启动多进程处理，最大进程数: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_population, i, input_dir, output_base_dir, config): i
            for i in range(start, num_populations)
        }

        for future in as_completed(futures):
            pop_idx = futures[future]
            try:
                pop_result = future.result()
                written_rows = pop_result["written_rows"]
                skipped_rows = pop_result["skipped_rows"]

                if written_rows:
                    df_tmp = pd.DataFrame(written_rows)
                    if not os.path.exists(csv_path):
                        df_tmp.to_csv(csv_path, index=False, mode="w")
                    else:
                        df_tmp.to_csv(csv_path, index=False, mode="a", header=False)

                if skipped_rows:
                    df_skip = pd.DataFrame(skipped_rows)
                    if not os.path.exists(skip_csv_path):
                        df_skip.to_csv(skip_csv_path, index=False, mode="w")
                    else:
                        df_skip.to_csv(skip_csv_path, index=False, mode="a", header=False)

                    with open(skip_txt_path, "a", encoding="utf-8") as f:
                        for row in skipped_rows:
                            f.write(f"{row['h1_file']}\n")

                if written_rows or skipped_rows:
                    print(
                        f"超参数组 {pop_idx:05d} 完成: 写入 frame={len(written_rows)}, "
                        f"跳过 frame={len(skipped_rows)}"
                    )
            except Exception as e:
                print(f"超参数组 {pop_idx:05d} 运行出错: {e}")


if __name__ == "__main__":
    generate_samples(max_workers=1)
