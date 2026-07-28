import os
import logging

import bilby
import gwpy
import gwpy.timeseries as ts
import numpy as np
import pandas as pd


bilby.core.utils.setup_logger(log_level=logging.WARNING)
SNR_THRESH = 12.0
start = 0
temp_idx = 0
sampling_frequency = 512
version = "fixed_one_test_v100"
num_populations = 1
all_jobs = 4  # frame count + 1
input_dir = f"./parameter_sampling_test/{version}"
output_base_dir = f"./training_set_test/{version}/training_set0"


def process_population(pop_idx, input_dir, output_base_dir, config):
    duration = config["duration"]
    sampling_frequency_cfg = config["sampling_frequency"]
    minimum_frequency = config["minimum_frequency"]
    reference_frequency = config["reference_frequency"]

    param_file = os.path.join(input_dir, f"CBC_params_example{pop_idx:05d}.npz")
    if not os.path.exists(param_file):
        return []

    results = []
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

        def estimation_t0(tc, m1, m2):
            mtsun_si = 4.925491025543576e-06
            f0 = minimum_frequency - 1
            mc = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
            c = 5 ** (3.0 / 8) / (8 * np.pi)
            return tc - (f0 / c) ** (-8.0 / 3) * (mc * mtsun_si) ** (-5.0 / 3)

        m1_det_logic = m1_raw * (1 + z_raw)
        m2_det_logic = m2_raw * (1 + z_raw)
        t0_all = estimation_t0(tc_all_raw, m1_det_logic, m2_det_logic)
        list_cbc_mask = (tc_all_raw > st) * (t0_all < et)

        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        for ifo in interferometers:
            ifo.minimum_frequency = minimum_frequency
            ifo.sampling_frequency = sampling_frequency_cfg
            ifo.duration = duration

        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency_cfg, duration=duration, start_time=st
        )

        waveform_arguments = {
            "waveform_approximant": "IMRPhenomXAS",
            "reference_frequency": reference_frequency,
            "minimum_frequency": minimum_frequency,
        }

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
        skipped_strong = 0
        injected_weak = 0
        errored = 0

        for event in range(num_events):
            tc = tc_all[event]
            m1 = m1_all[event]
            m2 = m2_all[event]
            dL = dL_all[event]
            z = z_all[event]
            ra = ra_all[event]
            dec = dec_all[event]
            psi = psi_all[event]
            phi_c = phi_c_all[event]
            iota = iota_all[event]
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
                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    waveform_arguments=waveform_arguments,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                )

                tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
                plus = tds["plus"]
                cross = tds["cross"]

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
                    plus = ptemp
                    cross = ctemp

                if tc >= et:
                    idx = int(abs(tc - et) / dt)
                    nidx = plus.size - idx
                    ptemp = np.zeros_like(plus)
                    ctemp = np.zeros_like(cross)
                    ptemp[:nidx] = plus[:nidx]
                    ctemp[:nidx] = cross[:nidx]
                    plus = ptemp
                    cross = ctemp

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
                    raise_error=False,
                )

                for ifo_obj in interferometers:
                    idx_cut = int((tc - st) * sampling_frequency_cfg)
                    if idx_cut < len(ifo_obj.time_domain_strain):
                        ifo_obj.time_domain_strain[idx_cut:] = 0

            except Exception:
                errored += 1
                continue

        print(
            f"total events: {num_events}, errored: {errored}, "
            f"injected weak events: {injected_weak}, skipped strong events: {skipped_strong}"
        )
        for ifo in interferometers:
            new_channel_names = ifo.name + ":TEST_INJ"
            inj_channel = gwpy.detector.Channel(new_channel_names)
            injected_ts = ts.TimeSeries(
                ifo.time_domain_strain,
                times=ifo.time_array,
                name=new_channel_names,
                channel=inj_channel,
                dtype=float,
            )
            full_path = os.path.join(output_base_dir, ifo.name, f"{ifo.name}-{sample_id_str}.gwf")
            injected_ts.write(full_path)

        results.append(
            {
                "sample_id": sample_id_str,
                "st": st,
                **pop_labels,
            }
        )

    return results


def generate_samples():
    csv_path = os.path.join(output_base_dir, "train_idx.csv")

    config = {
        "samples_per_pop": 5,
        "duration": 2048,
        "sampling_frequency": sampling_frequency,
        "minimum_frequency": 10,
        "reference_frequency": 25.0,
    }

    for ifo_name in ["H1", "L1"]:
        os.makedirs(os.path.join(output_base_dir, ifo_name), exist_ok=True)

    all_results = []
    for pop_idx in range(start, num_populations):
        pop_results = process_population(pop_idx, input_dir, output_base_dir, config)
        if pop_results:
            all_results.extend(pop_results)
            print(f"population {pop_idx:05d} done")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    generate_samples()
