import os
import bilby
import logging
from contextlib import contextmanager
import pandas as pd
import gwpy.timeseries as ts
import gwpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


bilby.core.utils.setup_logger(log_level="warning")
logging.getLogger("bilby").setLevel(logging.WARNING)
logging.getLogger("bilby.gw").setLevel(logging.WARNING)


@contextmanager
def suppress_c_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


start = 0
temp_idx = 0
#sampling_frequency = 64 # this is remember
sampling_frequency = 32 # this is remember
#version = "fixed_one_test_v61"
#version = "fixed_one_test_v64"
version = "fixed_one_test_v80"
num_populations = 1
all_jobs = 11 # num of frame+1
input_dir = f"./parameter_sampling_test/{version}"
output_base_dir = f"./training_set_test/{version}/training_set0"


def process_population(pop_idx, input_dir, output_base_dir, config):
    duration = config["duration"]
    sampling_frequency = config["sampling_frequency"]
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

    for s_idx, jobNumber in enumerate(random_jobs):
        sample_id_str = f"pop{pop_idx:05d}_sample{s_idx:05d}"

        h1_path = os.path.join(output_base_dir, "H1", f"H1-{sample_id_str}.gwf")
        l1_path = os.path.join(output_base_dir, "L1", f"L1-{sample_id_str}.gwf")
        if os.path.exists(h1_path) and os.path.exists(l1_path):
            continue

        st = (jobNumber - 1) * duration
        et = st + duration

        tc_all_raw = list_cbc["tc"]
        m1_raw = list_cbc["m1"]
        m2_raw = list_cbc["m2"]
        z_raw = list_cbc["z"]

        def estimation_t0(tc, m1, m2):
            MTSUN_SI = 4.925491025543576e-06
            f0 = minimum_frequency - 1
            mc = (m1 * m2) ** (3.0 / 5) / (m1 + m2) ** (1.0 / 5)
            C = 5 ** (3.0 / 8) / (8 * np.pi)
            return tc - (f0 / C) ** (-8.0 / 3) * (mc * MTSUN_SI) ** (-5.0 / 3)

        m1_det_logic, m2_det_logic = m1_raw * (1 + z_raw), m2_raw * (1 + z_raw)
        t0_all = estimation_t0(tc_all_raw, m1_det_logic, m2_det_logic)
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
        injected_events = 0
        errored = 0
        for EVENT in range(num_events):
            tc, m1, m2 = tc_all[EVENT], m1_all[EVENT], m2_all[EVENT]
            dL, z, ra, dec = dL_all[EVENT], z_all[EVENT], ra_all[EVENT], dec_all[EVENT]
            psi, phi_c, iota = psi_all[EVENT], phi_c_all[EVENT], iota_all[EVENT]
            m1_det, m2_det = m1 * (1 + z), m2 * (1 + z)

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
                    sampling_frequency=sampling_frequency,
                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    waveform_arguments=waveform_arguments,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                )

                with suppress_c_stderr():
                    tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
                plus, cross = tds["plus"], tds["cross"]

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
                injected_events += 1
                interferometers.inject_signal(
                    injection_parameters,
                    injection_polarizations=fds,
                    raise_error=False,
                )

                for ifo_obj in interferometers:
                    idx_cut = int((tc - st) * sampling_frequency)
                    if idx_cut < len(ifo_obj.time_domain_strain):
                        ifo_obj.time_domain_strain[idx_cut:] = 0

            except Exception:
                errored += 1
                continue

        print(f"num_events={num_events}, errored={errored}, injected_events={injected_events}")
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

        results.append({"sample_id": sample_id_str, "st": st, **pop_labels})

    return results


def generate_samples(max_workers=8):
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

    print(f"Start multiprocessing, max_workers={max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_population, i, input_dir, output_base_dir, config): i
            for i in range(start, num_populations)
        }

        for future in as_completed(futures):
            pop_idx = futures[future]
            try:
                pop_results = future.result()
                if pop_results:
                    df_tmp = pd.DataFrame(pop_results)
                    if not os.path.exists(csv_path):
                        df_tmp.to_csv(csv_path, index=False, mode="w")
                    else:
                        df_tmp.to_csv(csv_path, index=False, mode="a", header=False)
                    print(f"Population {pop_idx:05d} finished and written to CSV.")
            except Exception as e:
                print(f"Population {pop_idx:05d} failed: {e}")


if __name__ == "__main__":
    generate_samples(max_workers=3)
