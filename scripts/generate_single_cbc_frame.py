import logging
import os
from pathlib import Path

import bilby
import gwpy
import gwpy.timeseries as ts
import numpy as np


bilby.core.utils.setup_logger(log_level=logging.WARNING)

# Dataset / sample selection
version = "v63"
pop_idx = 30632
sample_idx = 0
job_number = 1
detectors_to_write = ["L1"]  # e.g. ["L1"] or ["H1", "L1"]

# Waveform / frame config
duration = 2048
#sampling_frequency = 64
sampling_frequency = 32
minimum_frequency = 10
reference_frequency = 25.0
snr_thresh = 12.0


def estimation_t0(tc, m1, m2):
    mtsun_si = 4.925491025543576e-06
    f0 = minimum_frequency - 1
    mc = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    c_const = 5 ** (3.0 / 8.0) / (8 * np.pi)
    return tc - (f0 / c_const) ** (-8.0 / 3.0) * (mc * mtsun_si) ** (-5.0 / 3.0)


def load_population_file(input_dir: Path, current_pop_idx: int):
    param_file = input_dir / f"CBC_params_example{current_pop_idx:05d}.npz"
    if not param_file.exists():
        raise FileNotFoundError(f"Population file not found: {param_file}")
    return np.load(param_file)


def build_interferometers(frame_start: float):
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in interferometers:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=frame_start,
    )
    return interferometers


def inject_population_events(list_cbc, frame_start: float, frame_end: float):
    waveform_arguments = {
        "waveform_approximant": "IMRPhenomXAS",
        "reference_frequency": reference_frequency,
        "minimum_frequency": minimum_frequency,
    }

    tc_all_raw = list_cbc["tc"]
    m1_raw = list_cbc["m1"]
    m2_raw = list_cbc["m2"]
    z_raw = list_cbc["z"]

    m1_det_logic = m1_raw * (1 + z_raw)
    m2_det_logic = m2_raw * (1 + z_raw)
    t0_all = estimation_t0(tc_all_raw, m1_det_logic, m2_det_logic)
    list_cbc_mask = (tc_all_raw > frame_start) * (t0_all < frame_end)

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

    interferometers = build_interferometers(frame_start)
    skipped_strong = 0
    injected_weak = 0
    errored = 0

    for event in range(len(tc_all)):
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
                sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments=waveform_arguments,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            )

            tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
            plus = np.roll(tds["plus"], -512)
            cross = np.roll(tds["cross"], -512)
            time = waveform_generator.time_array
            dt = time[1] - time[0]

            if tc <= frame_end:
                event_dur = int((tc - frame_start) / dt)
                ptemp, ctemp = np.zeros_like(plus), np.zeros_like(cross)
                ptemp[-event_dur:] = plus[-event_dur:]
                ctemp[-event_dur:] = cross[-event_dur:]
                plus, cross = ptemp, ctemp

            if tc >= frame_end:
                idx = int(abs(tc - frame_end) / dt)
                nidx = plus.size - idx
                ptemp, ctemp = np.zeros_like(plus), np.zeros_like(cross)
                ptemp[:nidx] = plus[:nidx]
                ctemp[:nidx] = cross[:nidx]
                plus, cross = ptemp, ctemp

            fds = dict(plus=np.fft.rfft(plus) * dt, cross=np.fft.rfft(cross) * dt)
            snr2_net = 0.0
            for ifo in interferometers:
                det_signal_fd = ifo.get_detector_response(fds, injection_parameters)
                snr2_net += ifo.optimal_snr_squared(det_signal_fd)

            snr_net = float(np.sqrt(snr2_net))
            if snr_net >= snr_thresh:
                skipped_strong += 1
                continue

            injected_weak += 1
            interferometers.inject_signal(
                injection_parameters,
                injection_polarizations=fds,
                raise_error=False,
            )

            for ifo_obj in interferometers:
                idx_cut = int((tc - frame_start) * sampling_frequency)
                if idx_cut < len(ifo_obj.time_domain_strain):
                    ifo_obj.time_domain_strain[idx_cut:] = 0

        except Exception as exc:  # noqa: PERF203
            errored += 1
            print(f"Event {event} failed: {exc}")

    print(
        f"Total events in frame: {len(tc_all)}, injected weak: {injected_weak}, "
        f"skipped strong: {skipped_strong}, errored: {errored}"
    )
    return interferometers


def write_outputs(interferometers, output_dir: Path, sample_id_str: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ifo in interferometers:
        if ifo.name not in detectors_to_write:
            continue
        new_channel_name = ifo.name + ":TEST_INJ"
        inj_channel = gwpy.detector.Channel(new_channel_name)
        injected_ts = ts.TimeSeries(
            ifo.time_domain_strain,
            times=ifo.time_array,
            name=new_channel_name,
            channel=inj_channel,
            dtype=float,
        )
        full_path = output_dir / ifo.name / f"{ifo.name}-{sample_id_str}.gwf"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        injected_ts.write(full_path)
        print(f"Saved {full_path}")


def main():
    scripts_dir = Path(__file__).resolve().parent
    input_dir = scripts_dir / "parameter_sampling_train" / version
    output_base_dir = scripts_dir / "training_set" / version / "training_set0"

    list_cbc = load_population_file(input_dir, pop_idx)

    frame_start = (job_number - 1) * duration
    frame_end = frame_start + duration
    sample_id_str = f"pop{pop_idx:05d}_sample{sample_idx:05d}"

    print(f"Generating sample: {sample_id_str}")
    print(f"Version: {version}")
    print(f"Frame window: [{frame_start}, {frame_end}]")
    print(f"Writing detectors: {detectors_to_write}")

    interferometers = inject_population_events(list_cbc, frame_start, frame_end)
    write_outputs(interferometers, output_base_dir, sample_id_str)


if __name__ == "__main__":
    main()
