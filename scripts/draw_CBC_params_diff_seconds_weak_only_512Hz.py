import os
import sys
from contextlib import contextmanager
import logging

import bilby
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18
from jax import jit
from jax.numpy import pi

sys.path.append("./gwfast/")
from gwfast.population import POPmodels
from gwfast.population.popdistributions import massdistribution, ratedistribution, spindistribution
from utils import *  # noqa: F401,F403


bilby.core.utils.setup_logger(log_level="warning")
logging.getLogger("bilby").setLevel(logging.WARNING)
logging.getLogger("bilby.gw").setLevel(logging.WARNING)


SECONDS_PER_YEAR = 365.25 * 24 * 3600
FRAME_DURATION_SECONDS = 2048
SNR_THRESH = 12.0
SAMPLING_FREQUENCY = 512 # 32 or 512 or 2048
MINIMUM_FREQUENCY = 10
REFERENCE_FREQUENCY = 25.0

#version = "v60"
#version = "v61"
#version = "v64"
version = "v100"
output_path = f"./parameter_sampling_train/{version}"
data_hyper = np.load(f"./joint_hyperparams_train/{version}/joint_hyperparams_train_{version}.npz")
#hyper_number = 50000
start = 0
#hyper_number = 100000
hyper_number = 100000
num_frames = 1
max_attempts = 200


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


@jit
def psi_z(z, gamma, kappa, z_peak):
    const = 1 / (1 + (1 / (1 + z_peak)) ** kappa)
    return const * (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)


@jit
def dN(z, gamma, kappa, z_peak, R_0, dVdz_grid):
    return R_0 * psi_z(z, gamma, kappa, z_peak) * dVdz_grid * 4 * pi / (1 + z)


def build_event_polarizations(waveform_generator, injection_parameters, duration_seconds):
    with suppress_c_stderr():
        tds = waveform_generator.time_domain_strain(parameters=injection_parameters)
    plus, cross = tds["plus"], tds["cross"]

    plus = np.roll(plus, -512)
    cross = np.roll(cross, -512)
    time = waveform_generator.time_array
    dt = time[1] - time[0]
    tc = injection_parameters["geocent_time"]
    st = 0.0
    et = duration_seconds

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


def compute_network_snrs(
    tc,
    m1,
    m2,
    dL,
    z,
    ra,
    dec,
    psi,
    phi_c,
    iota,
    duration_seconds,
):
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in interferometers:
        ifo.minimum_frequency = MINIMUM_FREQUENCY
        ifo.sampling_frequency = SAMPLING_FREQUENCY
        ifo.duration = duration_seconds

    waveform_arguments = {
        "waveform_approximant": "IMRPhenomXAS",
        "reference_frequency": REFERENCE_FREQUENCY,
        "minimum_frequency": MINIMUM_FREQUENCY,
    }
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration_seconds,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    )

    snr_net = np.full_like(tc, fill_value=np.nan, dtype=np.float32)
    errored_event_mask = np.zeros_like(tc, dtype=bool)
    for idx in range(len(tc)):
        try:
            m1_det = m1[idx] * (1 + z[idx])
            m2_det = m2[idx] * (1 + z[idx])
            injection_parameters = dict(
                mass_1=m1_det,
                mass_2=m2_det,
                a_1=0.0,
                a_2=0.0,
                luminosity_distance=dL[idx],
                psi=psi[idx],
                phase=phi_c[idx],
                geocent_time=tc[idx],
                ra=ra[idx],
                dec=dec[idx],
                theta_jn=iota[idx],
                tilt_1=0.0,
                tilt_2=0.0,
                phi_12=0.0,
                phi_jl=0.0,
            )
            fds = build_event_polarizations(waveform_generator, injection_parameters, duration_seconds)

            snr2_net = 0.0
            for ifo in interferometers:
                det_signal_fd = ifo.get_detector_response(fds, injection_parameters)
                snr2_net += np.real(ifo.optimal_snr_squared(det_signal_fd))
            snr_net[idx] = np.sqrt(max(float(snr2_net), 0.0))
        except Exception:
            errored_event_mask[idx] = True
            continue

    return snr_net, errored_event_mask


def sample_one_segment(
    total_duration_seconds,
    R0,
    alpha_z,
    beta_z,
    zp,
    alpha_m,
    m_max,
    delta_m,
    m_min,
    lambda_peak,
    mu_m,
    sigma_m,
    beta_q,
):
    z_grid_n = 2048
    zmax = 20.0
    z_grid_np = np.linspace(0.0, zmax, z_grid_n, dtype=np.float32)
    dVdz_grid_np = Planck18.differential_comoving_volume(z_grid_np).value / 1e9
    z_grid = jnp.asarray(z_grid_np)
    dVdz_grid = jnp.asarray(dVdz_grid_np)

    t_obs_year = total_duration_seconds / SECONDS_PER_YEAR
    _, z_w = trapz_weights(z_grid_n, 0.0, zmax)
    kappa = alpha_z + beta_z
    expected_n = jnp.dot(dN(z_grid, alpha_z, kappa, zp, R0, dVdz_grid), z_w) * t_obs_year
    n_inj = np.random.poisson(float(expected_n))

    mass_dist = massdistribution.PowerLawPlusPeak_MassDistribution(
        hyperparameters={
            "alpha_m": alpha_m,
            "beta_q": beta_q,
            "m_min": m_min,
            "m_max": m_max,
            "lambda_peak": lambda_peak,
            "delta_m": delta_m,
            "mu_m": mu_m,
            "sigma_m": sigma_m,
        },
        priorlims_parameters={"m1_src": (m_min, m_max), "m2_src": (m_min, m_max)},
    )

    spin_dist = spindistribution.SameFlatNonPrecessing_SpinDistribution(
        hyperparameters={"abs_chi": 0.99},
        priorlims_parameters={"chi1z": (-1.0, 1.0), "chi2z": (-1.0, 1.0)},
    )

    redshift_dist = ratedistribution.MadauDickinson_RateDistribution(
        hyperparameters={"alpha_z": alpha_z, "beta_z": beta_z, "zp": zp},
#        priorlims_parameters={"z": (0.0, 20.0)},
        priorlims_parameters={"z": (0.0, 10.0)},
    )

    full_pop = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
        mass_function=mass_dist,
        spin_function=spin_dist,
        rate_function=redshift_dist,
    )

    injpars = full_pop.sample_population(n_inj) if n_inj > 0 else {"m1": np.array([]), "m2": np.array([]), "z": np.array([])}

    tc = np.sort(np.random.uniform(0.0, total_duration_seconds, size=n_inj)).astype(np.float32)
    ra = np.random.uniform(0, 2 * np.pi, n_inj).astype(np.float32)
    psi = np.random.uniform(0, 2 * np.pi, n_inj).astype(np.float32)
    phi_c = np.random.uniform(0, 2 * np.pi, n_inj).astype(np.float32)
    iota = np.arccos(np.random.uniform(-1, 1, n_inj)).astype(np.float32)
    dec = np.arcsin(np.random.uniform(-1, 1, n_inj)).astype(np.float32)
    if n_inj > 0:
        dL = np.asarray(Planck18.luminosity_distance(injpars["z"]), dtype=np.float32)
    else:
        dL = np.array([], dtype=np.float32)

    if n_inj > 0:
        snr_net, errored_event_mask = compute_network_snrs(
            tc, injpars["m1"], injpars["m2"], dL, injpars["z"], ra, dec, psi, phi_c, iota, total_duration_seconds
        )
    else:
        snr_net = np.array([], dtype=np.float32)
        errored_event_mask = np.array([], dtype=bool)

    valid_event_mask = ~errored_event_mask
    strong_event_mask = np.zeros_like(valid_event_mask, dtype=bool)
    strong_event_mask[valid_event_mask] = snr_net[valid_event_mask] >= SNR_THRESH
    kept_event_mask = valid_event_mask & (~strong_event_mask)
    return {
        "tc": tc,
        "m1": np.asarray(injpars["m1"], dtype=np.float32),
        "m2": np.asarray(injpars["m2"], dtype=np.float32),
        "dL": dL,
        "z": np.asarray(injpars["z"], dtype=np.float32),
        "ra": ra,
        "dec": dec,
        "psi": psi,
        "phi_c": phi_c,
        "iota": iota,
        "snr_net": snr_net,
        "valid_event_mask": valid_event_mask,
        "errored_event_mask": errored_event_mask,
        "strong_event_mask": strong_event_mask,
        "kept_event_mask": kept_event_mask,
        "num_events": int(n_inj),
        "num_valid_events": int(np.sum(valid_event_mask)),
        "num_errored_events": int(np.sum(errored_event_mask)),
        "num_strong_events": int(np.sum(strong_event_mask)),
        "num_kept_events": int(np.sum(kept_event_mask)),
    }


def run_parameter_sampling_seconds_weak_only(
    output_dir="./parameter_sampling",
    sample_id=0,
    total_duration_seconds=FRAME_DURATION_SECONDS,
    R0=320.0,
    alpha_z=2.7,
    beta_z=2.9,
    zp=1.9,
    alpha_m=3.4,
    m_max=86.85,
    delta_m=4.83,
    m_min=5.08,
    lambda_peak=0.04,
    mu_m=33.73,
    sigma_m=3.56,
    beta_q=1.08,
    seed=3333,
    max_attempts=200,
    alpha_z_bounds=None,
):
    os.makedirs(output_dir, exist_ok=True)

    for attempt_idx in range(max_attempts):
        if seed is not None:
            np.random.seed(seed + attempt_idx)

        current_alpha_z = float(alpha_z)
        while True:
            sampled = sample_one_segment(
                total_duration_seconds,
                R0,
                current_alpha_z,
                beta_z,
                zp,
                alpha_m,
                m_max,
                delta_m,
                m_min,
                lambda_peak,
                mu_m,
                sigma_m,
                beta_q,
            )
            if sampled["num_events"] > 0:
                break
            if alpha_z_bounds is None:
                break
            alpha_lo, alpha_hi = map(float, alpha_z_bounds)
            if not alpha_hi > alpha_lo:
                break
            current_alpha_z = float(np.random.uniform(alpha_lo, alpha_hi))

        accepted = sampled
        accepted["attempt_index"] = attempt_idx
        accepted["attempts_used"] = attempt_idx + 1
        accepted["actual_alpha_z"] = current_alpha_z
        accepted["alpha_z_resampled"] = not np.isclose(current_alpha_z, alpha_z)
        break

    if accepted is None:
        raise RuntimeError(f"sample_id={sample_id}: failed to sample a segment within {max_attempts} attempts")

    kept = accepted["kept_event_mask"]

    filename = os.path.join(output_dir, f"CBC_params_example{sample_id:05d}.npz")
    np.savez(
        filename,
        tc=accepted["tc"][kept],
        m1=accepted["m1"][kept],
        m2=accepted["m2"][kept],
        dL=accepted["dL"][kept],
        z=accepted["z"][kept],
        ra=accepted["ra"][kept],
        dec=accepted["dec"][kept],
        psi=accepted["psi"][kept],
        phi_c=accepted["phi_c"][kept],
        iota=accepted["iota"][kept],
        snr_net=accepted["snr_net"][kept],
        valid_event_mask=accepted["valid_event_mask"],
        errored_event_mask=accepted["errored_event_mask"],
        strong_event_mask=accepted["strong_event_mask"],
        kept_event_mask=accepted["kept_event_mask"],
        num_events=accepted["num_events"],
        num_valid_events=accepted["num_valid_events"],
        num_errored_events=accepted["num_errored_events"],
        num_strong_events=accepted["num_strong_events"],
        num_kept_events=accepted["num_kept_events"],
        attempt_index=accepted["attempt_index"],
        attempts_used=accepted["attempts_used"],
        true_R0=R0,
        true_alpha_z=accepted.get("actual_alpha_z", alpha_z),
        true_beta_z=beta_z,
        true_zp=zp,
        true_alpha_m=alpha_m,
        true_m_max=m_max,
        true_delta_m=delta_m,
        true_m_min=m_min,
        true_lambda_peak=lambda_peak,
        true_mu_m=mu_m,
        true_sigma_m=sigma_m,
        true_beta_q=beta_q,
        total_duration_seconds=total_duration_seconds,
        frame_duration_seconds=FRAME_DURATION_SECONDS,
        snr_threshold=SNR_THRESH,
    )
    return {
        "filename": filename,
        "num_events": accepted["num_events"],
        "num_valid_events": accepted["num_valid_events"],
        "num_errored_events": accepted["num_errored_events"],
        "num_strong_events": accepted["num_strong_events"],
        "num_kept_events": accepted["num_kept_events"],
        "attempts_used": accepted["attempts_used"],
    }


if __name__ == "__main__":
    total_duration_seconds = num_frames * FRAME_DURATION_SECONDS

    print(f"Frame duration: {FRAME_DURATION_SECONDS} s")
    print(f"Number of frames per segment: {num_frames}")
    print(f"Total duration per segment: {total_duration_seconds} s")
    print(f"SNR threshold: {SNR_THRESH}")
    print()

    for i in range(start,hyper_number):
        target_file = os.path.join(output_path, f"CBC_params_example{i:05d}.npz")
        if os.path.exists(target_file):
            print(f"sample_id={i} skipped, existing file: {target_file}")
            print()
            continue

        r_zp = data_hyper["zp"][i]
        r_alpha_z = data_hyper["alpha_z"][i]
        r_beta_z = data_hyper["beta_z"][i]
        r_alpha_m = data_hyper["alpha_m"][i]
        r_m_max = data_hyper["m_max"][i]
        r_delta_m = data_hyper["delta_m"][i]
        r_m_min = data_hyper["m_min"][i]
        r_lambda_peak = data_hyper["lambda_peak"][i]
        r_mu_m = data_hyper["mu_m"][i]
        r_sigma_m = data_hyper["sigma_m"][i]
        r_beta_q = data_hyper["beta_q"][i]
        alpha_z_bounds = (
            float(np.min(data_hyper["alpha_z"])),
            float(np.max(data_hyper["alpha_z"])),
        )

        result = run_parameter_sampling_seconds_weak_only(
            sample_id=i,
            output_dir=output_path,
            total_duration_seconds=total_duration_seconds,
            zp=r_zp,
            alpha_z=r_alpha_z,
            beta_z=r_beta_z,
            alpha_m=r_alpha_m,
            m_max=r_m_max,
            delta_m=r_delta_m,
            m_min=r_m_min,
            lambda_peak=r_lambda_peak,
            mu_m=r_mu_m,
            sigma_m=r_sigma_m,
            beta_q=r_beta_q,
            seed=i,
            max_attempts=max_attempts,
            alpha_z_bounds=alpha_z_bounds,
        )
        print(
            f"sample_id={i} done, saved: {result['filename']}, "
            f"num_events={result['num_events']}, "
            f"valid_events={result['num_valid_events']}, "
            f"errored_events={result['num_errored_events']}, "
            f"strong_events={result['num_strong_events']}, "
            f"kept_events={result['num_kept_events']}, "
            f"attempts_used={result['attempts_used']}"
        )
        saved = np.load(result["filename"])
        saved_alpha_z = float(saved["true_alpha_z"])
        if not np.isclose(saved_alpha_z, r_alpha_z):
            print(
                f"sample_id={i} alpha_z resampled: "
                f"original={r_alpha_z:.10f}, saved={saved_alpha_z:.10f}"
            )
        print()
