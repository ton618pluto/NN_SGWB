import os
import sys

import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18
from jax import jit
from jax.numpy import pi

sys.path.append("./gwfast/")
from gwfast.population import POPmodels
from gwfast.population.popdistributions import massdistribution, ratedistribution, spindistribution
from utils import *  # noqa: F401,F403


SECONDS_PER_YEAR = 365.25 * 24 * 3600
FRAME_DURATION_SECONDS = 2048
Z_MAX = 10.0

# version = "fixed_one_test_v61"
# version = "fixed_one_test_v64"
# version = "fixed_one_test_v80"
# version = "fixed_one_test_v70"
# version = "fixed_one_test_v71"
version = "fixed_one_test_v100"
output_path = f"./parameter_sampling_test/{version}"
data_hyper = np.load(f"./joint_hyperparams_test/{version}/joint_hyperparams_test_{version}.npz")
start = 0
hyper_number = 1
num_frames = 1
max_attempts = 200
REFERENCE_TRAIN_FILE_INDEX = 3  # this will be changed, control randomness


@jit
def psi_z(z, gamma, kappa, z_peak):
    const = 1 / (1 + (1 / (1 + z_peak)) ** kappa)
    return const * (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)


@jit
def dN(z, gamma, kappa, z_peak, R_0, dVdz_grid):
    return R_0 * psi_z(z, gamma, kappa, z_peak) * dVdz_grid * 4 * pi / (1 + z)


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
    z_grid_np = np.linspace(0.0, Z_MAX, z_grid_n, dtype=np.float32)
    dVdz_grid_np = Planck18.differential_comoving_volume(z_grid_np).value / 1e9
    z_grid = jnp.asarray(z_grid_np)
    dVdz_grid = jnp.asarray(dVdz_grid_np)

    t_obs_year = total_duration_seconds / SECONDS_PER_YEAR
    _, z_w = trapz_weights(z_grid_n, 0.0, Z_MAX)
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
        priorlims_parameters={"z": (0.0, Z_MAX)},
    )

    full_pop = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
        mass_function=mass_dist,
        spin_function=spin_dist,
        rate_function=redshift_dist,
    )

    if n_inj > 0:
        injpars = full_pop.sample_population(n_inj)
    else:
        injpars = {"m1": np.array([]), "m2": np.array([]), "z": np.array([])}

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
        "num_events": int(n_inj),
    }


def run_parameter_sampling_seconds_no_snr_cut(
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

    accepted = None
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

        if sampled["num_events"] > 0:
            accepted = sampled
            accepted["attempt_index"] = attempt_idx
            accepted["attempts_used"] = attempt_idx + 1
            accepted["actual_alpha_z"] = current_alpha_z
            accepted["alpha_z_resampled"] = not np.isclose(current_alpha_z, alpha_z)
            break

    if accepted is None:
        raise RuntimeError(
            f"sample_id={sample_id}: failed to sample a non-empty segment within {max_attempts} attempts"
        )

    filename = os.path.join(output_dir, f"CBC_params_example{sample_id:05d}.npz")
    np.savez(
        filename,
        tc=accepted["tc"],
        m1=accepted["m1"],
        m2=accepted["m2"],
        dL=accepted["dL"],
        z=accepted["z"],
        ra=accepted["ra"],
        dec=accepted["dec"],
        psi=accepted["psi"],
        phi_c=accepted["phi_c"],
        iota=accepted["iota"],
        num_events=accepted["num_events"],
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
        z_max=Z_MAX,
    )
    return {
        "filename": filename,
        "num_events": accepted["num_events"],
        "attempts_used": accepted["attempts_used"],
    }


if __name__ == "__main__":
    total_duration_seconds = num_frames * FRAME_DURATION_SECONDS

    print(f"Frame duration: {FRAME_DURATION_SECONDS} s")
    print(f"Number of frames per segment: {num_frames}")
    print(f"Total duration per segment: {total_duration_seconds} s")
    print("SNR filtering: disabled")
    print()

    for i in range(start, hyper_number):
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

        result = run_parameter_sampling_seconds_no_snr_cut(
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
            seed=REFERENCE_TRAIN_FILE_INDEX,
            max_attempts=max_attempts,
            alpha_z_bounds=alpha_z_bounds,
        )
        print(
            f"sample_id={i} done, saved: {result['filename']}, "
            f"num_events={result['num_events']}, "
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
