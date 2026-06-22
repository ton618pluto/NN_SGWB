import os
import sys
from pathlib import Path

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
#version = "v6"
#version = "v11"
#version = "v12"
#version = "v20"
#version = "v21"
#version = "v22"
#version = "v23"

#version = "v44"
version = "v45"
#version = "v51"

#version = "v0000"
#hyper_number = 1
#hyper_number = 5000
#hyper_number = 100
#hyper_number = 100000
hyper_number = 200000
#hyper_number = 200000
#num_frames = 23
num_frames = 1
#num_frames = 20
output_path = f"./parameter_sampling_train/{version}"
data_hyper = np.load(f"./joint_hyperparams_train/{version}/joint_hyperparams_train_{version}.npz")
keys = data_hyper.files



@jit
def psi_z(z, gamma, kappa, z_peak):
    const = 1 / (1 + (1 / (1 + z_peak)) ** kappa)
    return const * (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)


@jit
def dN(z, gamma, kappa, z_peak, R_0, dVdz_grid):
    return R_0 * psi_z(z, gamma, kappa, z_peak) * dVdz_grid * 4 * pi / (1 + z)


def run_parameter_sampling_seconds(
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
):
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    z_grid_n = 2048
    zmax = 20.0
    z_grid_np = np.linspace(0.0, zmax, z_grid_n, dtype=np.float32)
    dVdz_grid_np = Planck18.differential_comoving_volume(z_grid_np).value / 1e9
    z_grid = jnp.asarray(z_grid_np)
    dVdz_grid = jnp.asarray(dVdz_grid_np)

    T_obs_year = total_duration_seconds / SECONDS_PER_YEAR
    _, z_w = trapz_weights(z_grid_n, 0.0, zmax)
    kappa = alpha_z + beta_z
    expected_N = jnp.dot(dN(z_grid, alpha_z, kappa, zp, R0, dVdz_grid), z_w) * T_obs_year
    Ninj = np.random.poisson(float(expected_N))

    if Ninj == 0:
        print('warning:events_num=0')
        return 0

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
        priorlims_parameters={"z": (0.0, 20.0)},
    )

    full_pop = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
        mass_function=mass_dist,
        spin_function=spin_dist,
        rate_function=redshift_dist,
    )

    injpars = full_pop.sample_population(Ninj)

    tc = np.sort(np.random.uniform(0.0, total_duration_seconds, size=Ninj))
    ra = np.random.uniform(0, 2 * np.pi, Ninj)
    psi = np.random.uniform(0, 2 * np.pi, Ninj)
    phi_c = np.random.uniform(0, 2 * np.pi, Ninj)
    iota = np.arccos(np.random.uniform(-1, 1, Ninj))
    dec = np.arcsin(np.random.uniform(-1, 1, Ninj))
    dL = np.asarray(Planck18.luminosity_distance(injpars["z"]))

    filename = os.path.join(output_dir, f"CBC_params_example{sample_id:05d}.npz")
    np.savez(
        filename,
        tc=tc,
        m1=injpars["m1"],
        m2=injpars["m2"],
        dL=dL,
        z=injpars["z"],
        ra=ra,
        dec=dec,
        psi=psi,
        phi_c=phi_c,
        iota=iota,
        true_R0=R0,
        true_alpha_z=alpha_z,
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
    )
    return Ninj


if __name__ == "__main__":
    total_duration_seconds = num_frames * FRAME_DURATION_SECONDS
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Frame duration: {FRAME_DURATION_SECONDS} s")
    print(f"Number of frames: {num_frames}")
    print(f"Total duration: {total_duration_seconds} s")

    for i in range(hyper_number):
        output_file = output_dir / f"CBC_params_example{i:05d}.npz"
        if output_file.exists():
            print(f"sample_id={i} skipped, file already exists: {output_file.name}")
            continue
        else:
            print(f"loss{i}")

        np.random.seed(i)
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

        print(
            f"sample_id={i} hyperparams: "
            f"zp={float(r_zp):.6f}, "
            f"alpha_z={float(r_alpha_z):.6f}, "
            f"beta_z={float(r_beta_z):.6f}"
        )

        num_events = run_parameter_sampling_seconds(
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
        )
        print(f"sample_id={i} done, generated events: {num_events}")
