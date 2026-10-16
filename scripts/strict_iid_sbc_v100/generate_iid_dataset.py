from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import bilby
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18
from bilby.gw.detector import PowerSpectralDensity
from jax import jit
from tqdm import tqdm

from config import (
    CONDITION_ON_NONEMPTY_POPULATION,
    FRAME_DURATION_SECONDS,
    FULL_LABEL_NAMES,
    GENERATION_LOG_PATH,
    H1_DIR,
    HYPERPARAMETER_PATH,
    L1_DIR,
    LOCAL_RATE_GPC3_YR,
    MAX_POPULATION_ATTEMPTS,
    MINIMUM_FREQUENCY_HZ,
    NETWORK_SNR_THRESHOLD,
    NOISE_SCALE,
    NUM_SBC_SAMPLES,
    O4_ASD_PATH,
    REFERENCE_FREQUENCY_HZ,
    SAMPLING_FREQUENCY_HZ,
    SCRIPTS_DIR,
    Z_MAX,
    noise_seed,
    signal_seed,
    validate_config,
)


if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gwfast.population import POPmodels  # noqa: E402
from gwfast.population.popdistributions import (  # noqa: E402
    massdistribution,
    ratedistribution,
    spindistribution,
)


bilby.core.utils.setup_logger(log_level="warning")
logging.getLogger("bilby").setLevel(logging.WARNING)
logging.getLogger("bilby.gw").setLevel(logging.WARNING)


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
LABEL_INDEX = {name: index for index, name in enumerate(FULL_LABEL_NAMES)}


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
def psi_z(z, alpha_z, kappa, zp):
    normalization = 1.0 / (1.0 + (1.0 / (1.0 + zp)) ** kappa)
    return normalization * (1.0 + z) ** alpha_z / (
        1.0 + ((1.0 + z) / (1.0 + zp)) ** kappa
    )


def _event_count_expectation(theta: dict[str, float]) -> float:
    z_grid = np.linspace(0.0, Z_MAX, 2048, dtype=np.float32)
    d_v_dz = Planck18.differential_comoving_volume(z_grid).value / 1e9
    kappa = theta["alpha_z"] + theta["beta_z"]
    rate = (
        LOCAL_RATE_GPC3_YR
        * np.asarray(psi_z(jnp.asarray(z_grid), theta["alpha_z"], kappa, theta["zp"]))
        * d_v_dz
        * 4.0
        * np.pi
        / (1.0 + z_grid)
    )
    observation_years = FRAME_DURATION_SECONDS / SECONDS_PER_YEAR
    return float(np.trapezoid(rate, z_grid) * observation_years)


def _sample_population_once(theta: dict[str, float], seed: int) -> dict[str, np.ndarray | int]:
    random.seed(seed)
    np.random.seed(seed)
    expected_count = _event_count_expectation(theta)
    event_count = int(np.random.poisson(expected_count))

    mass_dist = massdistribution.PowerLawPlusPeak_MassDistribution(
        hyperparameters={
            "alpha_m": theta["alpha_m"],
            "beta_q": theta["beta_q"],
            "m_min": theta["m_min"],
            "m_max": theta["m_max"],
            "lambda_peak": theta["lambda_peak"],
            "delta_m": theta["delta_m"],
            "mu_m": theta["mu_m"],
            "sigma_m": theta["sigma_m"],
        },
        priorlims_parameters={
            "m1_src": (theta["m_min"], theta["m_max"]),
            "m2_src": (theta["m_min"], theta["m_max"]),
        },
    )
    spin_dist = spindistribution.SameFlatNonPrecessing_SpinDistribution(
        hyperparameters={"abs_chi": 0.99},
        priorlims_parameters={"chi1z": (-1.0, 1.0), "chi2z": (-1.0, 1.0)},
    )
    redshift_dist = ratedistribution.MadauDickinson_RateDistribution(
        hyperparameters={
            "alpha_z": theta["alpha_z"],
            "beta_z": theta["beta_z"],
            "zp": theta["zp"],
        },
        priorlims_parameters={"z": (0.0, Z_MAX)},
    )
    full_population = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
        mass_function=mass_dist,
        spin_function=spin_dist,
        rate_function=redshift_dist,
    )

    if event_count:
        population = full_population.sample_population(event_count)
        m1 = np.asarray(population["m1"], dtype=np.float32)
        m2 = np.asarray(population["m2"], dtype=np.float32)
        redshift = np.asarray(population["z"], dtype=np.float32)
    else:
        m1 = np.empty(0, dtype=np.float32)
        m2 = np.empty(0, dtype=np.float32)
        redshift = np.empty(0, dtype=np.float32)

    merger_time = np.sort(
        np.random.uniform(0.0, FRAME_DURATION_SECONDS, size=event_count)
    ).astype(np.float32)
    ra = np.random.uniform(0.0, 2.0 * np.pi, event_count).astype(np.float32)
    psi = np.random.uniform(0.0, 2.0 * np.pi, event_count).astype(np.float32)
    phase = np.random.uniform(0.0, 2.0 * np.pi, event_count).astype(np.float32)
    inclination = np.arccos(np.random.uniform(-1.0, 1.0, event_count)).astype(
        np.float32
    )
    dec = np.arcsin(np.random.uniform(-1.0, 1.0, event_count)).astype(np.float32)
    distance = np.asarray(
        Planck18.luminosity_distance(redshift).value, dtype=np.float32
    )
    return {
        "tc": merger_time,
        "m1": m1,
        "m2": m2,
        "z": redshift,
        "dL": distance,
        "ra": ra,
        "dec": dec,
        "psi": psi,
        "phi_c": phase,
        "iota": inclination,
        "num_events": event_count,
    }


def sample_population(theta: dict[str, float], population_index: int):
    attempt_count = MAX_POPULATION_ATTEMPTS if CONDITION_ON_NONEMPTY_POPULATION else 1
    for attempt_index in range(attempt_count):
        seed = signal_seed(population_index, attempt_index)
        population = _sample_population_once(theta, seed)
        if not CONDITION_ON_NONEMPTY_POPULATION or population["num_events"] > 0:
            return population, seed, attempt_index + 1
    raise RuntimeError(
        f"Population {population_index} remained empty after {attempt_count} attempts."
    )


def _new_interferometers(zero_noise: bool, seed: int | None = None):
    psd = PowerSpectralDensity(asd_file=str(O4_ASD_PATH))
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for interferometer in interferometers:
        interferometer.minimum_frequency = MINIMUM_FREQUENCY_HZ
        interferometer.sampling_frequency = SAMPLING_FREQUENCY_HZ
        interferometer.duration = FRAME_DURATION_SECONDS
        interferometer.power_spectral_density = psd

    if zero_noise:
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY_HZ,
            duration=FRAME_DURATION_SECONDS,
            start_time=0.0,
        )
    else:
        if seed is None:
            raise ValueError("A noise seed is required for PSD noise generation.")
        random.seed(seed)
        np.random.seed(seed)
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=SAMPLING_FREQUENCY_HZ,
            duration=FRAME_DURATION_SECONDS,
            start_time=0.0,
        )
    return interferometers


def synthesize_signal(population: dict[str, np.ndarray | int]):
    interferometers = _new_interferometers(zero_noise=True)
    waveform_arguments = {
        "waveform_approximant": "IMRPhenomXAS",
        "reference_frequency": REFERENCE_FREQUENCY_HZ,
        "minimum_frequency": MINIMUM_FREQUENCY_HZ,
    }
    injected_count = 0
    skipped_strong_count = 0
    errored_count = 0

    event_count = int(population["num_events"])
    for event_index in range(event_count):
        merger_time = float(population["tc"][event_index])
        injection_parameters = {
            # v100 no-double-redshift convention: saved masses are used directly.
            "mass_1": float(population["m1"][event_index]),
            "mass_2": float(population["m2"][event_index]),
            "a_1": 0.0,
            "a_2": 0.0,
            "luminosity_distance": float(population["dL"][event_index]),
            "psi": float(population["psi"][event_index]),
            "phase": float(population["phi_c"][event_index]),
            "geocent_time": merger_time,
            "ra": float(population["ra"][event_index]),
            "dec": float(population["dec"][event_index]),
            "theta_jn": float(population["iota"][event_index]),
            "tilt_1": 0.0,
            "tilt_2": 0.0,
            "phi_12": 0.0,
            "phi_jl": 0.0,
        }
        try:
            waveform_generator = bilby.gw.WaveformGenerator(
                duration=FRAME_DURATION_SECONDS,
                sampling_frequency=SAMPLING_FREQUENCY_HZ,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                waveform_arguments=waveform_arguments,
                parameter_conversion=(
                    bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
                ),
            )
            with suppress_c_stderr():
                polarizations_td = waveform_generator.time_domain_strain(
                    parameters=injection_parameters
                )
            plus = np.roll(polarizations_td["plus"], -512)
            cross = np.roll(polarizations_td["cross"], -512)
            dt = float(waveform_generator.time_array[1] - waveform_generator.time_array[0])

            event_duration = int(merger_time / dt)
            plus_windowed = np.zeros_like(plus)
            cross_windowed = np.zeros_like(cross)
            if event_duration > 0:
                plus_windowed[-event_duration:] = plus[-event_duration:]
                cross_windowed[-event_duration:] = cross[-event_duration:]
            polarizations_fd = {
                "plus": np.fft.rfft(plus_windowed) * dt,
                "cross": np.fft.rfft(cross_windowed) * dt,
            }

            network_snr_squared = 0.0
            for interferometer in interferometers:
                response = interferometer.get_detector_response(
                    polarizations_fd, injection_parameters
                )
                snr_squared = interferometer.optimal_snr_squared(response)
                network_snr_squared += max(float(np.real(snr_squared)), 0.0)
            network_snr = float(np.sqrt(network_snr_squared))
            if network_snr >= NETWORK_SNR_THRESHOLD:
                skipped_strong_count += 1
                continue

            interferometers.inject_signal(
                injection_parameters,
                injection_polarizations=polarizations_fd,
                raise_error=False,
            )
            cut_index = int(merger_time * SAMPLING_FREQUENCY_HZ)
            for interferometer in interferometers:
                if cut_index < len(interferometer.time_domain_strain):
                    interferometer.time_domain_strain[cut_index:] = 0.0
            injected_count += 1
        except Exception as exc:
            errored_count += 1
            tqdm.write(
                f"population event {event_index} waveform failed: "
                f"{type(exc).__name__}: {exc}"
            )

    signals = {
        interferometer.name: np.asarray(
            interferometer.time_domain_strain, dtype=np.float64
        ).copy()
        for interferometer in interferometers
    }
    return signals, injected_count, skipped_strong_count, errored_count


def _label_vector(theta: dict[str, float]) -> np.ndarray:
    return np.asarray([theta[name] for name in FULL_LABEL_NAMES], dtype=np.float32)


def _atomic_save(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    temporary_path.replace(path)


def generate_one(index: int, hyperparameters: dict[str, np.ndarray]) -> dict:
    file_suffix = f"IID-SBC-pop{index:05d}_sample00000_p0.npy"
    h1_path = H1_DIR / f"H1-{file_suffix}"
    l1_path = L1_DIR / f"L1-{file_suffix}"
    if h1_path.exists() and l1_path.exists():
        return {"index": index, "status": "skipped", "h1": h1_path, "l1": l1_path}

    theta = {name: float(hyperparameters[name][index]) for name in FULL_LABEL_NAMES}
    population, used_signal_seed, attempts_used = sample_population(theta, index)
    signals, injected_count, skipped_strong_count, errored_count = synthesize_signal(
        population
    )

    used_noise_seed = noise_seed(index)
    noise_interferometers = _new_interferometers(
        zero_noise=False, seed=used_noise_seed
    )
    noises = {
        interferometer.name: np.asarray(
            interferometer.time_domain_strain, dtype=np.float64
        )
        for interferometer in noise_interferometers
    }

    label = _label_vector(theta)
    metadata = {
        "population_index": index,
        "signal_seed": used_signal_seed,
        "noise_seed": used_noise_seed,
        "population_attempts": attempts_used,
        "num_events": int(population["num_events"]),
        "injected_events": injected_count,
        "skipped_strong_events": skipped_strong_count,
        "errored_events": errored_count,
        "network_snr_threshold": NETWORK_SNR_THRESHOLD,
        "noise_scale": NOISE_SCALE,
    }
    for detector_name, path in (("H1", h1_path), ("L1", l1_path)):
        combined = signals[detector_name] + NOISE_SCALE * noises[detector_name]
        _atomic_save(
            path,
            {
                "data": combined.astype(np.float32),
                "label": label.copy(),
                "metadata": dict(metadata, detector=detector_name),
            },
        )
    return {"index": index, "status": "generated", "h1": h1_path, "l1": l1_path, **metadata}


def write_generation_log() -> None:
    rows = []
    for h1_path in sorted(H1_DIR.glob("H1-IID-SBC-pop*_sample00000_p0.npy")):
        sample = np.load(h1_path, allow_pickle=True).item()
        metadata = dict(sample.get("metadata", {}))
        metadata["h1_file"] = h1_path.name
        metadata["l1_file"] = h1_path.name.replace("H1-", "L1-", 1)
        rows.append(metadata)
    if not rows:
        return
    fieldnames = list(rows[0])
    with GENERATION_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict-IID v100 SBC observations.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive population index.")
    parser.add_argument(
        "--stop", type=int, default=NUM_SBC_SAMPLES, help="Exclusive population index."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_config()
    if not HYPERPARAMETER_PATH.is_file():
        raise FileNotFoundError(
            f"Generate IID hyperparameters first: {HYPERPARAMETER_PATH}"
        )
    if not O4_ASD_PATH.is_file():
        raise FileNotFoundError(f"O4 ASD file not found: {O4_ASD_PATH}")
    start = max(0, args.start)
    stop = min(NUM_SBC_SAMPLES, args.stop)
    if stop <= start:
        raise ValueError(f"Invalid generation range [{start}, {stop}).")

    H1_DIR.mkdir(parents=True, exist_ok=True)
    L1_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(HYPERPARAMETER_PATH, allow_pickle=False) as saved:
        hyperparameters = {name: np.asarray(saved[name]) for name in FULL_LABEL_NAMES}

    print(f"Generating strict-IID observations [{start}, {stop})")
    print(f"H1 output: {H1_DIR}")
    print(f"L1 output: {L1_DIR}")
    print(f"O4 ASD: {O4_ASD_PATH}")
    print(f"Condition on non-empty population: {CONDITION_ON_NONEMPTY_POPULATION}")
    progress = tqdm(
        range(start, stop),
        total=stop - start,
        desc="Generating strict-IID observations",
        unit="population",
    )
    for index in progress:
        result = generate_one(index, hyperparameters)
        progress.set_postfix_str(
            f"pop={index:05d} status={result['status']}",
            refresh=True,
        )
    write_generation_log()
    print(f"Generation log: {GENERATION_LOG_PATH}")


if __name__ == "__main__":
    main()
