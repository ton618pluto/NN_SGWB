from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Make the repository-local scripts/gwfast package importable regardless of
# the directory from which this file is launched.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from gwfast import gwfastGlobals as gwfast_globals  # noqa: E402
    from gwfast import gwfastUtils as gwfast_utils  # noqa: E402
    from gwfast.network import DetNet  # noqa: E402
    from gwfast.signal import GWSignal  # noqa: E402
    from gwfast.waveforms import IMRPhenomD, LAL_WF  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires the repository-local gwfast package and its "
        "dependencies (including jax; LALSuite is also required when using "
        "WAVEFORM_MODEL='IMRPhenomXAS'). Run it in the configured GWFAST environment."
    ) from exc


# ---------------------------- User controls ----------------------------
VERSION = "v100"
POPULATION_INDEX = 0

# The generator uses job numbers starting at 1. Job 1 corresponds to
# [0, 2048) s, job 2 to [2048, 4096) s, and so on.
FRAME_JOB_NUMBER = 1
FRAME_DURATION_SECONDS = 2048.0

MINIMUM_FREQUENCY_HZ = 10.0
MAXIMUM_FREQUENCY_HZ = 256.0  # Nyquist frequency for the 512 Hz generator.
REFERENCE_FREQUENCY_HZ = 25.0
SNR_FREQUENCY_POINTS = 1000
BATCH_SIZE = 2048

# "IMRPhenomXAS" uses GWFAST's LAL wrapper, matches the original generator,
# and does not require GWFAST's optional WFfiles/QNMData_* lookup tables.
# "IMRPhenomD" is faster and vectorized, but requires those lookup tables.
WAVEFORM_MODEL = "IMRPhenomXAS"

PARAMETER_DIR = SCRIPTS_DIR / "parameter_sampling_train" / VERSION
O4_ASD_PATH = SCRIPTS_DIR / "O4_mock.txt"
OUTPUT_DIR = SCRIPTS_DIR / "gwfast_snr_results" / VERSION


def _parameter_path() -> Path:
    return PARAMETER_DIR / f"CBC_params_example{POPULATION_INDEX:05d}.npz"


def _select_events_in_frame(
    parameters: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, dict[str, np.ndarray], float, float]:
    required = ("tc", "m1", "m2", "dL", "ra", "dec", "psi", "phi_c", "iota")
    missing = [name for name in required if name not in parameters.files]
    if missing:
        raise KeyError(f"Parameter file is missing required arrays: {missing}")

    all_values = {
        name: np.asarray(parameters[name], dtype=np.float64).reshape(-1)
        for name in required
    }
    lengths = {name: values.size for name, values in all_values.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Event parameter arrays have inconsistent lengths: {lengths}")

    frame_start = (FRAME_JOB_NUMBER - 1) * FRAME_DURATION_SECONDS
    frame_end = frame_start + FRAME_DURATION_SECONDS

    # This reproduces the overlap mask in
    # generate_pure_cbc_frames_sampling5_silent_no_double_redshift_o4_mock.py.
    m1 = all_values["m1"]
    m2 = all_values["m2"]
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mtsun_si = 4.925491025543576e-6
    f0 = MINIMUM_FREQUENCY_HZ - 1.0
    coefficient = 5.0 ** (3.0 / 8.0) / (8.0 * np.pi)
    estimated_start = all_values["tc"] - (f0 / coefficient) ** (-8.0 / 3.0) * (
        chirp_mass * mtsun_si
    ) ** (-5.0 / 3.0)
    mask = (all_values["tc"] > frame_start) & (estimated_start < frame_end)

    original_indices = np.flatnonzero(mask).astype(np.int64)
    selected = {name: values[mask] for name, values in all_values.items()}
    return original_indices, selected, frame_start, frame_end


def _build_gwfast_events(selected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    m1 = selected["m1"]
    m2 = selected["m2"]
    total_mass = m1 + m2
    eta = (m1 * m2) / total_mass**2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)

    # The parameter files and Bilby generator use luminosity distance in Mpc,
    # whereas GWFAST expects Gpc.
    distance_gpc = selected["dL"] / 1000.0

    # GWFAST sky angles are theta=colatitude and phi=right ascension.
    theta = 0.5 * np.pi - selected["dec"]
    phi = selected["ra"]

    # The original generator passes tc to Bilby as geocent_time. Convert the
    # same numeric GPS time to GWFAST's GMST fraction-of-day convention.
    tcoal = np.asarray(gwfast_utils.GPSt_to_GMST_alt(selected["tc"]), dtype=np.float64)

    zeros = np.zeros_like(chirp_mass)
    return {
        "Mc": chirp_mass,
        "eta": eta,
        "dL": distance_gpc,
        "theta": theta,
        "phi": phi,
        "iota": selected["iota"],
        "psi": selected["psi"],
        "tcoal": tcoal,
        "Phicoal": selected["phi_c"],
        "chi1z": zeros,
        "chi2z": zeros,
    }


def _build_network() -> DetNet:
    if not O4_ASD_PATH.exists():
        raise FileNotFoundError(f"O4 ASD file not found: {O4_ASD_PATH}")

    if WAVEFORM_MODEL == "IMRPhenomD":
        waveform = IMRPhenomD(fRef=REFERENCE_FREQUENCY_HZ)
    elif WAVEFORM_MODEL == "IMRPhenomXAS":
        waveform = LAL_WF("IMRPhenomXAS", fRef=REFERENCE_FREQUENCY_HZ)
    else:
        raise ValueError(
            f"Unsupported WAVEFORM_MODEL={WAVEFORM_MODEL!r}; "
            "choose 'IMRPhenomD' or 'IMRPhenomXAS'."
        )

    detector_signals = {}
    for detector_name in ("H1", "L1"):
        detector = gwfast_globals.detectors[detector_name]
        detector_signals[detector_name] = GWSignal(
            waveform,
            psd_path=str(O4_ASD_PATH),
            detector_shape=detector["shape"],
            det_lat=detector["lat"],
            det_long=detector["long"],
            det_xax=detector["xax"],
            verbose=False,
            is_ASD=True,
            useEarthMotion=False,
            fmin=MINIMUM_FREQUENCY_HZ,
            fmax=MAXIMUM_FREQUENCY_HZ,
            IntTablePath=None,
        )
    return DetNet(detector_signals, verbose=False)


def _slice_events(
    events: dict[str, np.ndarray], start: int, stop: int
) -> dict[str, np.ndarray]:
    return {name: values[start:stop] for name, values in events.items()}


def _calculate_snrs(
    network: DetNet,
    events: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_count = len(events["Mc"])
    h1_parts: list[np.ndarray] = []
    l1_parts: list[np.ndarray] = []
    network_parts: list[np.ndarray] = []

    for start in range(0, event_count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, event_count)
        batch = _slice_events(events, start, stop)
        result = network.SNR(
            batch,
            res=SNR_FREQUENCY_POINTS,
            return_all=True,
        )
        h1_parts.append(np.asarray(result["H1"], dtype=np.float64).reshape(-1))
        l1_parts.append(np.asarray(result["L1"], dtype=np.float64).reshape(-1))
        network_parts.append(np.asarray(result["net"], dtype=np.float64).reshape(-1))
        print(f"GWFAST SNR progress: {stop}/{event_count}")

    return (
        np.concatenate(h1_parts),
        np.concatenate(l1_parts),
        np.concatenate(network_parts),
    )


def _save_results(
    parameter_path: Path,
    original_indices: np.ndarray,
    selected: dict[str, np.ndarray],
    h1_snr: np.ndarray,
    l1_snr: np.ndarray,
    network_snr: np.ndarray,
    frame_start: float,
    frame_end: float,
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pop{POPULATION_INDEX:05d}_frame{FRAME_JOB_NUMBER:05d}_gwfast_snr"
    array_path = OUTPUT_DIR / f"{stem}.npy"
    archive_path = OUTPUT_DIR / f"{stem}.npz"
    text_path = OUTPUT_DIR / f"{stem}.txt"

    np.save(array_path, network_snr)
    np.savez_compressed(
        archive_path,
        event_index=original_indices,
        tc=selected["tc"],
        m1=selected["m1"],
        m2=selected["m2"],
        dL_Mpc=selected["dL"],
        snr_H1=h1_snr,
        snr_L1=l1_snr,
        snr_network=network_snr,
        frame_start_seconds=np.asarray(frame_start),
        frame_end_seconds=np.asarray(frame_end),
        waveform_model=np.asarray(WAVEFORM_MODEL),
        source_parameter_file=np.asarray(str(parameter_path)),
    )

    minimum = float(network_snr.min())
    maximum = float(network_snr.max())
    lines = [
        f"Parameter file: {parameter_path}",
        f"Population index: {POPULATION_INDEX}",
        f"Frame job number: {FRAME_JOB_NUMBER}",
        f"Frame interval [s]: [{frame_start:.6f}, {frame_end:.6f})",
        f"Waveform model: {WAVEFORM_MODEL}",
        f"Detector network: H1 + L1",
        f"ASD: {O4_ASD_PATH}",
        f"Frequency range [Hz]: {MINIMUM_FREQUENCY_HZ:g}-{MAXIMUM_FREQUENCY_HZ:g}",
        f"Frequency integration points: {SNR_FREQUENCY_POINTS}",
        f"Number of overlapping events: {network_snr.size}",
        f"Minimum network SNR: {minimum:.10f}",
        f"Maximum network SNR: {maximum:.10f}",
        "",
        "Network SNR array:",
        np.array2string(
            network_snr,
            precision=10,
            separator=", ",
            threshold=network_snr.size + 1,
            max_line_width=120,
        ),
        "",
        "event_index tc_s m1_Msun m2_Msun dL_Mpc snr_H1 snr_L1 snr_network",
    ]
    for row in zip(
        original_indices,
        selected["tc"],
        selected["m1"],
        selected["m2"],
        selected["dL"],
        h1_snr,
        l1_snr,
        network_snr,
    ):
        lines.append(
            f"{int(row[0]):d} "
            f"{row[1]:.10f} {row[2]:.10f} {row[3]:.10f} {row[4]:.10f} "
            f"{row[5]:.10f} {row[6]:.10f} {row[7]:.10f}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return array_path, archive_path, text_path


def main() -> None:
    if FRAME_JOB_NUMBER < 1:
        raise ValueError("FRAME_JOB_NUMBER must be at least 1.")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")
    if SNR_FREQUENCY_POINTS < 10:
        raise ValueError("SNR_FREQUENCY_POINTS must be at least 10.")

    parameter_path = _parameter_path()
    if not parameter_path.exists():
        raise FileNotFoundError(f"CBC parameter file not found: {parameter_path}")

    with np.load(parameter_path, allow_pickle=False) as parameters:
        original_indices, selected, frame_start, frame_end = _select_events_in_frame(
            parameters
        )

    event_count = original_indices.size
    print(f"Parameter file: {parameter_path}")
    print(f"Frame interval: [{frame_start:.6f}, {frame_end:.6f}) s")
    print(f"Overlapping events: {event_count}")
    if event_count == 0:
        print("No overlapping events; no SNR files were written.")
        return

    events = _build_gwfast_events(selected)
    network = _build_network()
    h1_snr, l1_snr, network_snr = _calculate_snrs(network, events)

    if not np.all(np.isfinite(network_snr)):
        bad = np.flatnonzero(~np.isfinite(network_snr))
        raise RuntimeError(f"Non-finite network SNR values at selected indices: {bad}")

    print("Network SNR array:")
    print(
        np.array2string(
            network_snr,
            precision=10,
            separator=", ",
            threshold=network_snr.size + 1,
            max_line_width=120,
        )
    )
    print(f"Minimum network SNR: {network_snr.min():.10f}")
    print(f"Maximum network SNR: {network_snr.max():.10f}")

    array_path, archive_path, text_path = _save_results(
        parameter_path,
        original_indices,
        selected,
        h1_snr,
        l1_snr,
        network_snr,
        frame_start,
        frame_end,
    )
    print(f"Saved network SNR array: {array_path}")
    print(f"Saved detailed NPZ: {archive_path}")
    print(f"Saved TXT report: {text_path}")


if __name__ == "__main__":
    main()
