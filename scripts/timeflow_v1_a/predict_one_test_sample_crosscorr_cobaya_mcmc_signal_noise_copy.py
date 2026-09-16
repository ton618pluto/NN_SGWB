from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import csd, welch

from gwfast.gwfastGlobals import detectors as GWFAST_DETECTORS
from gwfast.stochastic.stochasticTools import overlap_reduction_function
from predict_one_test_sample_cobaya_mcmc import (
    _collection_to_array,
    _credible_levels_from_density,
    _expand_if_degenerate,
    _range_based_limits,
    _safe_kde_1d,
    _safe_kde_2d,
    cobaya_run,
)


BASE_VERSION = "v100"
NOISE_SCALE = 100
SIGNAL_VERSION = f"fixed_one_test_{BASE_VERSION}"
SUPERIMPOSED_VERSION = f"{SIGNAL_VERSION}_random_two_noise_{NOISE_SCALE}"
FOLDER_NAME = "training_set0"

START_SAMPLE_INDEX = 0
NUM_TEST_SAMPLES = 1
SEED = 42
MAX_SAMPLES = 8000
PROPOSAL_STD_LOG10_OMEGA = 0.2
PROPOSAL_STD_ALPHA = 0.15
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
SAVE_TEXT_SUMMARY = False
SAVE_COBAYA_CHAIN = False
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"

FFT_LENGTH = 8.0
OVERLAP = 4.0
WINDOW = "hann"
FREQ_MIN = 20.0
FREQ_MAX = 172.0
F_REF = 25.0
MIN_ORF_ABS = 1e-3
EPS = 1e-30

LOG10_OMEGA_PRIOR = (-14.0, -5.0)
ALPHA_PRIOR = (-4.0, 4.0)

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(__file__).resolve().parent
RESULT_DIR = VERSION_DIR / "result_fig"
SUPERIMPOSED_ROOT = SCRIPTS_DIR / "training_set_superimposed_test" / SUPERIMPOSED_VERSION
SIGNAL_ROOT = SCRIPTS_DIR / "training_set_test" / SIGNAL_VERSION / FOLDER_NAME

H0_SI = 67.66 * 1000.0 / 3.085677581491367e22
STRAIN_PREFAC = 3.0 * H0_SI * H0_SI / (10.0 * np.pi * np.pi)


def build_sorted_pairs(root: Path) -> list[tuple[Path, Path]]:
    h1_dir = root / "H1"
    l1_dir = root / "L1"
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 directory not found: {l1_dir}")

    h1_files = sorted(path for path in h1_dir.iterdir() if path.suffix == ".gwf")
    l1_map = {
        path.name.replace("L1-", "", 1): path
        for path in l1_dir.iterdir()
        if path.suffix == ".gwf"
    }

    pairs: list[tuple[Path, Path]] = []
    for h1_path in h1_files:
        key = h1_path.name.replace("H1-", "", 1)
        l1_path = l1_map.get(key)
        if l1_path is not None:
            pairs.append((h1_path, l1_path))
    if not pairs:
        raise RuntimeError(f"No matched H1/L1 GWF pairs found under {root}")
    return pairs


def load_series(path: Path, detector: str) -> TimeSeries:
    channel = f"{detector}:{STRAIN_CHANNEL_SUFFIX}"
    return TimeSeries.read(str(path), channel=channel)


def compute_num_averages(num_samples: int, nperseg: int, noverlap: int) -> int:
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("nperseg must be larger than noverlap")
    if num_samples < nperseg:
        return 1
    return 1 + (num_samples - nperseg) // step


def prepare_crosscorr_summary(h1: np.ndarray, l1: np.ndarray, sample_rate: float) -> dict[str, np.ndarray | float | int]:
    nperseg = int(round(FFT_LENGTH * sample_rate))
    noverlap = int(round(OVERLAP * sample_rate))
    if nperseg <= 0:
        raise ValueError("FFT_LENGTH is too small for the sample rate")
    if nperseg > len(h1):
        nperseg = len(h1)
        noverlap = min(noverlap, max(0, nperseg // 2))

    freqs, psd_h1 = welch(h1, fs=sample_rate, window=WINDOW, nperseg=nperseg, noverlap=noverlap, scaling="density")
    _, psd_l1 = welch(l1, fs=sample_rate, window=WINDOW, nperseg=nperseg, noverlap=noverlap, scaling="density")
    _, cpsd = csd(h1, l1, fs=sample_rate, window=WINDOW, nperseg=nperseg, noverlap=noverlap, scaling="density")

    gamma = np.real(
        overlap_reduction_function(
            freqs,
            GWFAST_DETECTORS["H1"],
            GWFAST_DETECTORS["L1"],
            det1_name="H1",
            det2_name="L1",
        )["H1_L1"]
    )

    freq_max = min(FREQ_MAX, sample_rate * 0.5 * 0.98)
    mask = (freqs >= FREQ_MIN) & (freqs <= freq_max) & (np.abs(gamma) >= MIN_ORF_ABS)
    if not np.any(mask):
        raise RuntimeError("No frequency bins remain after applying the cross-correlation mask.")

    freqs = freqs[mask]
    psd_h1 = psd_h1[mask]
    psd_l1 = psd_l1[mask]
    cpsd = cpsd[mask]
    gamma = gamma[mask]

    num_avg = compute_num_averages(len(h1), nperseg, noverlap)
    sigma = np.sqrt(np.maximum(psd_h1 * psd_l1 / (2.0 * num_avg), EPS))
    return {
        "freqs": freqs.astype(np.float64),
        "psd_h1": psd_h1.astype(np.float64),
        "psd_l1": psd_l1.astype(np.float64),
        "cpsd_real": np.real(cpsd).astype(np.float64),
        "gamma": gamma.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "sample_rate": float(sample_rate),
        "num_avg": int(num_avg),
    }


def omega_powerlaw(freqs: np.ndarray, log10_omega_ref: float, alpha: float) -> np.ndarray:
    return (10.0 ** log10_omega_ref) * np.power(freqs / F_REF, alpha)


def model_cross_psd(freqs: np.ndarray, gamma: np.ndarray, log10_omega_ref: float, alpha: float) -> np.ndarray:
    omega = omega_powerlaw(freqs, log10_omega_ref, alpha)
    strain_psd = STRAIN_PREFAC * omega / np.maximum(freqs, 1e-12) ** 3
    return gamma * strain_psd


def fit_effective_powerlaw(freqs: np.ndarray, gamma: np.ndarray, cpsd_real: np.ndarray) -> tuple[float, float] | None:
    valid = (cpsd_real > 0.0) & (np.abs(gamma) >= MIN_ORF_ABS)
    if np.count_nonzero(valid) < 4:
        return None

    x = np.log(freqs[valid] / F_REF)
    y = np.log(cpsd_real[valid] / (np.abs(gamma[valid]) * STRAIN_PREFAC) * np.maximum(freqs[valid], 1e-12) ** 3)
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept / np.log(10.0)), float(slope)


def draw_corner_plot(
    posterior: np.ndarray,
    parameter_names: list[str],
    parameter_ranges: dict[str, tuple[float, float]],
    figure_path: Path,
    title: str,
    reference_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q50 = np.percentile(posterior, 50, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(parameter_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(3.2 * n_params, 3.2 * n_params))
    if n_params == 1:
        axes = np.asarray([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                ref_value = means[col] if reference_values is None else reference_values[col]
                limits = _range_based_limits(samples, ref_value, means[col], parameter_ranges.get(parameter_names[col]))
                x_grid, y_grid = _safe_kde_1d(samples, limits)
                if x_grid is None:
                    ax.hist(samples, bins=40, density=True, color="#4DAA57", alpha=0.35)
                else:
                    ax.plot(x_grid, y_grid, color="#4DAA57", linewidth=1.8)
                if reference_values is not None:
                    ax.axvline(reference_values[col], color="red", linestyle="--", linewidth=1.0)
                ax.set_xlim(*limits)
                ax.set_yticks([])
                ax.set_title(parameter_names[col])
            elif row > col:
                x = posterior[:, col]
                y = posterior[:, row]
                x_ref = means[col] if reference_values is None else reference_values[col]
                y_ref = means[row] if reference_values is None else reference_values[row]
                x_limits = _range_based_limits(x, x_ref, means[col], parameter_ranges.get(parameter_names[col]))
                y_limits = _range_based_limits(y, y_ref, means[row], parameter_ranges.get(parameter_names[row]))
                xx, yy, zz = _safe_kde_2d(x, y, x_limits, y_limits)
                if zz is None:
                    ax.scatter(x, y, s=4, alpha=0.15, color="#4DAA57")
                else:
                    ax.contourf(xx, yy, zz, levels=14, cmap="Greens")
                    level_68, level_95 = _credible_levels_from_density(zz, [0.68, 0.95])
                    contour_levels = sorted(set([level_95, level_68]))
                    ax.contour(
                        xx,
                        yy,
                        zz,
                        levels=contour_levels,
                        colors=["#3F8F4A", "#1E5F2B"][: len(contour_levels)],
                        linewidths=[1.0, 1.4][: len(contour_levels)],
                    )
                if reference_values is not None:
                    ax.scatter(reference_values[col], reference_values[row], color="red", marker="x", s=60, linewidths=2)
                ax.scatter(means[col], means[row], color="black", marker="o", s=18)
                ax.set_xlim(*_expand_if_degenerate(*x_limits))
                ax.set_ylim(*_expand_if_degenerate(*y_limits))
            else:
                ax.axis("off")

            if row == n_params - 1 and col <= row:
                ax.set_xlabel(parameter_names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(parameter_names[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return means, q16, q50, q84


def plot_cross_spectrum(
    summary: dict[str, np.ndarray | float | int],
    figure_path: Path,
    title: str,
    posterior_mean: np.ndarray | None = None,
    reference_values: np.ndarray | None = None,
) -> None:
    freqs = np.asarray(summary["freqs"], dtype=np.float64)
    cpsd_real = np.asarray(summary["cpsd_real"], dtype=np.float64)
    sigma = np.asarray(summary["sigma"], dtype=np.float64)
    gamma = np.asarray(summary["gamma"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(freqs, cpsd_real, color="black", linewidth=1.1, label="Observed Re[CSD]")
    ax.fill_between(freqs, cpsd_real - sigma, cpsd_real + sigma, color="#BDBDBD", alpha=0.35, label="Approx. 1 sigma")

    if posterior_mean is not None:
        mean_curve = model_cross_psd(freqs, gamma, posterior_mean[0], posterior_mean[1])
        ax.plot(freqs, mean_curve, color="#4DAA57", linewidth=1.6, label="Posterior mean template")

    if reference_values is not None:
        ref_curve = model_cross_psd(freqs, gamma, reference_values[0], reference_values[1])
        ax.plot(freqs, ref_curve, color="#C65F5F", linestyle="--", linewidth=1.3, label="Signal-only fitted template")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cross PSD [1/Hz]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if cobaya_run is None:
        raise ImportError("Cobaya is not installed. Please install 'cobaya' before running this script.")

    np.random.seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    superimposed_pairs = build_sorted_pairs(SUPERIMPOSED_ROOT)
    signal_pairs = build_sorted_pairs(SIGNAL_ROOT)
    if len(signal_pairs) < len(superimposed_pairs):
        raise RuntimeError("Signal-only files are fewer than the superimposed files.")

    end_sample_index = min(len(superimposed_pairs), START_SAMPLE_INDEX + NUM_TEST_SAMPLES)
    parameter_names = ["log10_omega_ref", "alpha"]
    parameter_ranges = {
        "log10_omega_ref": LOG10_OMEGA_PRIOR,
        "alpha": ALPHA_PRIOR,
    }

    print(f"Signal root: {SIGNAL_ROOT}")
    print(f"Superimposed root: {SUPERIMPOSED_ROOT}")
    print(f"Frequency band: {FREQ_MIN} - {FREQ_MAX} Hz")
    print(f"FFT settings: fftlength={FFT_LENGTH}, overlap={OVERLAP}, window={WINDOW}")

    for sample_index in range(START_SAMPLE_INDEX, end_sample_index):
        h1_noise_path, l1_noise_path = superimposed_pairs[sample_index]
        h1_signal_path, l1_signal_path = signal_pairs[sample_index]

        h1_noise = load_series(h1_noise_path, "H1")
        l1_noise = load_series(l1_noise_path, "L1")
        h1_signal = load_series(h1_signal_path, "H1")
        l1_signal = load_series(l1_signal_path, "L1")

        if len(h1_noise) != len(l1_noise):
            raise ValueError(f"Superimposed H1/L1 length mismatch for sample {sample_index}")
        if h1_noise.sample_rate.value != l1_noise.sample_rate.value:
            raise ValueError(f"Superimposed H1/L1 sample rate mismatch for sample {sample_index}")

        sample_rate = float(h1_noise.sample_rate.value)
        summary = prepare_crosscorr_summary(
            np.asarray(h1_noise.value, dtype=np.float64),
            np.asarray(l1_noise.value, dtype=np.float64),
            sample_rate,
        )
        signal_summary = prepare_crosscorr_summary(
            np.asarray(h1_signal.value, dtype=np.float64),
            np.asarray(l1_signal.value, dtype=np.float64),
            sample_rate,
        )

        effective_truth = fit_effective_powerlaw(
            np.asarray(signal_summary["freqs"], dtype=np.float64),
            np.asarray(signal_summary["gamma"], dtype=np.float64),
            np.asarray(signal_summary["cpsd_real"], dtype=np.float64),
        )

        def crosscorr_loglike(log10_omega_ref: float, alpha: float) -> float:
            freqs = np.asarray(summary["freqs"], dtype=np.float64)
            gamma = np.asarray(summary["gamma"], dtype=np.float64)
            data = np.asarray(summary["cpsd_real"], dtype=np.float64)
            sigma = np.asarray(summary["sigma"], dtype=np.float64)
            model = model_cross_psd(freqs, gamma, log10_omega_ref, alpha)
            resid = (data - model) / sigma
            return float(-0.5 * np.sum(resid * resid + np.log(2.0 * np.pi * sigma * sigma)))

        ref_log10_omega = -9.0
        ref_alpha = 2.0 / 3.0
        if effective_truth is not None:
            ref_log10_omega, ref_alpha = effective_truth

        info = {
            "force": True,
            "resume": False,
            "params": {
                "log10_omega_ref": {
                    "prior": {"min": LOG10_OMEGA_PRIOR[0], "max": LOG10_OMEGA_PRIOR[1]},
                    "ref": float(np.clip(ref_log10_omega, *LOG10_OMEGA_PRIOR)),
                    "proposal": PROPOSAL_STD_LOG10_OMEGA,
                },
                "alpha": {
                    "prior": {"min": ALPHA_PRIOR[0], "max": ALPHA_PRIOR[1]},
                    "ref": float(np.clip(ref_alpha, *ALPHA_PRIOR)),
                    "proposal": PROPOSAL_STD_ALPHA,
                },
            },
            "likelihood": {
                "crosscorr_posterior": {
                    "external": crosscorr_loglike,
                    "input_params": parameter_names,
                }
            },
            "sampler": {
                "mcmc": {
                    "max_samples": MAX_SAMPLES,
                    "learn_proposal": True,
                    "Rminus1_stop": RMINUS1_STOP,
                    "Rminus1_cl_stop": RMINUS1_CL_STOP,
                    "measure_speeds": False,
                }
            },
        }
        if SAVE_COBAYA_CHAIN:
            info["output"] = str(RESULT_DIR / f"crosscorr_chain_signal_noise_{BASE_VERSION}_{sample_index:05d}")

        print(f"Sample index: {sample_index}")
        print(f"Superimposed H1: {h1_noise_path}")
        print(f"Superimposed L1: {l1_noise_path}")
        print(f"Signal-only H1:   {h1_signal_path}")
        print(f"Signal-only L1:   {l1_signal_path}")
        print(f"Effective signal-only fit: {effective_truth}")
        print(f"Frequency bins used: {len(np.asarray(summary['freqs']))}")
        print(f"Welch averages used: {summary['num_avg']}")

        _, sampler = cobaya_run(info, force=True)
        posterior = _collection_to_array(sampler.products()["sample"], parameter_names)
        reference_values = None if effective_truth is None else np.asarray(effective_truth, dtype=np.float64)

        corner_path = RESULT_DIR / f"crosscorr_signal_noise_corner_{BASE_VERSION}_{sample_index:05d}.png"
        means, q16, q50, q84 = draw_corner_plot(
            posterior,
            parameter_names,
            parameter_ranges,
            corner_path,
            f"Cross-correlation Posterior (signal+noise, sample {sample_index})",
            reference_values=reference_values,
        )

        spectrum_path = RESULT_DIR / f"crosscorr_signal_noise_spectrum_{BASE_VERSION}_{sample_index:05d}.png"
        plot_cross_spectrum(
            summary,
            spectrum_path,
            f"Cross spectrum (signal+noise, sample {sample_index})",
            posterior_mean=means,
            reference_values=reference_values,
        )

        lines = [
            f"Sample index: {sample_index}",
            f"Base version: {BASE_VERSION}",
            f"Signal version: {SIGNAL_VERSION}",
            f"Superimposed version: {SUPERIMPOSED_VERSION}",
            f"H1 superimposed: {h1_noise_path}",
            f"L1 superimposed: {l1_noise_path}",
            f"H1 signal-only: {h1_signal_path}",
            f"L1 signal-only: {l1_signal_path}",
            f"Frequency bins used: {len(np.asarray(summary['freqs']))}",
            f"Welch averages: {summary['num_avg']}",
            f"Sample rate: {summary['sample_rate']}",
            f"FFT length: {FFT_LENGTH}",
            f"Overlap: {OVERLAP}",
            f"Window: {WINDOW}",
            "",
        ]
        for index, name in enumerate(parameter_names):
            line = (
                f"{name:<16} mean={means[index]:.6f} median={q50[index]:.6f} "
                f"ci16={q16[index]:.6f} ci84={q84[index]:.6f}"
            )
            if reference_values is not None:
                in68 = "Y" if q16[index] <= reference_values[index] <= q84[index] else "N"
                line += f" ref={reference_values[index]:.6f} in68={in68}"
            lines.append(line)

        summary_text = "\n".join(lines)
        print(summary_text)
        if SAVE_TEXT_SUMMARY:
            summary_path = RESULT_DIR / f"crosscorr_signal_noise_summary_{BASE_VERSION}_{sample_index:05d}.txt"
            summary_path.write_text(summary_text, encoding="utf-8")
            print(f"Saved summary: {summary_path}")
        print(f"Saved corner: {corner_path}")
        print(f"Saved spectrum: {spectrum_path}")


if __name__ == "__main__":
    main()
