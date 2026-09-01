from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "plot_noise_psd.py requires 'gwpy'. Install it in the active environment first."
    ) from exc


VERSION = "v100"
DETECTOR = "H1"
JOB_NUMBER = 1
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
SPECTRUM_MODE = "ASD"  # "PSD" or "ASD"

# If the generated noise was scaled in time domain, set this to the same factor.
# Example: generate_pure_noise_frames_seed.py currently uses NOISE_SCALE_FACTOR = 1 / 1000.
#NOISE_SCALE_FACTOR = 1.0 / 1000.0
NOISE_SCALE_FACTOR = 1000.0
UNDO_SCALE_FOR_PSD = False

# Welch-style PSD settings used by gwpy.
FFT_LENGTH = 8.0
OVERLAP = 4.0
WINDOW = "hann"

FREQ_MIN = 10.0
FREQ_MAX = 256.0

SCRIPTS_DIR = Path(__file__).resolve().parent
NOISE_ROOT = SCRIPTS_DIR / "noise_waveform" / VERSION / f"noise_waveform_{DETECTOR}"
PLOT_DIR = SCRIPTS_DIR / "plots_noise_psd"


def build_noise_path() -> Path:
    return NOISE_ROOT / f"{DETECTOR}-STRAIN-{JOB_NUMBER:05d}-duration.gwf"


def load_noise_series(path: Path) -> TimeSeries:
    channel = f"{DETECTOR}:{STRAIN_CHANNEL_SUFFIX}"
    return TimeSeries.read(path, channel)


def maybe_rescale(series: TimeSeries) -> TimeSeries:
    if not UNDO_SCALE_FOR_PSD:
        return series
    if NOISE_SCALE_FACTOR == 0:
        raise ValueError("NOISE_SCALE_FACTOR cannot be 0 when UNDO_SCALE_FOR_PSD=True")
    return TimeSeries(
        series.value / NOISE_SCALE_FACTOR,
        times=series.times,
        channel=series.channel,
        name=series.name,
        unit=series.unit,
    )


def plot_spectrum(series: TimeSeries, source_path: Path) -> Path:
    psd = series.psd(fftlength=FFT_LENGTH, overlap=OVERLAP, window=WINDOW)

    freqs = psd.frequencies.value
    if SPECTRUM_MODE.upper() == "ASD":
        values = np.sqrt(psd.value)
        spectrum_label = "Estimated ASD"
        y_label = "ASD"
        title_metric = "ASD"
        file_metric = "ASD"
    elif SPECTRUM_MODE.upper() == "PSD":
        values = psd.value
        spectrum_label = "Estimated PSD"
        y_label = "PSD"
        title_metric = "PSD"
        file_metric = "PSD"
    else:
        raise ValueError(f"Unsupported SPECTRUM_MODE: {SPECTRUM_MODE}")

    mask = freqs >= FREQ_MIN
    if FREQ_MAX is not None:
        mask &= freqs <= FREQ_MAX

    plot_freqs = freqs[mask]
    plot_values = values[mask]

    if plot_freqs.size == 0:
        raise ValueError("No PSD samples remain after applying frequency range.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_unscaled" if UNDO_SCALE_FOR_PSD else ""
    plot_path = PLOT_DIR / f"{DETECTOR}-{file_metric}-{VERSION}-{JOB_NUMBER:05d}{suffix}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(plot_freqs, plot_values, color="C0", linewidth=1.2, label=spectrum_label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{DETECTOR} noise {title_metric} | version={VERSION} | job={JOB_NUMBER:05d}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    info_lines = [
        f"source: {source_path.name}",
        f"sample_rate: {series.sample_rate.value:.3f} Hz",
        f"duration: {series.duration.value:.3f} s",
        f"fftlength: {FFT_LENGTH}",
        f"overlap: {OVERLAP}",
        f"window: {WINDOW}",
    ]
    if UNDO_SCALE_FOR_PSD:
        info_lines.append(f"undo scale: 1 / {NOISE_SCALE_FACTOR:g}")
    else:
        info_lines.append(f"scale kept: {NOISE_SCALE_FACTOR:g}")

    ax.text(
        0.02,
        0.02,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main() -> None:
    noise_path = build_noise_path()
    if not noise_path.exists():
        raise FileNotFoundError(f"Noise file not found: {noise_path}")

    print(f"[read] {noise_path}")
    series = load_noise_series(noise_path)
    print(
        "[info] "
        f"duration={series.duration.value:.3f}s "
        f"sample_rate={series.sample_rate.value:.3f}Hz "
        f"samples={len(series)}"
    )

    series_for_psd = maybe_rescale(series)
    plot_path = plot_spectrum(series_for_psd, noise_path)
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
