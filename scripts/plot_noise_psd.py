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
SPECTRUM_MODE = "ASD"  # "ASD" or "PSD"
# PSD1 is calculated after dividing the time-domain noise by this factor and
# multiplying it back. PSD2 is calculated directly from the original noise.
SCALE_DOWN_FACTOR = 1000.0

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


def scale_down_then_restore(series: TimeSeries) -> TimeSeries:
    if SCALE_DOWN_FACTOR == 0:
        raise ValueError("SCALE_DOWN_FACTOR cannot be 0")
    return TimeSeries(
        (series.value / SCALE_DOWN_FACTOR) * SCALE_DOWN_FACTOR,
        times=series.times,
        channel=series.channel,
        name=series.name,
        unit=series.unit,
    )


def plot_spectrum_comparison(original: TimeSeries, source_path: Path) -> Path:
    restored = scale_down_then_restore(original)
    psd1 = restored.psd(fftlength=FFT_LENGTH, overlap=OVERLAP, window=WINDOW)
    psd2 = original.psd(fftlength=FFT_LENGTH, overlap=OVERLAP, window=WINDOW)

    freqs = psd2.frequencies.value
    mode = SPECTRUM_MODE.upper()
    if mode == "ASD":
        values1 = np.sqrt(psd1.value)
        values2 = np.sqrt(psd2.value)
    elif mode == "PSD":
        values1 = psd1.value
        values2 = psd2.value
    else:
        raise ValueError(f"Unsupported SPECTRUM_MODE: {SPECTRUM_MODE}")

    mask = freqs >= FREQ_MIN
    if FREQ_MAX is not None:
        mask &= freqs <= FREQ_MAX

    plot_freqs = freqs[mask]
    plot_values1 = values1[mask]
    plot_values2 = values2[mask]

    if plot_freqs.size == 0:
        raise ValueError("No PSD samples remain after applying frequency range.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / (
        f"{DETECTOR}-{mode}-scale_down_restore-{SCALE_DOWN_FACTOR:g}-"
        f"{VERSION}-{JOB_NUMBER:05d}.png"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(plot_freqs, plot_values2, color="black", linewidth=1.4, label=f"{mode}2: Original noise")
    ax.loglog(
        plot_freqs,
        plot_values1,
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label=f"{mode}1: /{SCALE_DOWN_FACTOR:g} then x{SCALE_DOWN_FACTOR:g}",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(mode)
    ax.set_title(
        f"{DETECTOR} noise {mode} comparison | version={VERSION} | job={JOB_NUMBER:05d}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    info_lines = [
        f"source: {source_path.name}",
        f"sample_rate: {original.sample_rate.value:.3f} Hz",
        f"duration: {original.duration.value:.3f} s",
        f"fftlength: {FFT_LENGTH}",
        f"overlap: {OVERLAP}",
        f"window: {WINDOW}",
        f"scale down factor: {SCALE_DOWN_FACTOR:g}",
    ]

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

    max_abs_diff = float(np.max(np.abs(values1 - values2)))
    max_relative_diff = float(
        np.max(np.abs(values1 - values2) / np.maximum(np.abs(values2), np.finfo(float).tiny))
    )
    print(f"[compare] {mode}1 vs {mode}2 max abs difference: {max_abs_diff:.8e}")
    print(f"[compare] {mode}1 vs {mode}2 max relative difference: {max_relative_diff:.8e}")
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

    plot_path = plot_spectrum_comparison(series, noise_path)
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
