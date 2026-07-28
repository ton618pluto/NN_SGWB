from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "testCBCSignalNoiseFigure.py requires 'gwpy'. Install it in the active environment first."
    ) from exc


POP_NUMS = 1
FRAME_NUMS = 1
# VERSION = "v1"
# VERSION = "v61_200"
# VERSION = "v61_random_two_noise_200"
VERSION = "v100_1"
FOLDER_NAME = "training_set_superimposed"
# FOLDER_NAME = "training_set_superimposed_test"

DETECTORS = ["H1", "L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
BANDPASS_LOW = 5
BANDPASS_HIGH = 170

SCRIPTS_DIR = Path(__file__).resolve().parent

SUPERIMPOSED_ROOT = SCRIPTS_DIR / FOLDER_NAME / VERSION
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / "v100" / "training_set0"

# SUPERIMPOSED_ROOT = SCRIPTS_DIR / FOLDER_NAME / ("fixed_one_test_" + VERSION)
# SIGNAL_ROOT = SCRIPTS_DIR / "training_set_test" / ("fixed_one_test_" + VERSION) / "training_set0"

PLOT_DIR = SCRIPTS_DIR / "plots_CBC_signal_noise"


def _safe_bandpass(series: TimeSeries, low_hz: float, high_hz: float) -> TimeSeries:
    sample_rate = float(series.sample_rate.value)
    nyquist = sample_rate / 2.0
    safe_high = min(high_hz, nyquist * 0.95)

    if low_hz <= 0:
        raise ValueError(f"Bandpass low cutoff must be > 0, got {low_hz}")
    if low_hz >= safe_high:
        raise ValueError(
            f"Invalid bandpass for sample_rate={sample_rate:.3f} Hz: "
            f"low={low_hz} Hz, high={high_hz} Hz, usable_high<{safe_high:.3f} Hz"
        )

    if safe_high < high_hz:
        print(
            f"[warn] Adjusting high cutoff from {high_hz} Hz to {safe_high:.3f} Hz "
            f"for sample_rate={sample_rate:.3f} Hz",
        )

    return series.bandpass(low_hz, safe_high).crop(series.times.min(), series.times.max())


def show_signal_and_noise(detector_name: str) -> None:
    signal_dir = SIGNAL_ROOT / detector_name
    superimposed_dir = SUPERIMPOSED_ROOT / detector_name
    strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"

    if not superimposed_dir.exists():
        raise FileNotFoundError(f"Superimposed directory not found: {superimposed_dir}")
    if not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for pop_num in range(POP_NUMS):
        for frame_num in range(FRAME_NUMS):
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            superimposed_path = superimposed_dir / f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            plot_path = (
                PLOT_DIR
                / f"{detector_name}-SIGNAL-NOISE-pop{pop_num:05d}_sample{frame_num:05d}-{VERSION}.png"
            )

            if not superimposed_path.exists():
                print(f"[skip] Missing superimposed file: {superimposed_path}")
                continue
            if not signal_path.exists():
                print(f"[skip] Missing signal file: {signal_path}")
                continue

            print(f"[read] superimposed: {superimposed_path}")
            print(f"[read] signal: {signal_path}")

            data_superimposed = TimeSeries.read(superimposed_path, strain_channel)
            data_signal = TimeSeries.read(signal_path, strain_channel)

            if len(data_superimposed) != len(data_signal) or data_superimposed.dt != data_signal.dt:
                print(
                    f"[skip] Length or dt mismatch for {detector_name}-pop{pop_num:05d}_sample{frame_num:05d}"
                )
                continue

            data_noise = data_superimposed - data_signal

            print(
                "[info] superimposed: "
                f"duration={data_superimposed.duration.value:.3f}s "
                f"sample_rate={data_superimposed.sample_rate.value:.3f}Hz "
                f"samples={len(data_superimposed)}"
            )
            print(
                "[info] signal: "
                f"duration={data_signal.duration.value:.3f}s "
                f"sample_rate={data_signal.sample_rate.value:.3f}Hz "
                f"samples={len(data_signal)}"
            )

            data_signal_filtered = _safe_bandpass(data_signal, BANDPASS_LOW, BANDPASS_HIGH)
            data_noise_filtered = _safe_bandpass(data_noise, BANDPASS_LOW, BANDPASS_HIGH)

            fig, ax_signal = plt.subplots(figsize=(12, 6))
            ax_noise = ax_signal.twinx()
            ax_signal.set_zorder(2)
            ax_noise.set_zorder(1)
            ax_signal.patch.set_alpha(0.0)

            ax_noise.plot(
                data_noise_filtered.times.value,
                data_noise_filtered.value,
                label="Recovered Noise",
                color="C0",
                linewidth=0.6,
                alpha=0.85,
                zorder=1,
            )
            ax_signal.plot(
                data_signal_filtered.times.value,
                data_signal_filtered.value,
                label="Injected Signal (Pure)",
                color="C1",
                linestyle="--",
                linewidth=1.5,
                zorder=3,
            )

            duration = data_signal.duration.value
            start_time = data_signal_filtered.times.value[0]
            mid_time = data_signal.times.min().value + duration / 2.0

            ax_signal.set_xlim(start_time, start_time + duration)
            ax_signal.axvline(mid_time, color="r", linestyle=":", linewidth=1.5, label="Approx. Injection Time")
            ax_signal.set_ylabel("Signal Strain (Filtered)", fontsize=12, color="C1")
            ax_noise.set_ylabel("Noise Strain (Filtered)", fontsize=12, color="C0")
            ax_signal.set_xlabel("Time (s)", fontsize=12)
            ax_signal.set_title(
                f"Event {detector_name}-pop{pop_num:05d}_sample{frame_num:05d}: "
                f"Signal vs Noise ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)",
                fontsize=16,
            )
            ax_signal.tick_params(axis="y", colors="C1")
            ax_noise.tick_params(axis="y", colors="C0")
            signal_handles, signal_labels = ax_signal.get_legend_handles_labels()
            noise_handles, noise_labels = ax_noise.get_legend_handles_labels()
            ax_signal.legend(signal_handles + noise_handles, signal_labels + noise_labels, loc="upper right")

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)

            print(f"[saved] {plot_path}")


def main() -> None:
    for detector_name in DETECTORS:
        show_signal_and_noise(detector_name)
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
