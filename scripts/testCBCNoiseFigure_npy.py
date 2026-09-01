from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "testCBCNoiseFigure_npy.py requires 'gwpy'. Install it in the active environment first."
    ) from exc


POP_NUMS = 1
FRAME_NUMS = 1
TARGET_POP_INDEX: int | None = 1
TARGET_SAMPLE_INDEX: int | None = 0
SIGNAL_SAMPLE_INDEX = 0
#VERSION = "v61_random_two_noise_200"  # this will be changed
#VERSION = "v61_random_two_noise_100"  # this will be changed
VERSION = "v100_random_two_noise_100"  # this will be changed
#SIGNAL_VERSION = "v61"  # this will be changed
SIGNAL_VERSION = "v100"  # this will be changed

DETECTORS = ["H1", "L1"]
SPLIT_INDEX = 0

#SAMPLE_RATE = 32.0  # this will be changed
SAMPLE_RATE = 512.0  # this will be changed

BANDPASS_LOW = 5
#BANDPASS_HIGH = 10  # this will be changed
BANDPASS_HIGH = 170  # this will be changed
APPLY_BANDPASS = False

ZOOM_WINDOW_SECONDS = 2048.0
USE_SHARED_Y_AXIS = True

SCRIPTS_DIR = Path(__file__).resolve().parent
SUPERIMPOSED_ROOT = SCRIPTS_DIR / "processed_data_superimposed" / VERSION
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / SIGNAL_VERSION / "training_set0"
PLOT_DIR = SCRIPTS_DIR / "plots_SGWB_noise_npy_no_filter"


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


def _load_npy_series(path: Path, sample_rate: float) -> TimeSeries:
    sample = np.load(path, allow_pickle=True).item()
    data = np.asarray(sample["data"], dtype=np.float32)
    return TimeSeries(data, sample_rate=sample_rate)


def iter_target_indices() -> list[tuple[int, int]]:
    if TARGET_POP_INDEX is not None and TARGET_SAMPLE_INDEX is not None:
        return [(TARGET_POP_INDEX, TARGET_SAMPLE_INDEX)]

    indices: list[tuple[int, int]] = []
    for pop_num in range(POP_NUMS):
        for frame_num in range(FRAME_NUMS):
            indices.append((pop_num, frame_num))
    return indices


def show_superimposed(detector_name: str) -> None:
    signal_dir = SIGNAL_ROOT / detector_name
    superimposed_dir = SUPERIMPOSED_ROOT / f"{detector_name}_splits"

    if not superimposed_dir.exists():
        raise FileNotFoundError(f"Superimposed directory not found: {superimposed_dir}")
    if not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for pop_num, frame_num in iter_target_indices():
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{SIGNAL_SAMPLE_INDEX:05d}.gwf"
            superimposed_path = superimposed_dir / (
                f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}_p{SPLIT_INDEX}.npy"
            )
            plot_path = PLOT_DIR / (
                f"{detector_name}-WAVEFORMS-pop{pop_num:05d}_sample{frame_num:05d}"
                f"-split{SPLIT_INDEX:02d}-comparison-{VERSION}.png"
            )

            if not superimposed_path.exists():
                print(f"[skip] Missing superimposed file: {superimposed_path}")
                continue
            if not signal_path.exists():
                print(f"[skip] Missing signal file: {signal_path}")
                continue

            print(f"[read] superimposed: {superimposed_path}")
            print(f"[read] signal: {signal_path}")

            data_superimposed = _load_npy_series(superimposed_path, SAMPLE_RATE)
            data_signal = TimeSeries.read(signal_path, f"{detector_name}:TEST_INJ")
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

            if APPLY_BANDPASS:
                data_superimposed_plot = _safe_bandpass(data_superimposed, BANDPASS_LOW, BANDPASS_HIGH)
                data_signal_plot = _safe_bandpass(data_signal, BANDPASS_LOW, BANDPASS_HIGH)
                plot_description = f"Filtered Waveform Comparison ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)"
                y_label = "Strain (Filtered)"
            else:
                data_superimposed_plot = data_superimposed
                data_signal_plot = data_signal
                plot_description = "Full-Band Waveform Comparison (Unfiltered)"
                y_label = "Strain (Unfiltered)"

            fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
            ax_full_signal = ax_full if USE_SHARED_Y_AXIS else ax_full.twinx()
            ax_full.plot(
                data_superimposed_plot.times.value,
                data_superimposed_plot.value,
                label="Superimposed (Signal + Noise)",
                color="C0",
                linewidth=0.5,
            )
            ax_full_signal.plot(
                data_signal_plot.times.value,
                data_signal_plot.value,
                label="Injected Signal (Pure)",
                color="C1",
                linestyle="--",
                linewidth=1.5,
            )

            duration = data_superimposed.duration.value
            start_time = data_superimposed_plot.times.value[0]
            mid_time = data_superimposed.times.min().value + duration / 2.0

            ax_full.set_xlim(start_time, start_time + duration)
            ax_full.axvline(mid_time, color="r", linestyle=":", linewidth=1.5, label="Approx. Injection Time")
            if USE_SHARED_Y_AXIS:
                ax_full.set_ylabel(y_label, fontsize=12)
            else:
                ax_full.set_ylabel(f"Superimposed {y_label}", fontsize=12, color="C0")
                ax_full_signal.set_ylabel(f"Pure Signal {y_label}", fontsize=12, color="C1")
            ax_full.set_xlabel("Time (s)", fontsize=12)
            ax_full.set_title(
                f"Event {detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}: "
                f"{plot_description}",
                fontsize=16,
            )
            if USE_SHARED_Y_AXIS:
                ax_full.legend(loc="upper right")
            else:
                ax_full.tick_params(axis="y", colors="C0")
                ax_full_signal.tick_params(axis="y", colors="C1")
                full_handles, full_labels = ax_full.get_legend_handles_labels()
                signal_handles, signal_labels = ax_full_signal.get_legend_handles_labels()
                ax_full.legend(full_handles + signal_handles, full_labels + signal_labels, loc="upper right")

            ax_zoom.plot(
                data_signal_plot.times.value,
                data_signal_plot.value,
                label="Injected Signal (Pure)",
                color="C1",
                linestyle="--",
                linewidth=1.5,
            )
            half_window = ZOOM_WINDOW_SECONDS / 2.0
            zoom_start = max(start_time, mid_time - half_window)
            zoom_end = min(start_time + duration, mid_time + half_window)
            ax_zoom.set_xlim(zoom_start, zoom_end)
            ax_zoom.axvline(mid_time, color="r", linestyle=":", linewidth=1.5, label="Approx. Injection Time")
            ax_zoom.set_ylabel(y_label, fontsize=12)
            ax_zoom.set_xlabel("Time (s)", fontsize=12)
            ax_zoom.set_title(f"Pure Signal Zoom Around Injection ({ZOOM_WINDOW_SECONDS:.1f} s window)", fontsize=13)
            ax_zoom.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)

            print(f"[saved] {plot_path}")


def main() -> None:
    for detector_name in DETECTORS:
        show_superimposed(detector_name)
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
