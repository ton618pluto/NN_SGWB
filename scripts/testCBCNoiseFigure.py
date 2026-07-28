from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "testCBCNoiseFigure.py requires 'gwpy'. Install it in the active environment first."
    ) from exc


POP_NUMS = 1
FRAME_NUMS = 1
#VERSION = "v61_1"
#VERSION = "v61_200"
#VERSION = "v61_random_two_noise_200"
VERSION = "v100_1"
folder_name="training_set_superimposed"
#folder_name="training_set_superimposed_test"

DETECTORS = ["H1", "L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
BANDPASS_LOW = 5
#BANDPASS_HIGH = 10
BANDPASS_HIGH = 170
APPLY_BANDPASS = False
ZOOM_WINDOW_SECONDS = 2048.0
USE_SHARED_Y_AXIS = False
FIGURE_SIZE = (17, 8)
PLOT_RIGHT_FRACTION = 0.72

SCRIPTS_DIR = Path(__file__).resolve().parent

SUPERIMPOSED_ROOT = SCRIPTS_DIR / folder_name / VERSION
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / "v100" / "training_set0"

#SUPERIMPOSED_ROOT = SCRIPTS_DIR / folder_name / ("fixed_one_test_" + VERSION)
#SIGNAL_ROOT = SCRIPTS_DIR / "training_set_test" / ("fixed_one_test_"+VERSION) / "training_set0"

PLOT_DIR = SCRIPTS_DIR / "plots_SGWB_noise_no_filter"


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


def show_superimposed() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for pop_num in range(POP_NUMS):
        for frame_num in range(FRAME_NUMS):
            data_by_detector: dict[str, tuple[TimeSeries, TimeSeries]] = {}
            missing_file = False

            for detector_name in DETECTORS:
                signal_dir = SIGNAL_ROOT / detector_name
                superimposed_dir = SUPERIMPOSED_ROOT / detector_name
                strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"
                signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
                superimposed_path = superimposed_dir / f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf"

                if not superimposed_path.exists():
                    print(f"[skip] Missing superimposed file: {superimposed_path}")
                    missing_file = True
                    continue
                if not signal_path.exists():
                    print(f"[skip] Missing signal file: {signal_path}")
                    missing_file = True
                    continue

                print(f"[read] superimposed: {superimposed_path}")
                print(f"[read] signal: {signal_path}")
                data_by_detector[detector_name] = (
                    TimeSeries.read(superimposed_path, strain_channel),
                    TimeSeries.read(signal_path, strain_channel),
                )

            if missing_file or len(data_by_detector) != len(DETECTORS):
                continue

            plot_data: dict[str, tuple[TimeSeries, TimeSeries]] = {}
            if APPLY_BANDPASS:
                for detector_name, (data_superimposed, data_signal) in data_by_detector.items():
                    plot_data[detector_name] = (
                        _safe_bandpass(data_superimposed, BANDPASS_LOW, BANDPASS_HIGH),
                        _safe_bandpass(data_signal, BANDPASS_LOW, BANDPASS_HIGH),
                    )
                plot_description = f"Filtered Waveform Comparison ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)"
                y_label = "Strain (Filtered)"
            else:
                plot_data = data_by_detector
                plot_description = "Full-Band Waveform Comparison (Unfiltered)"
                y_label = "Strain (Unfiltered)"

            fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=False)
            ax_full_signal = ax_full if USE_SHARED_Y_AXIS else ax_full.twinx()
            colors = {"H1": "C0", "L1": "C1"}  # Match testCBCFigure.py: H1 blue, L1 orange.

            for detector_name in DETECTORS:
                data_superimposed_plot, data_signal_plot = plot_data[detector_name]
                color = colors.get(detector_name, "C2")
                ax_full.plot(
                    data_superimposed_plot.times.value,
                    data_superimposed_plot.value,
                    label=f"{detector_name} Superimposed",
                    color=color,
                    alpha=0.35,
                    linewidth=0.5,
                )
                ax_full_signal.plot(
                    data_signal_plot.times.value,
                    data_signal_plot.value,
                    label=f"{detector_name} Pure Signal",
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                )
                ax_zoom.plot(
                    data_signal_plot.times.value,
                    data_signal_plot.value,
                    label=f"{detector_name} Pure Signal",
                    color=color,
                    linewidth=1.2,
                )

            reference_superimposed, _ = data_by_detector[DETECTORS[0]]
            reference_plot, _ = plot_data[DETECTORS[0]]
            duration = reference_superimposed.duration.value
            start_time = reference_plot.times.value[0]
            mid_time = reference_superimposed.times.min().value + duration / 2.0
            plot_path = PLOT_DIR / f"H1-L1-WAVEFORMS-pop{pop_num:05d}_sample{frame_num:05d}-comparison-{VERSION}.png"

            ax_full.set_xlim(start_time, start_time + duration)
            ax_full.axvline(mid_time, color="r", linestyle=":", linewidth=1.5, label="Approx. Injection Time")
            if USE_SHARED_Y_AXIS:
                ax_full.set_ylabel(y_label, fontsize=12)
            else:
                ax_full.set_ylabel(f"Superimposed {y_label}", fontsize=12)
                ax_full_signal.set_ylabel(f"Pure Signal {y_label}", fontsize=12)
            ax_full.set_xlabel("Time (s)", fontsize=12)
            ax_full.set_title(
                f"H1/L1 pop{pop_num:05d} sample{frame_num:05d}: {plot_description}",
                fontsize=16,
            )
            full_handles, full_labels = ax_full.get_legend_handles_labels()
            signal_handles, signal_labels = ax_full_signal.get_legend_handles_labels()
            fig.legend(
                full_handles + signal_handles,
                full_labels + signal_labels,
                loc="upper center",
                bbox_to_anchor=((1.0 + PLOT_RIGHT_FRACTION) / 2.0, 0.96),
            )

            half_window = ZOOM_WINDOW_SECONDS / 2.0
            zoom_start = max(start_time, mid_time - half_window)
            zoom_end = min(start_time + duration, mid_time + half_window)
            ax_zoom.set_xlim(zoom_start, zoom_end)
            ax_zoom.axvline(mid_time, color="r", linestyle=":", linewidth=1.5, label="Approx. Injection Time")
            ax_zoom.set_ylabel(y_label, fontsize=12)
            ax_zoom.set_xlabel("Time (s)", fontsize=12)
            ax_zoom.set_title(f"H1/L1 Pure Signal Zoom ({ZOOM_WINDOW_SECONDS:.1f} s window)", fontsize=13)
            plt.tight_layout(rect=(0, 0, PLOT_RIGHT_FRACTION, 1))
            plt.savefig(plot_path)
            plt.close(fig)

            print(f"[saved] {plot_path}")


def main() -> None:
    show_superimposed()
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
