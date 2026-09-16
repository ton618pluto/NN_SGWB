from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from gwpy.timeseries import TimeSeries
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "plot_v100_h1_signal_noise_paper.py requires gwpy. "
        "Run it in the LIGO_SGWB environment."
    ) from exc


# ---------------------------- User controls ----------------------------
VERSION = "v100"
SIGNAL_POP_INDEX = 0
SIGNAL_SAMPLE_INDEX = 0
NOISE_INDEX = 1

START_TIME_SECONDS = 0.0
DURATION_SECONDS = 2048.0

# Keep 1.0 to plot the raw noise file. If the paper needs the noise amplitude
# used by a 1/100 superposition experiment, change this value to 1 / 100.
NOISE_SCALE = 1.0

FIGURE_SIZE_INCHES = (10.0, 4.8)
OUTPUT_DPI = 500

# A 2048 s, 512 Hz frame contains more than one million points. The plotting
# downsampler preserves the minimum and maximum in each time block, so sharp
# excursions remain visible while the saved figure stays manageable.
MAX_PLOT_POINTS_PER_PANEL = 160_000

SIGNAL_COLOR = "#DA8983"  # muted soft red
NOISE_COLOR = "#9F8BC8"   # muted soft purple

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_PATH = (
    SCRIPTS_DIR
    / "training_set"
    / VERSION
    / "training_set0"
    / "H1"
    / f"H1-pop{SIGNAL_POP_INDEX:05d}_sample{SIGNAL_SAMPLE_INDEX:05d}.gwf"
)
NOISE_PATH = (
    SCRIPTS_DIR
    / "noise_waveform"
    / VERSION
    / "noise_waveform_H1"
    / f"H1-STRAIN-{NOISE_INDEX:05d}-duration.gwf"
)
CHANNEL = "H1:TEST_INJ"

OUTPUT_DIR = SCRIPTS_DIR / "paper_figures"
OUTPUT_PATH = OUTPUT_DIR / (
    f"H1_signal_noise_{VERSION}_0_{int(DURATION_SECONDS)}s.png"
)


def _read_relative_window(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not path.exists():
        raise FileNotFoundError(f"GWF file not found: {path}")

    series = TimeSeries.read(path, channel=CHANNEL)
    sample_rate = float(series.sample_rate.value)
    values = np.asarray(series.value, dtype=np.float64)

    start_index = int(round(start_seconds * sample_rate))
    sample_count = int(round(duration_seconds * sample_rate))
    end_index = start_index + sample_count
    if start_index < 0 or end_index > len(values):
        available_duration = len(values) / sample_rate
        raise ValueError(
            f"Requested [{start_seconds}, {start_seconds + duration_seconds}) s "
            f"from {path.name}, but its available relative duration is "
            f"0-{available_duration:.6f} s."
        )

    window = values[start_index:end_index] * float(scale)
    relative_time = np.arange(window.size, dtype=np.float64) / sample_rate
    relative_time += start_seconds
    return relative_time, window, sample_rate


def _minmax_downsample(
    time: np.ndarray,
    values: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dense traces while retaining extrema from every time block."""
    if max_points < 4:
        raise ValueError("MAX_PLOT_POINTS_PER_PANEL must be at least 4.")
    if values.size <= max_points:
        return time, values

    block_size = int(np.ceil(values.size / (max_points // 2)))
    full_size = (values.size // block_size) * block_size
    blocks = values[:full_size].reshape(-1, block_size)
    block_starts = np.arange(blocks.shape[0], dtype=np.int64) * block_size

    min_indices = block_starts + np.argmin(blocks, axis=1)
    max_indices = block_starts + np.argmax(blocks, axis=1)
    paired_indices = np.column_stack([min_indices, max_indices])
    paired_indices.sort(axis=1)
    selected = paired_indices.reshape(-1)

    if full_size < values.size:
        tail = values[full_size:]
        tail_indices = np.asarray(
            [full_size + int(np.argmin(tail)), full_size + int(np.argmax(tail))],
            dtype=np.int64,
        )
        selected = np.concatenate([selected, np.unique(np.sort(tail_indices))])

    return time[selected], values[selected]


def _style_axes(signal_axis: plt.Axes, noise_axis: plt.Axes) -> None:
    for axis in (signal_axis, noise_axis):
        axis.spines["top"].set_visible(False)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.yaxis.get_offset_text().set_size(8.5)

    signal_axis.spines["left"].set_color(SIGNAL_COLOR)
    signal_axis.spines["bottom"].set_color("#666666")
    signal_axis.spines["right"].set_visible(False)
    signal_axis.tick_params(axis="x", colors="#444444", labelsize=9, length=3.5)
    signal_axis.tick_params(axis="y", colors="#B76560", labelsize=9, length=3.5)
    signal_axis.yaxis.label.set_color("#B76560")
    signal_axis.yaxis.get_offset_text().set_color("#B76560")
    signal_axis.grid(
        axis="both", color="#D8D8D8", linestyle="--", linewidth=0.55, alpha=0.48
    )

    noise_axis.spines["left"].set_visible(False)
    noise_axis.spines["bottom"].set_visible(False)
    noise_axis.spines["right"].set_visible(True)
    noise_axis.spines["right"].set_color(NOISE_COLOR)
    noise_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    noise_axis.tick_params(axis="y", colors="#77659E", labelsize=9, length=3.5)
    noise_axis.yaxis.label.set_color("#77659E")
    noise_axis.yaxis.get_offset_text().set_color("#77659E")


def main() -> None:
    signal_time, signal, signal_rate = _read_relative_window(
        SIGNAL_PATH,
        start_seconds=START_TIME_SECONDS,
        duration_seconds=DURATION_SECONDS,
    )
    noise_time, noise, noise_rate = _read_relative_window(
        NOISE_PATH,
        start_seconds=START_TIME_SECONDS,
        duration_seconds=DURATION_SECONDS,
        scale=NOISE_SCALE,
    )

    signal_time_plot, signal_plot = _minmax_downsample(
        signal_time, signal, MAX_PLOT_POINTS_PER_PANEL
    )
    noise_time_plot, noise_plot = _minmax_downsample(
        noise_time, noise, MAX_PLOT_POINTS_PER_PANEL
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "savefig.facecolor": "white",
        }
    )

    fig, signal_axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    noise_axis = signal_axis.twinx()

    # Draw the signal axis above the dense noise trace so both remain visible.
    signal_axis.set_zorder(2)
    noise_axis.set_zorder(1)
    signal_axis.patch.set_visible(False)

    signal_axis.plot(
        signal_time_plot,
        signal_plot,
        color=SIGNAL_COLOR,
        linewidth=0.62,
        alpha=0.98,
        label="H1 CBC background signal",
        rasterized=True,
    )
    noise_axis.plot(
        noise_time_plot,
        noise_plot,
        color=NOISE_COLOR,
        linewidth=0.42,
        alpha=0.66,
        label="H1 O4-like detector noise",
        rasterized=True,
    )

    end_time = START_TIME_SECONDS + DURATION_SECONDS
    _style_axes(signal_axis, noise_axis)
    signal_axis.set_xlim(START_TIME_SECONDS, end_time)
    signal_axis.set_xlabel("Time [s]")
    signal_axis.set_ylabel("Signal strain")
    noise_axis.set_ylabel("Noise strain")

    signal_handles, signal_labels = signal_axis.get_legend_handles_labels()
    noise_handles, noise_labels = noise_axis.get_legend_handles_labels()
    signal_axis.legend(
        signal_handles + noise_handles,
        signal_labels + noise_labels,
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.92,
        handlelength=2.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.105, right=0.895, bottom=0.14, top=0.975)
    fig.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Signal file: {SIGNAL_PATH}")
    print(
        f"Signal: sample_rate={signal_rate:.3f} Hz, "
        f"window_samples={signal.size}, plotted_points={signal_plot.size}"
    )
    print(f"Noise file: {NOISE_PATH}")
    print(
        f"Noise: sample_rate={noise_rate:.3f} Hz, "
        f"window_samples={noise.size}, plotted_points={noise_plot.size}, "
        f"scale={NOISE_SCALE:g}"
    )
    print(f"Saved 500-dpi figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
