import os
import warnings

import matplotlib.pyplot as plt
import gwpy.timeseries as ts
from gwpy.timeseries import TimeSeries


# Suppress gwpy user warnings to keep console output clean.
warnings.filterwarnings("ignore", category=UserWarning, module="gwpy")

# Configure matplotlib with generic fonts and ASCII-safe rendering.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- Configuration ---
NOISE_VERSION = "v99999"  # Noise dataset version under noise_waveform/.
JOB_NUMBER = 1  # Job index to visualize.
DETECTORS = ["H1", "L1"]  # Detector names to plot.
BASE_OUTPUT_DIR = "plots_noise"  # Output directory.
DURATION_TO_PLOT = 10.0  # Plot only the first 1.0 seconds.


if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)
    print(f"Created output directory: {BASE_OUTPUT_DIR}")


def visualize_h1_l1_time_series_comparison(job_number, output_dir):
    """
    Read H1 and L1 noise time-series data and plot them on the same figure.
    """

    all_strain_data = {}

    for detector_name in DETECTORS:
        detector_base_dir = os.path.join(
            ".", "noise_waveform", NOISE_VERSION, f"noise_waveform_{detector_name}"
        )
        file_name = f"{detector_name}-STRAIN-{job_number:05d}-duration.gwf"
        full_path = os.path.join(detector_base_dir, file_name)
        channel_name = f"{detector_name}:TEST_INJ"

        print(f"Reading {detector_name} file: {full_path}")

        try:
            full_strain = TimeSeries.read(full_path, channel=channel_name)

            num_samples = int(DURATION_TO_PLOT * full_strain.sample_rate.value)
            all_strain_data[detector_name] = full_strain[:num_samples]
            print(f"Loaded {detector_name} data successfully.")

        except Exception as e:
            print(f"Failed to read {detector_name} GWF file: {e}")
            print(f"Skipping visualization for {detector_name}.")
            return

    plt.figure(figsize=(12, 6))

    h1_data = all_strain_data["H1"]
    plt.plot(h1_data.times.value, h1_data.value, label="H1 Strain", alpha=0.7)

    l1_data = all_strain_data["L1"]
    plt.plot(l1_data.times.value, l1_data.value, label="L1 Strain", alpha=0.7)

    plt.xlabel("Time (s)")
    plt.ylabel(r"Strain (m/$m$)")
    plt.title(
        f"H1 and L1 Noise Time Series Comparison "
        f"(Job {job_number}, First {DURATION_TO_PLOT}s)"
    )
    plt.legend()
    plt.grid(True)

    comparison_fig_path = os.path.join(
        output_dir,
        f"H1_L1_Noise_TimeSeries_Comparison_{NOISE_VERSION}_{job_number}.png",
    )
    plt.savefig(comparison_fig_path)
    plt.close()

    print(f"\nSaved H1/L1 time-series comparison figure: {comparison_fig_path}")


visualize_h1_l1_time_series_comparison(JOB_NUMBER, BASE_OUTPUT_DIR)

print("\n=============== Done ===============")
