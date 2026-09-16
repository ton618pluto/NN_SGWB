from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import bilby
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------- User controls ----------------------------
# Match the frequency range used by the 512-Hz waveform-generation scripts.
FREQUENCY_MIN_HZ = 10.0
FREQUENCY_MAX_HZ = 256.0
DETECTORS = ("H1", "L1")

# Numerical-equality tolerances. These do not affect the plotted differences.
ALLCLOSE_RTOL = 1.0e-10
ALLCLOSE_ATOL = 0.0

OUTPUT_DPI = 500
FIGURE_SIZE_INCHES = (7.4, 6.2)

SCRIPT_DIR = Path(__file__).resolve().parent
O4_ASD_PATH = SCRIPT_DIR / "O4_mock.txt"
OUTPUT_DIR = SCRIPT_DIR / "paper_figures"
FIGURE_PATH = OUTPUT_DIR / "bilby_default_vs_O4_mock_ASD.png"
REPORT_PATH = OUTPUT_DIR / "bilby_default_vs_O4_mock_ASD.txt"


def load_o4_asd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"O4 ASD file not found: {path}")

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected two columns (frequency, ASD) in {path}, got {data.shape}."
        )

    frequency = data[:, 0]
    asd = data[:, 1]
    valid = (
        np.isfinite(frequency)
        & np.isfinite(asd)
        & (frequency > 0.0)
        & (asd > 0.0)
        & (frequency >= FREQUENCY_MIN_HZ)
        & (frequency <= FREQUENCY_MAX_HZ)
    )
    frequency = frequency[valid]
    asd = asd[valid]
    if frequency.size < 2:
        raise ValueError("Fewer than two valid O4 ASD points remain after filtering.")

    order = np.argsort(frequency)
    frequency = frequency[order]
    asd = asd[order]
    unique_frequency, unique_indices = np.unique(frequency, return_index=True)
    return unique_frequency, asd[unique_indices]


def describe_source(psd: object) -> tuple[str, str]:
    asd_file = getattr(psd, "asd_file", None)
    psd_file = getattr(psd, "psd_file", None)
    return str(asd_file), str(psd_file)


def compare_detector(
    detector: str, frequency: np.ndarray, o4_asd: np.ndarray
) -> dict[str, object]:
    interferometer = bilby.gw.detector.get_empty_interferometer(detector)
    default_psd = interferometer.power_spectral_density
    default_psd_values = np.asarray(
        default_psd.get_power_spectral_density_array(frequency), dtype=np.float64
    )
    default_asd = np.sqrt(default_psd_values)

    valid = np.isfinite(default_asd) & (default_asd > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            f"{detector}: Bilby returned fewer than two finite default-ASD values."
        )

    compared_frequency = frequency[valid]
    compared_o4_asd = o4_asd[valid]
    compared_default_asd = default_asd[valid]
    signed_relative = (compared_default_asd - compared_o4_asd) / compared_o4_asd
    absolute_relative = np.abs(signed_relative)
    maximum_index = int(np.argmax(absolute_relative))
    asd_file, psd_file = describe_source(default_psd)

    return {
        "detector": detector,
        "frequency": compared_frequency,
        "o4_asd": compared_o4_asd,
        "default_asd": compared_default_asd,
        "signed_relative": signed_relative,
        "asd_file": asd_file,
        "psd_file": psd_file,
        "exactly_equal": bool(np.array_equal(compared_default_asd, compared_o4_asd)),
        "allclose": bool(
            np.allclose(
                compared_default_asd,
                compared_o4_asd,
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
            )
        ),
        "mean_absolute_relative": float(np.mean(absolute_relative)),
        "median_absolute_relative": float(np.median(absolute_relative)),
        "rms_relative": float(np.sqrt(np.mean(signed_relative**2))),
        "maximum_absolute_relative": float(absolute_relative[maximum_index]),
        "maximum_difference_frequency": float(compared_frequency[maximum_index]),
    }


def make_report(results: list[dict[str, object]]) -> str:
    lines = [
        "Bilby default PSD versus O4_mock.txt",
        "=" * 44,
        f"Bilby version: {bilby.__version__}",
        f"O4 ASD file: {O4_ASD_PATH}",
        f"Requested frequency range: {FREQUENCY_MIN_HZ:g}-{FREQUENCY_MAX_HZ:g} Hz",
        f"allclose tolerances: rtol={ALLCLOSE_RTOL:.3e}, atol={ALLCLOSE_ATOL:.3e}",
    ]

    for result in results:
        frequency = result["frequency"]
        lines.extend(
            [
                "",
                f"Detector: {result['detector']}",
                f"Bilby default ASD file: {result['asd_file']}",
                f"Bilby default PSD file: {result['psd_file']}",
                f"Compared points: {len(frequency)}",
                f"Actual frequency range: {frequency[0]:.9g}-{frequency[-1]:.9g} Hz",
                f"Exactly equal: {result['exactly_equal']}",
                f"Numerically allclose: {result['allclose']}",
                "Relative difference definition: (default ASD - O4 ASD) / O4 ASD",
                f"Mean absolute relative difference: {result['mean_absolute_relative']:.12e} "
                f"({100.0 * result['mean_absolute_relative']:.12e} %)",
                f"Median absolute relative difference: {result['median_absolute_relative']:.12e} "
                f"({100.0 * result['median_absolute_relative']:.12e} %)",
                f"RMS relative difference: {result['rms_relative']:.12e} "
                f"({100.0 * result['rms_relative']:.12e} %)",
                f"Maximum absolute relative difference: {result['maximum_absolute_relative']:.12e} "
                f"({100.0 * result['maximum_absolute_relative']:.12e} %)",
                f"Frequency of maximum difference: "
                f"{result['maximum_difference_frequency']:.9g} Hz",
            ]
        )
    return "\n".join(lines) + "\n"


def plot_results(results: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.facecolor": "white",
        }
    )

    fig, (axis_asd, axis_difference) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE_INCHES,
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
    )
    colors = {"H1": "#315A8A", "L1": "#2F7D6B"}

    # O4_mock is common to both detectors, so draw it only once.
    first = results[0]
    axis_asd.loglog(
        first["frequency"],
        first["o4_asd"],
        color="#B44C52",
        linewidth=2.1,
        alpha=0.90,
        label="O4_mock.txt",
        zorder=2,
    )

    for result in results:
        detector = str(result["detector"])
        color = colors.get(detector, "#555555")
        axis_asd.loglog(
            result["frequency"],
            result["default_asd"],
            color=color,
            linewidth=1.25,
            alpha=0.95,
            linestyle="--",
            label=f"Bilby default ({detector})",
            zorder=3,
        )
        axis_difference.semilogx(
            result["frequency"],
            100.0 * result["signed_relative"],
            color=color,
            linewidth=1.25,
            alpha=0.95,
            label=detector,
        )

    axis_asd.set_ylabel(r"Strain ASD [$\mathrm{Hz}^{-1/2}$]")
    axis_difference.set_xlabel("Frequency [Hz]")
    axis_difference.set_ylabel("Difference [%]")
    axis_difference.axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
    axis_difference.set_xlim(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ)

    for axis in (axis_asd, axis_difference):
        axis.grid(which="major", color="#B8BCC6", linestyle="--", linewidth=0.65, alpha=0.52)
        axis.grid(which="minor", color="#D8DAE0", linestyle=":", linewidth=0.42, alpha=0.34)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(which="major", colors="#3F3F3F", length=4.0, width=0.8)
        axis.tick_params(which="minor", colors="#666666", length=2.3, width=0.55)

    axis_asd.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.93,
    )
    axis_difference.legend(
        loc="best",
        ncol=max(1, len(results)),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.93,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.105, top=0.98)
    fig.savefig(FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    frequency, o4_asd = load_o4_asd(O4_ASD_PATH)
    results = [compare_detector(name, frequency, o4_asd) for name in DETECTORS]
    report = make_report(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    plot_results(results)

    print(report, end="")
    print(f"Saved 500-dpi comparison figure: {FIGURE_PATH}")
    print(f"Saved numerical comparison report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
