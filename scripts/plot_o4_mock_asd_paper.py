from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


# ---------------------------- User controls ----------------------------
FREQUENCY_MIN_HZ = 10.0
FREQUENCY_MAX_HZ: float | None = None  # None keeps the full frequency range.

FIGURE_SIZE_INCHES = (7.4, 4.8)
OUTPUT_DPI = 600
CURVE_COLOR = "#59689A"

SCRIPTS_DIR = Path(__file__).resolve().parent
ASD_PATH = SCRIPTS_DIR / "O4_mock.txt"
OUTPUT_DIR = SCRIPTS_DIR / "paper_figures"
OUTPUT_PATH = OUTPUT_DIR / "O4_mock_ASD.pdf"


def load_asd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"ASD file not found: {path}")

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns (frequency, ASD), got shape={data.shape}"
        )

    frequency = data[:, 0]
    asd = data[:, 1]
    valid = np.isfinite(frequency) & np.isfinite(asd) & (frequency > 0.0) & (asd > 0.0)
    valid &= frequency >= FREQUENCY_MIN_HZ
    if FREQUENCY_MAX_HZ is not None:
        valid &= frequency <= FREQUENCY_MAX_HZ

    frequency = frequency[valid]
    asd = asd[valid]
    if frequency.size < 2:
        raise ValueError("Fewer than two valid ASD samples remain after filtering.")

    order = np.argsort(frequency)
    return frequency[order], asd[order]


def main() -> None:
    frequency, asd = load_asd(ASD_PATH)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.facecolor": "white",
        }
    )

    fig, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    axis.loglog(
        frequency,
        asd,
        color=CURVE_COLOR,
        linewidth=1.55,
        alpha=0.96,
        label="O4 mock sensitivity",
    )

    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(r"Strain ASD [$\mathrm{Hz}^{-1/2}$]")
    axis.set_xlim(float(frequency.min()), float(frequency.max()))

    # Add a small logarithmic vertical margin without imposing arbitrary limits.
    log_asd = np.log10(asd)
    log_margin = max(0.08, 0.05 * float(log_asd.max() - log_asd.min()))
    axis.set_ylim(10.0 ** (log_asd.min() - log_margin), 10.0 ** (log_asd.max() + log_margin))

    axis.xaxis.set_major_locator(LogLocator(base=10.0))
    axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())

    axis.grid(which="major", color="#B8BCC6", linestyle="--", linewidth=0.65, alpha=0.55)
    axis.grid(which="minor", color="#D8DAE0", linestyle=":", linewidth=0.45, alpha=0.38)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(which="major", colors="#3F3F3F", length=4.2, width=0.8)
    axis.tick_params(which="minor", colors="#666666", length=2.4, width=0.55)
    axis.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.93,
        handlelength=2.7,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.15, top=0.98)
    fig.savefig(OUTPUT_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"ASD source: {ASD_PATH}")
    print(f"Plotted points: {frequency.size}")
    print(f"Frequency range: {frequency.min():.6g}-{frequency.max():.6g} Hz")
    print(f"ASD range: {asd.min():.6e}-{asd.max():.6e} Hz^(-1/2)")
    print(f"Saved vector PDF figure ({OUTPUT_DPI} dpi for raster elements): {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
