from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIVIDE_FACTOR = 5.0
FREQ_MIN = 1.0
FREQ_MAX = None

SCRIPTS_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPTS_DIR / "O4_mock.txt"
PLOT_DIR = SCRIPTS_DIR / "plots_noise_psd"


def load_asd_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")

    freqs = table[:, 0]
    asd = table[:, 1]
    return freqs, asd


def apply_frequency_division(freqs: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0:
        raise ValueError("DIVIDE_FACTOR cannot be 0")
    return freqs / factor


def plot_asd_comparison(
    original_freqs: np.ndarray,
    shifted_freqs: np.ndarray,
    asd: np.ndarray,
    output_path: Path,
) -> None:
    mask_original = original_freqs >= FREQ_MIN
    mask_shifted = shifted_freqs >= FREQ_MIN
    if FREQ_MAX is not None:
        mask_original &= original_freqs <= FREQ_MAX
        mask_shifted &= shifted_freqs <= FREQ_MAX

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(
        original_freqs[mask_original],
        asd[mask_original],
        color="black",
        linewidth=1.4,
        label="Original ASD",
    )
    ax.loglog(
        shifted_freqs[mask_shifted],
        asd[mask_shifted],
        color="C1",
        linestyle="--",
        linewidth=1.4,
        label=f"ASD with frequency / {DIVIDE_FACTOR:g}",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ASD")
    ax.set_title(f"O4_mock ASD comparison | first column divided by {DIVIDE_FACTOR:g}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    info_lines = [
        f"source: {INPUT_PATH.name}",
        f"divide factor: {DIVIDE_FACTOR:g}",
        f"points: {len(asd)}",
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
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    original_freqs, asd = load_asd_table(INPUT_PATH)
    shifted_freqs = apply_frequency_division(original_freqs, DIVIDE_FACTOR)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"O4_mock_ASD_freq_divide_by_scale_{DIVIDE_FACTOR:g}.png"
    plot_asd_comparison(original_freqs, shifted_freqs, asd, output_path)

    print(f"[read] {INPUT_PATH}")
    print(f"[saved] {output_path}")
    print(f"[info] first original frequencies: {original_freqs[:5]}")
    print(f"[info] first shifted frequencies:  {shifted_freqs[:5]}")


if __name__ == "__main__":
    main()
