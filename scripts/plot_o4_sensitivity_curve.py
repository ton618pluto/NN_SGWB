from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPTS_DIR / "O4_mock.txt"
OUTPUT_DIR = SCRIPTS_DIR / "sensitivity_fig"
OUTPUT_FILE = OUTPUT_DIR / "O4_sensitivity_curve.png"


def load_o4_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected file format: {path}")
    frequency = data[:, 0]
    amplitude = data[:, 1]
    return frequency, amplitude


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    frequency, amplitude = load_o4_curve(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))
    plt.loglog(frequency, amplitude, color="#1f77b4", linewidth=1.8, label="O4 mock")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain ASD [1/sqrt(Hz)]")
    plt.title("O4 Sensitivity Curve")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Input file: {INPUT_FILE}")
    print(f"Points loaded: {len(frequency)}")
    print(f"Frequency range: [{frequency.min():.6g}, {frequency.max():.6g}] Hz")
    print(f"Amplitude range: [{amplitude.min():.6g}, {amplitude.max():.6g}]")
    print(f"Saved figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
