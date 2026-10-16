from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ---------------------------- User controls ----------------------------
DATA_VERSION = "v100_random_two_noise_100"
INITIAL_EPOCH = 0
INITIAL_NLL = 18.0
OUTPUT_DPI = 600
FIGURE_SIZE_INCHES = (6.2, 4.6)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    SCRIPT_DIR
    / "timeflow_v1_a"
    / f"outputs_dataset_{DATA_VERSION}"
    / "epoch_nll_history_v1_a.txt"
)
OUTPUT_DIR = SCRIPT_DIR / "paper_figures"
OUTPUT_PATH = OUTPUT_DIR / f"nll_training_history_{DATA_VERSION}.pdf"


def load_nll_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"NLL history file not found: {path}")

    history = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if history.size == 0:
        raise ValueError(f"NLL history file contains no data rows: {path}")

    history = np.atleast_1d(history)
    required_columns = {"epoch", "train_nll", "val_nll"}
    available_columns = set(history.dtype.names or ())
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(
            f"Missing columns {sorted(missing_columns)} in {path.name}; "
            f"available columns are {sorted(available_columns)}."
        )

    epochs = np.asarray(history["epoch"], dtype=np.float64)
    train_nll = np.asarray(history["train_nll"], dtype=np.float64)
    val_nll = np.asarray(history["val_nll"], dtype=np.float64)
    if not (
        np.all(np.isfinite(epochs))
        and np.all(np.isfinite(train_nll))
        and np.all(np.isfinite(val_nll))
    ):
        raise ValueError(f"NLL history contains non-finite values: {path}")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("Epoch values must be strictly increasing.")

    epochs = np.insert(epochs, 0, float(INITIAL_EPOCH))
    train_nll = np.insert(train_nll, 0, INITIAL_NLL)
    val_nll = np.insert(val_nll, 0, INITIAL_NLL)
    return epochs, train_nll, val_nll


def apply_axis_style(axis: plt.Axes) -> None:
    axis.grid(
        which="major",
        color="#C8CAD0",
        linestyle="--",
        linewidth=0.58,
        alpha=0.52,
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(colors="#3F3F3F", length=3.8, width=0.75)


def plot_nll_history(
    epochs: np.ndarray,
    train_nll: np.ndarray,
    val_nll: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "pdf.compression": 9,
            "ps.fonttype": 42,
        }
    )

    train_color = "#3F6F9F"
    validation_color = "#C45D52"
    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    axis.plot(
        epochs,
        train_nll,
        color=train_color,
        linewidth=1.75,
        marker="o",
        markersize=4.0,
        markeredgewidth=0.0,
        label="Training NLL",
        zorder=3,
    )
    axis.plot(
        epochs,
        val_nll,
        color=validation_color,
        linewidth=1.75,
        marker="s",
        markersize=3.8,
        markeredgewidth=0.0,
        label="Validation NLL",
        zorder=3,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative log-likelihood (NLL)")
    axis.set_xlim(float(np.min(epochs)) - 0.15, float(np.max(epochs)) + 0.25)
    all_nll = np.concatenate((train_nll, val_nll))
    axis.set_ylim(float(np.min(all_nll)) - 0.40, float(np.max(all_nll)) + 0.80)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    axis.legend(loc="upper right", frameon=False, handlelength=2.4)
    apply_axis_style(axis)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.14, right=0.975, bottom=0.14, top=0.97)
    figure.savefig(
        OUTPUT_PATH,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


def main() -> None:
    epochs, train_nll, val_nll = load_nll_history(INPUT_PATH)
    plot_nll_history(epochs, train_nll, val_nll)

    recorded_epochs = epochs[1:]
    recorded_train_nll = train_nll[1:]
    recorded_val_nll = val_nll[1:]
    best_validation_index = int(np.argmin(recorded_val_nll))

    print(f"Input NLL history: {INPUT_PATH}")
    print(f"Inserted initial point: epoch={INITIAL_EPOCH}, NLL={INITIAL_NLL:g}")
    print(f"Recorded epochs: {len(recorded_epochs)}")
    print(f"Final training NLL: {recorded_train_nll[-1]:.8f}")
    print(f"Final validation NLL: {recorded_val_nll[-1]:.8f}")
    print(
        "Best validation NLL: "
        f"{recorded_val_nll[best_validation_index]:.8f} "
        f"at epoch {int(recorded_epochs[best_validation_index])}"
    )
    print(f"Saved 600-dpi paper PDF: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
