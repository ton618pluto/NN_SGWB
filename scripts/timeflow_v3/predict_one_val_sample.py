import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_utils import (
    CHECKPOINT_PATH,
    PARAMETER_NAMES,
    build_model,
    build_val_dataset,
    inverse_label_normalization,
    load_checkpoint,
    sample_posterior,
)


SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 4000
OUTPUT_DIR = Path(__file__).resolve().parent / "result_fig"
OUTPUT_NAME = "val_sample_posterior_corner_v3.png"


def plot_corner(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
    dim = samples.shape[1]
    names = PARAMETER_NAMES[:dim]

    fig, axes = plt.subplots(dim, dim, figsize=(18, 18))
    fig.patch.set_facecolor("white")

    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(samples[:, col], bins=45, density=True, color="#5b8ff9", alpha=0.82)
                ax.axvline(true_values[col], color="#e84a5f", linewidth=1.6)
                ax.axvline(samples[:, col].mean(), color="#1f2d3d", linewidth=1.2, linestyle="--")
            else:
                hist, x_edges, y_edges = np.histogram2d(
                    samples[:, col],
                    samples[:, row],
                    bins=32,
                    density=True,
                )
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                x_grid, y_grid = np.meshgrid(x_centers, y_centers)
                ax.contourf(x_grid, y_grid, hist.T, levels=6, cmap="Blues", alpha=0.9)
                ax.contour(x_grid, y_grid, hist.T, levels=6, colors="#355c7d", linewidths=0.55)
                ax.scatter(true_values[col], true_values[row], s=18, color="#e84a5f", marker="x", linewidths=1.4)

            if row == dim - 1:
                ax.set_xlabel(names[col], fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(names[row], fontsize=8)
            elif col != 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=6, length=2)

    fig.suptitle(
        "Posterior joint distribution for one v3 validation sample\n"
        "red = true label, dashed black = posterior mean",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_val_dataset(checkpoint)
    model = build_model(checkpoint, device)

    waveform, label_normalized = dataset[SAMPLE_INDEX]
    samples_normalized = sample_posterior(model, waveform, NUM_POSTERIOR_SAMPLES, device)

    label_true = inverse_label_normalization(label_normalized, checkpoint["label_mean"], checkpoint["label_std"])
    posterior_samples = inverse_label_normalization(samples_normalized, checkpoint["label_mean"], checkpoint["label_std"])
    posterior_mean = posterior_samples.mean(axis=0)

    print(f"Sample index: {SAMPLE_INDEX}")
    print(f"Waveform shape: {tuple(waveform.shape)}")
    print("True label:")
    for name, value in zip(PARAMETER_NAMES, label_true):
        print(f"  {name}: {value:.6g}")
    print("Posterior mean:")
    for name, value in zip(PARAMETER_NAMES, posterior_mean):
        print(f"  {name}: {value:.6g}")

    output_path = OUTPUT_DIR / OUTPUT_NAME
    plot_corner(posterior_samples, label_true, output_path)
    print(f"Saved posterior figure: {output_path}")


if __name__ == "__main__":
    main()
