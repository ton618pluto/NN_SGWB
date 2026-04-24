import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from eval_utils import (
        CHECKPOINT_PATH,
        PARAMETER_NAMES,
        build_model,
        build_split_dataset,
        inverse_label_normalization,
        load_checkpoint,
        sample_posterior,
    )
else:
    from .eval_utils import (
        CHECKPOINT_PATH,
        PARAMETER_NAMES,
        build_model,
        build_split_dataset,
        inverse_label_normalization,
        load_checkpoint,
        sample_posterior,
    )


SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 1200
OUTPUT_DIR = Path(__file__).resolve().parent / "result_fig"
OUTPUT_NAME_TEMPLATE = "test_sample_posterior_kde_v8_sample{sample_index:05d}.png"


def draw_kde_contour(ax, x: np.ndarray, y: np.ndarray) -> None:
    values = np.vstack([x, y])
    try:
        kde = gaussian_kde(values)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_pad = max((x_max - x_min) * 0.08, 1e-6)
        y_pad = max((y_max - y_min) * 0.08, 1e-6)
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 60)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 60)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        density = kde(positions).reshape(grid_x.shape)
        ax.contourf(grid_x, grid_y, density, levels=6, cmap="Greens", alpha=0.9)
        ax.contour(grid_x, grid_y, density, levels=6, colors="#1f7a5c", linewidths=0.55)
    except Exception:
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=28, density=True)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)
        ax.contourf(grid_x, grid_y, hist.T, levels=6, cmap="Greens", alpha=0.9)
        ax.contour(grid_x, grid_y, hist.T, levels=6, colors="#1f7a5c", linewidths=0.55)


def _format_stat_value(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 100 or (0 < abs_value < 0.01):
        return f"{value:.3e}"
    return f"{value:.4f}"


def summarize_posterior(samples: np.ndarray, true_values: np.ndarray) -> list[dict[str, float]]:
    summaries = []
    for index in range(samples.shape[1]):
        param_samples = samples[:, index]
        q16, q50, q84 = np.percentile(param_samples, [16, 50, 84])
        mean_value = float(param_samples.mean())
        true_value = float(true_values[index])
        summaries.append(
            {
                "true": true_value,
                "mean": mean_value,
                "median": float(q50),
                "err": mean_value - true_value,
                "ci_low": float(q16),
                "ci_high": float(q84),
                "covered_68": float(q16) <= true_value <= float(q84),
            }
        )
    return summaries


def print_posterior_summary(parameter_names: list[str], posterior_stats: list[dict[str, float]]) -> None:
    header = (
        f"{'param':<12}"
        f"{'true':>14}"
        f"{'mean':>14}"
        f"{'err':>14}"
        f"{'ci16':>14}"
        f"{'ci84':>14}"
        f"{'in68':>8}"
    )
    print("Posterior summary:")
    print(header)
    print("-" * len(header))
    for name, stat in zip(parameter_names, posterior_stats):
        print(
            f"{name:<12}"
            f"{_format_stat_value(stat['true']):>14}"
            f"{_format_stat_value(stat['mean']):>14}"
            f"{_format_stat_value(stat['err']):>14}"
            f"{_format_stat_value(stat['ci_low']):>14}"
            f"{_format_stat_value(stat['ci_high']):>14}"
            f"{('Y' if stat['covered_68'] else 'N'):>8}"
        )


def plot_corner_kde(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
    dim = samples.shape[1]
    names = PARAMETER_NAMES[:dim]
    posterior_stats = summarize_posterior(samples, true_values)

    fig, axes = plt.subplots(dim, dim, figsize=(12, 12))
    fig.patch.set_facecolor("white")

    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(samples[:, col], bins=40, density=True, color="#61d9a8", alpha=0.82)
                stat = posterior_stats[col]
                ax.axvspan(stat["ci_low"], stat["ci_high"], color="#1f7a5c", alpha=0.12)
                ax.axvline(stat["true"], color="#e84a5f", linewidth=1.5)
                ax.axvline(stat["mean"], color="#1f2d3d", linewidth=1.2, linestyle="--")
                annotation_lines = [
                    f"true={_format_stat_value(stat['true'])}",
                    f"mean={_format_stat_value(stat['mean'])}",
                    f"err={_format_stat_value(stat['err'])}",
                    f"68%=[{_format_stat_value(stat['ci_low'])}, {_format_stat_value(stat['ci_high'])}]",
                ]
                ax.text(
                    0.03,
                    0.97,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6,
                    color="#1f2d3d",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b7c9c0", "alpha": 0.9},
                )
            else:
                draw_kde_contour(ax, samples[:, col], samples[:, row])
                ax.scatter(true_values[col], true_values[row], s=16, color="#e84a5f", marker="x", linewidths=1.2)

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
        "Posterior joint distribution (KDE) for one v8 test sample\n"
        "red = true label, dashed black = posterior mean, shaded band = 68% interval",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_split_dataset(checkpoint, split_name="test")
    model = build_model(checkpoint, device)

    waveform, label_normalized = dataset[SAMPLE_INDEX]
    samples_normalized = sample_posterior(model, waveform, NUM_POSTERIOR_SAMPLES, device)

    label_true = inverse_label_normalization(label_normalized, checkpoint["label_mean"], checkpoint["label_std"])
    posterior_samples = inverse_label_normalization(samples_normalized, checkpoint["label_mean"], checkpoint["label_std"])
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_stats = summarize_posterior(posterior_samples, label_true)

    print(f"Sample index: {SAMPLE_INDEX}")
    print(f"Waveform shape: {tuple(waveform.shape)}")
    print("True label:")
    for name, value in zip(PARAMETER_NAMES, label_true):
        print(f"  {name}: {value:.6g}")
    print("Posterior mean:")
    for name, value in zip(PARAMETER_NAMES, posterior_mean):
        print(f"  {name}: {value:.6g}")
    print_posterior_summary(PARAMETER_NAMES, posterior_stats)

    output_path = OUTPUT_DIR / OUTPUT_NAME_TEMPLATE.format(sample_index=SAMPLE_INDEX)
    plot_corner_kde(posterior_samples, label_true, output_path)
    print(f"Saved KDE posterior figure: {output_path}")


if __name__ == "__main__":
    main()
