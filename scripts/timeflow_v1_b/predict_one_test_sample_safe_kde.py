from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from eval_utils import load_model_and_dataset


SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 2000
DATA_VERSION = "v8"
GPU_INDEX = 1


def _safe_kde_2d(x: np.ndarray, y: np.ndarray):
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return None, None, None
    x_grid = np.linspace(x.min(), x.max(), 160)
    y_grid = np.linspace(y.min(), y.max(), 160)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def main() -> None:
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(__file__).resolve().parent / f"outputs_dataset_{DATA_VERSION}"
    checkpoint_path = output_dir / "best_flow_v1_b.pt"
    model, dataset, run_config = load_model_and_dataset("test", checkpoint_path, device)
    parameter_names = list(run_config["parameter_names"])

    data, label_norm = dataset[SAMPLE_INDEX]
    data = data.unsqueeze(0).to(device)
    label_norm = label_norm.unsqueeze(0).to(device)

    with torch.no_grad():
        posterior_norm = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)

    posterior_norm = posterior_norm.squeeze(0).detach().cpu().numpy()
    label_true_norm = label_norm.squeeze(0).cpu().numpy()
    label_mean = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std = np.asarray(run_config["label_std"], dtype=np.float64)
    posterior = posterior_norm * label_std + label_mean
    label_true = label_true_norm * label_std + label_mean

    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(parameter_names)

    figure_dir = Path(__file__).resolve().parent / "result_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_sample_posterior_kde_v1_b_sample{SAMPLE_INDEX:05d}.png"

    fig, axes = plt.subplots(n_params, n_params, figsize=(3.2 * n_params, 3.2 * n_params))
    if n_params == 1:
        axes = np.asarray([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                ax.hist(samples, bins=40, density=True, color="tab:blue", alpha=0.7)
                ax.axvline(label_true[col], color="red", linestyle="-", linewidth=1.5)
                ax.axvline(means[col], color="black", linestyle="--", linewidth=1.2)
                ax.axvspan(q16[col], q84[col], color="tab:blue", alpha=0.15)
                ax.set_title(parameter_names[col])
                ax.text(
                    0.02,
                    0.98,
                    f"true={label_true[col]:.4f}\nmean={means[col]:.4f}\n68%=[{q16[col]:.4f}, {q84[col]:.4f}]",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                )
            elif row > col:
                x = posterior[:, col]
                y = posterior[:, row]
                xx, yy, zz = _safe_kde_2d(x, y)
                if zz is not None:
                    ax.contourf(xx, yy, zz, levels=14, cmap="Blues")
                    ax.contour(xx, yy, zz, levels=7, colors="k", linewidths=0.4, alpha=0.5)
                else:
                    ax.scatter(x, y, s=4, alpha=0.15, color="tab:blue")
                ax.scatter(label_true[col], label_true[row], color="red", marker="x", s=60, linewidths=2)
                ax.scatter(means[col], means[row], color="black", marker="o", s=18)
            else:
                ax.axis("off")

            if row == n_params - 1 and col <= row:
                ax.set_xlabel(parameter_names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(parameter_names[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])

    fig.suptitle(f"Posterior KDE corner plot (test sample, {DATA_VERSION})", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure: {figure_path}")
    print("Posterior summary:")
    for i, name in enumerate(parameter_names):
        in68 = "Y" if q16[i] <= label_true[i] <= q84[i] else "N"
        print(
            f"{name:<12} true={label_true[i]:.6f} mean={means[i]:.6f} "
            f"err={means[i]-label_true[i]:.6f} ci16={q16[i]:.6f} ci84={q84[i]:.6f} in68={in68}"
        )


if __name__ == "__main__":
    main()
