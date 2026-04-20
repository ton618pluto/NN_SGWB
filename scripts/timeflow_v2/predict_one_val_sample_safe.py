import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import GWDatasetV2
from model import GWFlowModelV2


# =========================
# 直接在这里改推理配置
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

CHECKPOINT_PATH = SCRIPTS_ROOT / "timeflow_v2" / "outputs" / "best_flow_v2.pt"

VAL_ROOT = SCRIPTS_ROOT / "processed_data_val" / "v1"
VAL_H1_DIR = VAL_ROOT / "H1_splits"
VAL_L1_DIR = VAL_ROOT / "L1_splits"

SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 800

CONTEXT_DIM = 256
FLOW_LAYERS = 6
FLOW_HIDDEN = 256

OUTPUT_DIR = SCRIPTS_ROOT / "timeflow_v2" / "result_fig"
OUTPUT_NAME = "val_sample_posterior_corner_safe.png"

PARAMETER_NAMES = [
    "zp",
    "alpha_z",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def build_dataset(checkpoint: dict) -> GWDatasetV2:
    if not VAL_H1_DIR.exists():
        raise FileNotFoundError(f"Validation H1 directory not found: {VAL_H1_DIR}")
    if not VAL_L1_DIR.exists():
        raise FileNotFoundError(f"Validation L1 directory not found: {VAL_L1_DIR}")

    return GWDatasetV2(
        h1_dir=VAL_H1_DIR,
        l1_dir=VAL_L1_DIR,
        label_mean=checkpoint["label_mean"].cpu(),
        label_std=checkpoint["label_std"].cpu(),
        sample_step=1,
        dataset_name="val_predict",
    )


def build_model(checkpoint: dict, device: torch.device) -> GWFlowModelV2:
    model = GWFlowModelV2(
        param_dim=int(checkpoint["param_dim"]),
        in_channels=int(checkpoint["num_channels"]),
        context_dim=CONTEXT_DIM,
        flow_layers=FLOW_LAYERS,
        flow_hidden_features=FLOW_HIDDEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def inverse_label_normalization(values: torch.Tensor | np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    mean_array = mean.detach().cpu().numpy()
    std_array = std.detach().cpu().numpy()
    return values * std_array + mean_array


@torch.inference_mode()
def sample_posterior(model: GWFlowModelV2, waveform: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    return samples


def plot_corner(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
    dim = samples.shape[1]
    names = PARAMETER_NAMES[:dim]

    fig, axes = plt.subplots(dim, dim, figsize=(18, 18))

    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(samples[:, col], bins=40, density=True, color="#5b8ff9", alpha=0.82)
                ax.axvline(true_values[col], color="#e84a5f", linewidth=1.5)
                ax.axvline(samples[:, col].mean(), color="#1f2d3d", linewidth=1.2, linestyle="--")
            else:
                # 非对角线位置使用二维等高线图来表示联合后验分布。
                # 做法是先用 histogram2d 统计二维密度，再用 contourf / contour 绘制。
                hist, x_edges, y_edges = np.histogram2d(
                    samples[:, col],
                    samples[:, row],
                    bins=32,
                    density=True,
                )
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                X, Y = np.meshgrid(x_centers, y_centers)

                ax.contourf(X, Y, hist.T, levels=6, cmap="Blues", alpha=0.9)
                ax.contour(X, Y, hist.T, levels=6, colors="#355c7d", linewidths=0.55)

                # 红色 x：真实标签在二维参数空间中的位置。
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

    fig.suptitle("Posterior joint distribution for one validation sample", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation H1: {VAL_H1_DIR}")
    print(f"Validation L1: {VAL_L1_DIR}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_dataset(checkpoint)
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
