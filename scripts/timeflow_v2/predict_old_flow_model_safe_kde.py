import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from oldModel import OldGWFlowModel  # noqa: E402


# =========================
# 直接在这里改推理配置
# =========================
CHECKPOINT_PATH = CURRENT_DIR / "outputs" / "best_flow_model.pth"

TRAIN_H1_DIR = SCRIPTS_ROOT / "processed_data_train" / "v2" / "training_set0" / "H1_splits"
VAL_ROOT = SCRIPTS_ROOT / "processed_data_val" / "v1"
VAL_H1_DIR = VAL_ROOT / "H1_splits"
VAL_L1_DIR = VAL_ROOT / "L1_splits"

SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 1200

OUTPUT_DIR = CURRENT_DIR / "result_fig"
OUTPUT_NAME = "old_flow_model_val_posterior_kde.png"
OLD_STATS_CACHE_NAME = "old_model_label_stats_cache.npz"

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

SCALE_FACTOR = 1e23
PARAM_DIM = 10


def filter_label(label: np.ndarray) -> np.ndarray:
    return np.concatenate([label[:2], label[3:]]).astype(np.float32, copy=False)


def sample_key(path: Path) -> str:
    return path.stem.split("-", 1)[-1]


def preprocess_old_channel(data: np.ndarray) -> np.ndarray:
    # 对齐 origin_model/GWDataset.py：数值放大后逐通道去均值。
    data = np.asarray(data, dtype=np.float32) * SCALE_FACTOR
    return data - data.mean(dtype=np.float32)


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_dict = np.load(path, allow_pickle=True).item()
    data = preprocess_old_channel(sample_dict["data"])
    label = filter_label(np.asarray(sample_dict["label"], dtype=np.float32))
    return data, label


def collect_label_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    file_list = sorted(path for path in data_dir.iterdir() if path.suffix == ".npy")
    labels = []
    progress = tqdm(file_list, desc="Collecting old-model label stats", unit="file")
    for path in progress:
        _, label = load_sample(path)
        labels.append(label)

    all_labels = np.stack(labels, axis=0)
    mean = all_labels.mean(axis=0)
    std = all_labels.std(axis=0)
    std[std == 0] = 1.0
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def load_or_create_label_stats_cache(data_dir: Path, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        cache = np.load(cache_path)
        mean = torch.tensor(cache["mean"], dtype=torch.float32)
        std = torch.tensor(cache["std"], dtype=torch.float32)
        print(f"Loaded cached old-model label stats: {cache_path}")
        return mean, std

    print(f"Stats cache not found, calculating from: {data_dir}")
    mean, std = collect_label_stats(data_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mean=mean.numpy(), std=std.numpy())
    print(f"Saved old-model label stats cache: {cache_path}")
    return mean, std


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def infer_in_channels(state_dict: dict) -> int:
    return int(state_dict["embedding_net.conv_layers.0.weight"].shape[1])


def infer_context_dim(state_dict: dict) -> int:
    return int(state_dict["embedding_net.fc.3.weight"].shape[0])


def build_model(state_dict: dict, device: torch.device) -> tuple[OldGWFlowModel, int, int]:
    in_channels = infer_in_channels(state_dict)
    context_dim = infer_context_dim(state_dict)
    model = OldGWFlowModel(param_dim=PARAM_DIM, context_dim=context_dim, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, in_channels, context_dim


def inverse_label_normalization(values: torch.Tensor | np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return values * std.detach().cpu().numpy() + mean.detach().cpu().numpy()


@torch.inference_mode()
def sample_posterior(model: OldGWFlowModel, waveform: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    return samples


def draw_kde_contour(ax, x: np.ndarray, y: np.ndarray) -> None:
    # 旧模型的二维联合后验也改成 KDE 平滑等高线图，便于与新模型统一风格。
    values = np.vstack([x, y])
    try:
        kde = gaussian_kde(values)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_pad = max((x_max - x_min) * 0.08, 1e-6)
        y_pad = max((y_max - y_min) * 0.08, 1e-6)

        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 60)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 60)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

        ax.contourf(X, Y, Z, levels=6, cmap="Blues", alpha=0.9)
        ax.contour(X, Y, Z, levels=6, colors="#355c7d", linewidths=0.55)
    except Exception:
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=28, density=True)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)
        ax.contourf(X, Y, hist.T, levels=6, cmap="Blues", alpha=0.9)
        ax.contour(X, Y, hist.T, levels=6, colors="#355c7d", linewidths=0.55)


def plot_corner_kde(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
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
                ax.hist(samples[:, col], bins=40, density=True, color="#61d9a8", alpha=0.82)
                ax.axvline(true_values[col], color="#e84a5f", linewidth=1.5)
                ax.axvline(samples[:, col].mean(), color="#1f2d3d", linewidth=1.2, linestyle="--")
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
        "Posterior joint distribution (KDE version, old model)\n"
        "red = true label, dashed black = posterior mean",
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
    print(f"Validation H1: {VAL_H1_DIR}")
    print(f"Validation L1: {VAL_L1_DIR}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    model, in_channels, context_dim = build_model(checkpoint, device)
    print(f"Inferred input channels from checkpoint: {in_channels}")
    print(f"Inferred context dim from checkpoint: {context_dim}")

    stats_dir = TRAIN_H1_DIR if TRAIN_H1_DIR.exists() else VAL_H1_DIR
    cache_path = OUTPUT_DIR / OLD_STATS_CACHE_NAME
    label_mean, label_std = load_or_create_label_stats_cache(stats_dir, cache_path)

    h1_files = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    sample_path = h1_files[SAMPLE_INDEX]
    h1_waveform, label_true = load_sample(sample_path)

    if in_channels == 2:
        l1_map = {sample_key(path): path for path in VAL_L1_DIR.iterdir() if path.suffix == ".npy"}
        l1_path = l1_map.get(sample_key(sample_path))
        if l1_path is None:
            raise FileNotFoundError(f"No matching L1 sample found for H1 sample: {sample_path.name}")
        l1_waveform, _ = load_sample(l1_path)
        waveform_raw = np.stack([h1_waveform, l1_waveform], axis=0)
    else:
        waveform_raw = np.expand_dims(h1_waveform, axis=0)

    waveform_tensor = torch.from_numpy(waveform_raw).float()
    samples_normalized = sample_posterior(model, waveform_tensor, NUM_POSTERIOR_SAMPLES, device)
    posterior_samples = inverse_label_normalization(samples_normalized, label_mean, label_std)
    posterior_mean = posterior_samples.mean(axis=0)

    print(f"Sample path: {sample_path}")
    print(f"Waveform shape: {tuple(waveform_tensor.shape)}")
    print("Posterior mean:")
    for name, value in zip(PARAMETER_NAMES, posterior_mean):
        print(f"  {name}: {value:.6g}")

    output_path = OUTPUT_DIR / OUTPUT_NAME
    plot_corner_kde(posterior_samples, label_true, output_path)
    print(f"Saved KDE posterior figure: {output_path}")


if __name__ == "__main__":
    main()
