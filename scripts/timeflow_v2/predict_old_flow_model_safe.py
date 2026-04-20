import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from oldModel import OldGWFlowModel  # noqa: E402


# =========================
# 直接在这里改推理配置
# =========================
REPO_ROOT = CURRENT_DIR.parents[1]
# 仓库根目录。

CHECKPOINT_PATH = CURRENT_DIR / "outputs" / "best_flow_model.pth"
# 旧模型的 checkpoint；这是原始 flow 模型的 state_dict。

TRAIN_H1_DIR = SCRIPTS_ROOT / "processed_data_train"/ "v2"/ "training_set0" / "H1_splits"
# 优先使用训练集 H1 切片目录来计算标签均值和标准差。
# 如果该目录不存在，脚本会退化到验证集目录计算统计量。

VAL_ROOT = SCRIPTS_ROOT / "processed_data_val" / "v1"
# 验证集根目录；默认假设其中有 H1_splits。
VAL_H1_DIR = VAL_ROOT / "H1_splits"
VAL_L1_DIR = VAL_ROOT / "L1_splits"
# 如果旧 checkpoint 是双通道模型，则这里会自动读取 L1 并与 H1 配对。

SAMPLE_INDEX = 0
# 选取验证集中的第几个样本做预测。
NUM_POSTERIOR_SAMPLES = 800
# 后验采样数；为了稳妥，默认设置得不大，适合 CPU 推理。

OUTPUT_DIR = CURRENT_DIR / "result_fig"
OUTPUT_NAME = "old_flow_model_val_posterior_contour.png"
OLD_STATS_CACHE_NAME = "old_model_label_stats_cache.npz"
# 旧模型标签统计量缓存文件名；首次统计后会保存，下次直接复用。

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
# 原始标签是 11 维；这里沿用旧 dataset 的做法，删除第 3 个分量 beta_z，保留 10 维。

SCALE_FACTOR = 1e23
# 与旧版 GWDataset 一致：数值放大后逐通道去均值。

PARAM_DIM = 10
# 旧模型输出的参数维度。


def filter_label(label: np.ndarray) -> np.ndarray:
    # 保持和旧版 GWDataset 完全一致：删除原始标签中的第 3 个分量。
    return np.concatenate([label[:2], label[3:]]).astype(np.float32, copy=False)


def preprocess_old_channel(data: np.ndarray) -> np.ndarray:
    # 对齐 origin_model/GWDataset.py：数值放大后逐通道去均值。
    data = np.asarray(data, dtype=np.float32) * SCALE_FACTOR
    return data - data.mean(dtype=np.float32)


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_dict = np.load(path, allow_pickle=True).item()
    data = preprocess_old_channel(sample_dict["data"])
    label = filter_label(np.asarray(sample_dict["label"], dtype=np.float32))
    return data, label


def sample_key(path: Path) -> str:
    # 例如 H1-popxxxx 和 L1-popxxxx，会保留 '-' 后面的公共部分用于配对。
    return path.stem.split("-", 1)[-1]


def collect_label_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Label stats directory not found: {data_dir}")

    file_list = sorted(path for path in data_dir.iterdir() if path.suffix == ".npy")
    if not file_list:
        raise RuntimeError(f"No .npy files found in {data_dir}")

    labels = []
    progress = tqdm(file_list, desc="Collecting label stats", unit="file")
    for path in progress:
        _, label = load_sample(path)
        labels.append(label)

    all_labels = np.stack(labels, axis=0)
    mean = all_labels.mean(axis=0)
    std = all_labels.std(axis=0)
    std[std == 0] = 1.0
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def load_or_create_label_stats_cache(data_dir: Path, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # 如果缓存已经存在，就直接读取，避免每次推理都重新扫描整个训练集。
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
    # 从第一层卷积核权重形状推断输入通道数：
    # [16, 1, 64] -> 单通道
    # [16, 2, 64] -> 双通道
    return int(state_dict["embedding_net.conv_layers.0.weight"].shape[1])


def infer_context_dim(state_dict: dict) -> int:
    # 从最后一层全连接权重形状推断旧模型 context 维度。
    # embedding_net.fc.3.weight.shape = [context_dim, 256]
    return int(state_dict["embedding_net.fc.3.weight"].shape[0])


def build_model(state_dict: dict, device: torch.device) -> tuple[OldGWFlowModel, int, int]:
    # 旧模型可能是单通道或双通道；同时 context_dim 也可能不是默认值。
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
    # 这里会比较慢：模型需要先对超长时域输入做一次前向编码，再从 flow 中采样。
    print("Sampling posterior...")
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    print("Posterior sampling finished.")
    return samples


def plot_corner_contour(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
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
                # 对角线：一维边缘后验分布。
                ax.hist(samples[:, col], bins=40, density=True, color="#5b8ff9", alpha=0.82)
                # 红色竖线：该样本真实标签值。
                ax.axvline(true_values[col], color="#e84a5f", linewidth=1.5)
                # 黑色虚线：后验样本均值。
                ax.axvline(samples[:, col].mean(), color="#1f2d3d", linewidth=1.2, linestyle="--")
            else:
                # 非对角线：二维联合后验的等高线图。
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
                # 红色 x：真实标签在二维参数平面中的位置。
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
        "Posterior joint distribution for old flow model\n"
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

    if not VAL_H1_DIR.exists():
        raise FileNotFoundError(f"Validation H1 directory not found: {VAL_H1_DIR}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    print("Checkpoint loaded.")
    model, in_channels, context_dim = build_model(checkpoint, device)
    print(f"Inferred input channels from checkpoint: {in_channels}")
    print(f"Inferred context dim from checkpoint: {context_dim}")
    print("Model restored.")

    # 旧 checkpoint 不保存 label_mean/std，因此优先使用训练集统计量恢复真实物理尺度。
    stats_dir = TRAIN_H1_DIR if TRAIN_H1_DIR.exists() else VAL_H1_DIR
    if stats_dir == TRAIN_H1_DIR:
        print(f"Using training-set label stats from: {TRAIN_H1_DIR}")
    else:
        print("Training-set stats directory not found; fallback to validation-set stats.")
        print(f"Fallback stats directory: {VAL_H1_DIR}")
    print("Calculating label mean/std...")
    cache_path = OUTPUT_DIR / OLD_STATS_CACHE_NAME
    label_mean, label_std = load_or_create_label_stats_cache(stats_dir, cache_path)
    print("Label mean/std ready.")

    file_list = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    if not file_list:
        raise RuntimeError(f"No .npy files found in validation directory: {VAL_H1_DIR}")
    sample_path = file_list[SAMPLE_INDEX]
    print(f"Loading sample: {sample_path.name}")

    h1_waveform, label_true = load_sample(sample_path)
    label_normalized = (torch.tensor(label_true) - label_mean) / label_std

    if in_channels == 2:
        if not VAL_L1_DIR.exists():
            raise FileNotFoundError(
                f"Checkpoint expects 2-channel input, but validation L1 directory not found: {VAL_L1_DIR}"
            )
        l1_map = {sample_key(path): path for path in VAL_L1_DIR.iterdir() if path.suffix == ".npy"}
        paired_l1_path = l1_map.get(sample_key(sample_path))
        if paired_l1_path is None:
            raise FileNotFoundError(f"No matching L1 sample found for H1 sample: {sample_path.name}")
        print(f"Matched L1 sample: {paired_l1_path.name}")
        l1_waveform, _ = load_sample(paired_l1_path)
        waveform_raw = np.stack([h1_waveform, l1_waveform], axis=0)
    else:
        waveform_raw = np.expand_dims(h1_waveform, axis=0)

    waveform_tensor = torch.from_numpy(waveform_raw).float()
    print("Waveform tensor prepared.")

    samples_normalized = sample_posterior(model, waveform_tensor, NUM_POSTERIOR_SAMPLES, device)
    posterior_samples = inverse_label_normalization(samples_normalized, label_mean, label_std)
    posterior_mean = posterior_samples.mean(axis=0)
    print("Posterior statistics computed.")

    print(f"Sample path: {sample_path}")
    print(f"Waveform shape: {tuple(waveform_tensor.shape)}")
    print("True label:")
    for name, value in zip(PARAMETER_NAMES, label_true):
        print(f"  {name}: {value:.6g}")
    print("Posterior mean:")
    for name, value in zip(PARAMETER_NAMES, posterior_mean):
        print(f"  {name}: {value:.6g}")
    print("Normalized label (debug):")
    for name, value in zip(PARAMETER_NAMES, label_normalized.numpy()):
        print(f"  {name}: {value:.6g}")

    output_path = OUTPUT_DIR / OUTPUT_NAME
    print("Drawing contour figure...")
    plot_corner_contour(posterior_samples, label_true, output_path)
    print(f"Saved posterior contour figure: {output_path}")


if __name__ == "__main__":
    main()
