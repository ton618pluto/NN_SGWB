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

from dataset import GWDatasetV2  # noqa: E402
from model import GWFlowModelV2  # noqa: E402
from oldModel import OldGWFlowModel  # noqa: E402


# =========================
# 直接在这里改对比配置
# =========================
NEW_CHECKPOINT_PATH = CURRENT_DIR / "outputs" / "best_flow_v2.pt"
# 新模型 checkpoint。
OLD_CHECKPOINT_PATH = CURRENT_DIR / "outputs" / "best_flow_model.pth"
# 旧模型 checkpoint。

TRAIN_H1_DIR_FOR_OLD = SCRIPTS_ROOT / "processed_data_train" / "v2" / "training_set0" / "H1_splits"
# 旧模型不保存 label_mean/std，因此优先从这个训练集目录统计。

VAL_ROOT = SCRIPTS_ROOT / "processed_data_val" / "v1"
VAL_H1_DIR = VAL_ROOT / "H1_splits"
VAL_L1_DIR = VAL_ROOT / "L1_splits"

SAMPLE_INDEX = 0
NUM_POSTERIOR_SAMPLES = 800

NEW_CONTEXT_DIM = 256
NEW_FLOW_LAYERS = 6
NEW_FLOW_HIDDEN = 256

OUTPUT_DIR = CURRENT_DIR / "result_fig"
OUTPUT_FIG_NAME = "new_vs_old_single_sample_comparison.png"
OUTPUT_TXT_NAME = "new_vs_old_single_sample_metrics.txt"
OLD_STATS_CACHE_NAME = "old_model_label_stats_cache.npz"
# 旧模型标签统计量缓存文件名；首次计算后会保存到 result_fig 目录，下次直接复用。

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


def filter_label(label: np.ndarray) -> np.ndarray:
    return np.concatenate([label[:2], label[3:]]).astype(np.float32, copy=False)


def sample_key(path: Path) -> str:
    return path.stem.split("-", 1)[-1]


def preprocess_old_channel(data: np.ndarray) -> np.ndarray:
    # 对齐 origin_model/GWDataset.py：数值放大后逐通道去均值。
    data = np.asarray(data, dtype=np.float32) * SCALE_FACTOR
    return data - data.mean(dtype=np.float32)


def load_raw_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_dict = np.load(path, allow_pickle=True).item()
    data = preprocess_old_channel(sample_dict["data"])
    label = filter_label(np.asarray(sample_dict["label"], dtype=np.float32))
    return data, label


def collect_label_stats(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Label stats directory not found: {data_dir}")

    file_list = sorted(path for path in data_dir.iterdir() if path.suffix == ".npy")
    if not file_list:
        raise RuntimeError(f"No .npy files found in {data_dir}")

    labels = []
    progress = tqdm(file_list, desc="Collecting old-model label stats", unit="file")
    for path in progress:
        _, label = load_raw_sample(path)
        labels.append(label)

    all_labels = np.stack(labels, axis=0)
    mean = all_labels.mean(axis=0)
    std = all_labels.std(axis=0)
    std[std == 0] = 1.0
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def load_or_create_label_stats_cache(data_dir: Path, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # 如果缓存已存在，则直接读取，避免每次都重新扫描整个训练集。
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


def inverse_label_normalization(values: torch.Tensor | np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return values * std.detach().cpu().numpy() + mean.detach().cpu().numpy()


def infer_old_in_channels(state_dict: dict) -> int:
    return int(state_dict["embedding_net.conv_layers.0.weight"].shape[1])


def infer_old_context_dim(state_dict: dict) -> int:
    return int(state_dict["embedding_net.fc.3.weight"].shape[0])


def load_new_model(device: torch.device) -> tuple[GWFlowModelV2, dict]:
    checkpoint = torch.load(NEW_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = GWFlowModelV2(
        param_dim=int(checkpoint["param_dim"]),
        in_channels=int(checkpoint["num_channels"]),
        context_dim=NEW_CONTEXT_DIM,
        flow_layers=NEW_FLOW_LAYERS,
        flow_hidden_features=NEW_FLOW_HIDDEN,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_old_model(device: torch.device) -> tuple[OldGWFlowModel, dict, int, int]:
    state_dict = torch.load(OLD_CHECKPOINT_PATH, map_location=device, weights_only=False)
    in_channels = infer_old_in_channels(state_dict)
    context_dim = infer_old_context_dim(state_dict)
    model = OldGWFlowModel(param_dim=10, context_dim=context_dim, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict, in_channels, context_dim


def build_new_dataset(new_checkpoint: dict) -> GWDatasetV2:
    return GWDatasetV2(
        h1_dir=VAL_H1_DIR,
        l1_dir=VAL_L1_DIR if int(new_checkpoint["num_channels"]) == 2 else None,
        label_mean=new_checkpoint["label_mean"].cpu(),
        label_std=new_checkpoint["label_std"].cpu(),
        sample_step=1,
        dataset_name="val_compare_new",
    )


def build_waveform_for_old(sample_path: Path, old_in_channels: int) -> tuple[torch.Tensor, np.ndarray]:
    h1_waveform, label_true = load_raw_sample(sample_path)

    if old_in_channels == 2:
        if not VAL_L1_DIR.exists():
            raise FileNotFoundError(f"Old checkpoint expects L1 directory, but not found: {VAL_L1_DIR}")
        l1_map = {sample_key(path): path for path in VAL_L1_DIR.iterdir() if path.suffix == ".npy"}
        l1_path = l1_map.get(sample_key(sample_path))
        if l1_path is None:
            raise FileNotFoundError(f"No matching L1 file found for {sample_path.name}")
        l1_waveform, _ = load_raw_sample(l1_path)
        waveform = np.stack([h1_waveform, l1_waveform], axis=0)
    else:
        waveform = np.expand_dims(h1_waveform, axis=0)

    return torch.from_numpy(waveform).float(), label_true


@torch.inference_mode()
def sample_new_model(model: GWFlowModelV2, waveform: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    return samples


@torch.inference_mode()
def sample_old_model(model: OldGWFlowModel, waveform: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    return samples


@torch.inference_mode()
def log_prob_new_model(model: GWFlowModelV2, theta_normalized: torch.Tensor, waveform: torch.Tensor, device: torch.device) -> float:
    theta_normalized = theta_normalized.unsqueeze(0).to(device)
    waveform = waveform.unsqueeze(0).to(device)
    return float(model.log_prob(theta_normalized, waveform).item())


@torch.inference_mode()
def log_prob_old_model(model: OldGWFlowModel, theta_normalized: torch.Tensor, waveform: torch.Tensor, device: torch.device) -> float:
    theta_normalized = theta_normalized.unsqueeze(0).to(device)
    waveform = waveform.unsqueeze(0).to(device)
    context = model.embedding_net(waveform)
    return float(model.flow.log_prob(inputs=theta_normalized, context=context).item())


def mean_absolute_error(posterior_mean: np.ndarray, label_true: np.ndarray) -> float:
    return float(np.mean(np.abs(posterior_mean - label_true)))


def plot_model_comparison(
    new_samples: np.ndarray,
    old_samples: np.ndarray,
    label_true: np.ndarray,
    new_log_prob: float,
    old_log_prob: float,
    output_path: Path,
) -> None:
    dim = label_true.shape[0]
    names = PARAMETER_NAMES[:dim]

    fig, axes = plt.subplots(dim, 2, figsize=(12, 2.2 * dim))
    fig.patch.set_facecolor("white")

    for idx in range(dim):
        for col, (samples, title, color) in enumerate(
            [
                (new_samples[:, idx], "New Model", "#5b8ff9"),
                (old_samples[:, idx], "Old Model", "#61d9a8"),
            ]
        ):
            ax = axes[idx, col]
            ax.hist(samples, bins=40, density=True, color=color, alpha=0.82)
            ax.axvline(label_true[idx], color="#e84a5f", linewidth=1.5)
            ax.axvline(samples.mean(), color="#1f2d3d", linewidth=1.1, linestyle="--")
            if idx == 0:
                ax.set_title(title, fontsize=12)
            ax.set_ylabel(names[idx], fontsize=8)
            ax.tick_params(axis="both", labelsize=7, length=2)

    fig.suptitle(
        f"Single-sample posterior comparison\n"
        f"new log_prob = {new_log_prob:.4f}, old log_prob = {old_log_prob:.4f}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_metrics_text(
    output_path: Path,
    sample_name: str,
    label_true: np.ndarray,
    new_mean: np.ndarray,
    old_mean: np.ndarray,
    new_log_prob: float,
    old_log_prob: float,
) -> None:
    lines = []
    lines.append(f"sample: {sample_name}")
    lines.append(f"new_model_log_prob: {new_log_prob:.8f}")
    lines.append(f"old_model_log_prob: {old_log_prob:.8f}")
    lines.append(f"new_model_mae: {mean_absolute_error(new_mean, label_true):.8f}")
    lines.append(f"old_model_mae: {mean_absolute_error(old_mean, label_true):.8f}")
    lines.append("")
    lines.append("parameter,true,new_mean,old_mean")

    for name, true_value, new_value, old_value in zip(PARAMETER_NAMES, label_true, new_mean, old_mean):
        lines.append(f"{name},{true_value:.8f},{new_value:.8f},{old_value:.8f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")

    if not VAL_H1_DIR.exists():
        raise FileNotFoundError(f"Validation H1 directory not found: {VAL_H1_DIR}")

    print("Loading new model...")
    new_model, new_checkpoint = load_new_model(device)
    print("Loading old model...")
    old_model, _, old_in_channels, old_context_dim = load_old_model(device)
    print(f"Old model inferred input channels: {old_in_channels}")
    print(f"Old model inferred context dim: {old_context_dim}")

    print("Building new-model validation dataset...")
    new_dataset = build_new_dataset(new_checkpoint)

    h1_files = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    sample_path = h1_files[SAMPLE_INDEX]
    print(f"Using sample: {sample_path.name}")

    # New model sample and true label
    new_waveform, new_label_normalized = new_dataset[SAMPLE_INDEX]
    label_true = inverse_label_normalization(new_label_normalized, new_checkpoint["label_mean"], new_checkpoint["label_std"])
    new_samples_normalized = sample_new_model(new_model, new_waveform, NUM_POSTERIOR_SAMPLES, device)
    new_samples = inverse_label_normalization(new_samples_normalized, new_checkpoint["label_mean"], new_checkpoint["label_std"])
    new_mean = new_samples.mean(axis=0)
    new_log_prob = log_prob_new_model(new_model, new_label_normalized, new_waveform, device)

    # Old model sample and stats
    print("Preparing old-model waveform and label stats...")
    old_waveform, old_label_true = build_waveform_for_old(sample_path, old_in_channels)
    if not np.allclose(old_label_true, label_true, atol=1e-5):
        print("Warning: old/new true labels differ slightly after reconstruction.")

    stats_dir = TRAIN_H1_DIR_FOR_OLD if TRAIN_H1_DIR_FOR_OLD.exists() else VAL_H1_DIR
    if stats_dir == TRAIN_H1_DIR_FOR_OLD:
        print(f"Using old-model label stats from training directory: {TRAIN_H1_DIR_FOR_OLD}")
    else:
        print("Training stats for old model not found; fallback to validation stats.")
        print(f"Fallback stats directory: {VAL_H1_DIR}")
    cache_path = OUTPUT_DIR / OLD_STATS_CACHE_NAME
    old_label_mean, old_label_std = load_or_create_label_stats_cache(stats_dir, cache_path)
    old_label_normalized = (torch.tensor(old_label_true) - old_label_mean) / old_label_std
    old_samples_normalized = sample_old_model(old_model, old_waveform, NUM_POSTERIOR_SAMPLES, device)
    old_samples = inverse_label_normalization(old_samples_normalized, old_label_mean, old_label_std)
    old_mean = old_samples.mean(axis=0)
    old_log_prob = log_prob_old_model(old_model, old_label_normalized, old_waveform, device)

    fig_path = OUTPUT_DIR / OUTPUT_FIG_NAME
    txt_path = OUTPUT_DIR / OUTPUT_TXT_NAME

    print("Saving comparison figure...")
    plot_model_comparison(new_samples, old_samples, label_true, new_log_prob, old_log_prob, fig_path)
    print("Saving metrics text...")
    save_metrics_text(txt_path, sample_path.name, label_true, new_mean, old_mean, new_log_prob, old_log_prob)

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {txt_path}")
    print(f"New model log_prob: {new_log_prob:.6f}")
    print(f"Old model log_prob: {old_log_prob:.6f}")
    print(f"New model MAE: {mean_absolute_error(new_mean, label_true):.6f}")
    print(f"Old model MAE: {mean_absolute_error(old_mean, label_true):.6f}")


if __name__ == "__main__":
    main()
