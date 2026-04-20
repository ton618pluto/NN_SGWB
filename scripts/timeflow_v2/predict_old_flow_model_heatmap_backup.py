import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from predict_old_flow_model_safe import (  # noqa: E402
    CHECKPOINT_PATH,
    NUM_POSTERIOR_SAMPLES,
    OLD_STATS_CACHE_NAME,
    OUTPUT_DIR,
    PARAMETER_NAMES,
    SAMPLE_INDEX,
    TRAIN_H1_DIR,
    VAL_H1_DIR,
    VAL_L1_DIR,
    build_model,
    inverse_label_normalization,
    load_checkpoint,
    load_or_create_label_stats_cache,
    load_sample,
    sample_key,
    sample_posterior,
)


# =========================
# 这是旧模型保留下来的“原始热力图版本”
# 非对角线位置使用 hist2d 画二维热力图
# =========================
OUTPUT_NAME = "old_flow_model_val_posterior_heatmap_backup.png"


def plot_corner_heatmap(samples: np.ndarray, true_values: np.ndarray, output_path: Path) -> None:
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
                # 对角线：单参数一维边缘后验分布。
                ax.hist(samples[:, col], bins=40, density=True, color="#61d9a8", alpha=0.82)
                # 红色竖线：真实标签值。
                ax.axvline(true_values[col], color="#e84a5f", linewidth=1.5)
                # 黑色虚线：后验样本均值。
                ax.axvline(samples[:, col].mean(), color="#1f2d3d", linewidth=1.2, linestyle="--")
            else:
                # 非对角线：旧版备份图使用 hist2d 直接画二维热力图。
                # 小方格颜色越深，表示对应二维参数区域采样点越密集。
                ax.hist2d(samples[:, col], samples[:, row], bins=36, cmap="Greens")
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
        "Posterior joint distribution for old flow model (heatmap backup)\n"
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

    stats_dir = TRAIN_H1_DIR if TRAIN_H1_DIR.exists() else VAL_H1_DIR
    if stats_dir == TRAIN_H1_DIR:
        print(f"Using training-set label stats from: {TRAIN_H1_DIR}")
    else:
        print("Training-set stats directory not found; fallback to validation-set stats.")
        print(f"Fallback stats directory: {VAL_H1_DIR}")
    cache_path = OUTPUT_DIR / OLD_STATS_CACHE_NAME
    label_mean, label_std = load_or_create_label_stats_cache(stats_dir, cache_path)
    print("Label mean/std ready.")

    file_list = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    if not file_list:
        raise RuntimeError(f"No .npy files found in validation directory: {VAL_H1_DIR}")
    sample_path = file_list[SAMPLE_INDEX]
    print(f"Loading sample: {sample_path.name}")

    h1_waveform, label_true = load_sample(sample_path)

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
    samples_normalized = sample_posterior(model, waveform_tensor, NUM_POSTERIOR_SAMPLES, device)
    posterior_samples = inverse_label_normalization(samples_normalized, label_mean, label_std)

    output_path = OUTPUT_DIR / OUTPUT_NAME
    print("Drawing heatmap figure...")
    plot_corner_heatmap(posterior_samples, label_true, output_path)
    print(f"Saved posterior heatmap figure: {output_path}")


if __name__ == "__main__":
    main()
