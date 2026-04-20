import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from compare_new_old_models_one_sample import (  # noqa: E402
    NEW_CHECKPOINT_PATH,
    NUM_POSTERIOR_SAMPLES,
    OLD_CHECKPOINT_PATH,
    OLD_STATS_CACHE_NAME,
    OUTPUT_DIR,
    PARAMETER_NAMES,
    TRAIN_H1_DIR_FOR_OLD,
    VAL_H1_DIR,
    VAL_L1_DIR,
    build_new_dataset,
    build_waveform_for_old,
    collect_label_stats,
    inverse_label_normalization,
    load_new_model,
    load_old_model,
    load_or_create_label_stats_cache,
    log_prob_new_model,
    log_prob_old_model,
    mean_absolute_error,
    sample_old_model,
    sample_key,
    sample_new_model,
)


# =========================
# 直接在这里改多样本评估配置
# =========================
NUM_EVAL_SAMPLES = 100
# 从验证集前多少个样本做新旧模型对比。

OUTPUT_FIG_NAME = "new_vs_old_multi_sample_comparison.png"
# 输出汇总图。

OUTPUT_TXT_NAME = "new_vs_old_multi_sample_metrics.txt"
# 输出文本汇总。


def save_metrics_text(
    output_path: Path,
    indices: list[int],
    sample_names: list[str],
    new_log_probs: list[float],
    old_log_probs: list[float],
    new_maes: list[float],
    old_maes: list[float],
) -> None:
    lines = []
    lines.append(f"num_eval_samples: {len(indices)}")
    lines.append(f"new_model_log_prob_mean: {np.mean(new_log_probs):.8f}")
    lines.append(f"old_model_log_prob_mean: {np.mean(old_log_probs):.8f}")
    lines.append(f"new_model_log_prob_std: {np.std(new_log_probs):.8f}")
    lines.append(f"old_model_log_prob_std: {np.std(old_log_probs):.8f}")
    lines.append(f"new_model_mae_mean: {np.mean(new_maes):.8f}")
    lines.append(f"old_model_mae_mean: {np.mean(old_maes):.8f}")
    lines.append(f"new_model_mae_std: {np.std(new_maes):.8f}")
    lines.append(f"old_model_mae_std: {np.std(old_maes):.8f}")
    lines.append("")
    lines.append("index,sample_name,new_log_prob,old_log_prob,new_mae,old_mae")

    for index, sample_name, new_lp, old_lp, new_mae, old_mae in zip(
        indices, sample_names, new_log_probs, old_log_probs, new_maes, old_maes
    ):
        lines.append(
            f"{index},{sample_name},{new_lp:.8f},{old_lp:.8f},{new_mae:.8f},{old_mae:.8f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(
    indices: list[int],
    new_log_probs: list[float],
    old_log_probs: list[float],
    new_maes: list[float],
    old_maes: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(indices, new_log_probs, marker="o", markersize=3, linewidth=1.2, color="#5b8ff9", label="New Model")
    axes[0].plot(indices, old_log_probs, marker="o", markersize=3, linewidth=1.2, color="#61d9a8", label="Old Model")
    axes[0].axhline(np.mean(new_log_probs), color="#5b8ff9", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].axhline(np.mean(old_log_probs), color="#61d9a8", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("log_prob", fontsize=10)
    axes[0].set_title("Per-sample log_prob comparison", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(indices, new_maes, marker="o", markersize=3, linewidth=1.2, color="#5b8ff9", label="New Model")
    axes[1].plot(indices, old_maes, marker="o", markersize=3, linewidth=1.2, color="#61d9a8", label="Old Model")
    axes[1].axhline(np.mean(new_maes), color="#5b8ff9", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].axhline(np.mean(old_maes), color="#61d9a8", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("MAE", fontsize=10)
    axes[1].set_xlabel("Validation sample index", fontsize=10)
    axes[1].set_title("Per-sample posterior-mean MAE comparison", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "New vs Old model comparison on multiple validation samples\n"
        f"avg new log_prob = {np.mean(new_log_probs):.4f}, avg old log_prob = {np.mean(old_log_probs):.4f} | "
        f"avg new MAE = {np.mean(new_maes):.4f}, avg old MAE = {np.mean(old_maes):.4f}",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"New checkpoint: {NEW_CHECKPOINT_PATH}")
    print(f"Old checkpoint: {OLD_CHECKPOINT_PATH}")
    print(f"Validation H1: {VAL_H1_DIR}")
    print(f"Validation L1: {VAL_L1_DIR}")

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

    stats_dir = TRAIN_H1_DIR_FOR_OLD if TRAIN_H1_DIR_FOR_OLD.exists() else VAL_H1_DIR
    if stats_dir == TRAIN_H1_DIR_FOR_OLD:
        print(f"Using old-model label stats from training directory: {TRAIN_H1_DIR_FOR_OLD}")
    else:
        print("Training stats for old model not found; fallback to validation stats.")
        print(f"Fallback stats directory: {VAL_H1_DIR}")
    cache_path = OUTPUT_DIR / OLD_STATS_CACHE_NAME
    old_label_mean, old_label_std = load_or_create_label_stats_cache(stats_dir, cache_path)

    h1_files = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    if not h1_files:
        raise RuntimeError(f"No .npy files found in validation directory: {VAL_H1_DIR}")

    eval_count = min(NUM_EVAL_SAMPLES, len(h1_files), len(new_dataset))
    print(f"Evaluating {eval_count} validation samples...")

    indices = []
    sample_names = []
    new_log_probs = []
    old_log_probs = []
    new_maes = []
    old_maes = []

    progress = tqdm(range(eval_count), desc="Comparing new vs old", unit="sample")
    for sample_index in progress:
        sample_path = h1_files[sample_index]

        # New model
        new_waveform, new_label_normalized = new_dataset[sample_index]
        label_true = inverse_label_normalization(
            new_label_normalized,
            new_checkpoint["label_mean"],
            new_checkpoint["label_std"],
        )
        new_samples_normalized = sample_new_model(new_model, new_waveform, NUM_POSTERIOR_SAMPLES, device)
        new_samples = inverse_label_normalization(
            new_samples_normalized,
            new_checkpoint["label_mean"],
            new_checkpoint["label_std"],
        )
        new_mean = new_samples.mean(axis=0)
        new_log_prob = log_prob_new_model(new_model, new_label_normalized, new_waveform, device)

        # Old model
        old_waveform, old_label_true = build_waveform_for_old(sample_path, old_in_channels)
        old_label_normalized = (torch.tensor(old_label_true) - old_label_mean) / old_label_std
        old_samples_normalized = sample_old_model(old_model, old_waveform, NUM_POSTERIOR_SAMPLES, device)
        old_samples = inverse_label_normalization(old_samples_normalized, old_label_mean, old_label_std)
        old_mean = old_samples.mean(axis=0)
        old_log_prob = log_prob_old_model(old_model, old_label_normalized, old_waveform, device)

        indices.append(sample_index)
        sample_names.append(sample_path.name)
        new_log_probs.append(new_log_prob)
        old_log_probs.append(old_log_prob)
        new_maes.append(mean_absolute_error(new_mean, label_true))
        old_maes.append(mean_absolute_error(old_mean, old_label_true))

        progress.set_postfix(
            new_lp=f"{np.mean(new_log_probs):.2f}",
            old_lp=f"{np.mean(old_log_probs):.2f}",
            new_mae=f"{np.mean(new_maes):.2f}",
            old_mae=f"{np.mean(old_maes):.2f}",
        )

    fig_path = OUTPUT_DIR / OUTPUT_FIG_NAME
    txt_path = OUTPUT_DIR / OUTPUT_TXT_NAME

    print("Saving summary figure...")
    plot_summary(indices, new_log_probs, old_log_probs, new_maes, old_maes, fig_path)
    print("Saving summary metrics...")
    save_metrics_text(txt_path, indices, sample_names, new_log_probs, old_log_probs, new_maes, old_maes)

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {txt_path}")
    print(f"Average new model log_prob: {np.mean(new_log_probs):.6f}")
    print(f"Average old model log_prob: {np.mean(old_log_probs):.6f}")
    print(f"Average new model MAE: {np.mean(new_maes):.6f}")
    print(f"Average old model MAE: {np.mean(old_maes):.6f}")


if __name__ == "__main__":
    main()
