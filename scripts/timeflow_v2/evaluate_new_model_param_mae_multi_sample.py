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
    PARAMETER_NAMES,
    VAL_H1_DIR,
    VAL_L1_DIR,
    build_new_dataset,
    inverse_label_normalization,
    load_new_model,
    sample_new_model,
)


# =========================
# 直接在这里改新模型逐参数评估配置
# =========================
NUM_EVAL_SAMPLES = 100
# 从验证集前多少个样本统计每个参数单独的 MAE。

OUTPUT_FIG_NAME = "new_model_param_mae_multi_sample.png"
# 逐参数 MAE 柱状图。

OUTPUT_TXT_NAME = "new_model_param_mae_multi_sample.txt"
# 逐参数 MAE 文本结果。

OUTPUT_DIR = CURRENT_DIR / "MAE_fig"
# 逐参数 MAE 图和文本统一输出到 timeflow_v2/MAE_fig。


def compute_parameter_mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(predictions - targets), axis=0)


def save_metrics_text(
    output_path: Path,
    parameter_names: list[str],
    per_param_mae_mean: np.ndarray,
    per_param_mae_std: np.ndarray,
    overall_mae_per_sample: np.ndarray,
) -> None:
    lines = []
    lines.append(f"num_eval_samples: {overall_mae_per_sample.shape[0]}")
    lines.append(f"overall_mae_mean: {overall_mae_per_sample.mean():.8f}")
    lines.append(f"overall_mae_std: {overall_mae_per_sample.std():.8f}")
    lines.append("")
    lines.append("parameter,mae_mean,mae_std")

    for name, mae_mean, mae_std in zip(parameter_names, per_param_mae_mean, per_param_mae_std):
        lines.append(f"{name},{mae_mean:.8f},{mae_std:.8f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_parameter_mae(
    parameter_names: list[str],
    per_param_mae_mean: np.ndarray,
    per_param_mae_std: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(parameter_names))
    ax.bar(
        x,
        per_param_mae_mean,
        yerr=per_param_mae_std,
        color="#5b8ff9",
        alpha=0.88,
        capsize=4,
        edgecolor="#355c7d",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(parameter_names, rotation=25, ha="right")
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("New model per-parameter MAE on validation samples", fontsize=13)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {NEW_CHECKPOINT_PATH}")
    print(f"Validation H1: {VAL_H1_DIR}")
    print(f"Validation L1: {VAL_L1_DIR}")

    if not VAL_H1_DIR.exists():
        raise FileNotFoundError(f"Validation H1 directory not found: {VAL_H1_DIR}")

    print("Loading new model...")
    new_model, new_checkpoint = load_new_model(device)

    print("Building validation dataset...")
    new_dataset = build_new_dataset(new_checkpoint)

    h1_files = sorted(path for path in VAL_H1_DIR.iterdir() if path.suffix == ".npy")
    if not h1_files:
        raise RuntimeError(f"No .npy files found in validation directory: {VAL_H1_DIR}")

    eval_count = min(NUM_EVAL_SAMPLES, len(h1_files), len(new_dataset))
    print(f"Evaluating {eval_count} validation samples...")

    per_sample_param_maes = []
    overall_maes = []

    progress = tqdm(range(eval_count), desc="Evaluating new-model parameter MAE", unit="sample")
    for sample_index in progress:
        waveform, label_normalized = new_dataset[sample_index]
        label_true = inverse_label_normalization(
            label_normalized,
            new_checkpoint["label_mean"],
            new_checkpoint["label_std"],
        )

        samples_normalized = sample_new_model(new_model, waveform, NUM_POSTERIOR_SAMPLES, device)
        posterior_samples = inverse_label_normalization(
            samples_normalized,
            new_checkpoint["label_mean"],
            new_checkpoint["label_std"],
        )
        posterior_mean = posterior_samples.mean(axis=0)

        param_mae = np.abs(posterior_mean - label_true)
        per_sample_param_maes.append(param_mae)
        overall_maes.append(param_mae.mean())

        progress.set_postfix(overall_mae=f"{np.mean(overall_maes):.2f}")

    per_sample_param_maes_array = np.asarray(per_sample_param_maes, dtype=np.float64)
    overall_maes_array = np.asarray(overall_maes, dtype=np.float64)

    per_param_mae_mean = per_sample_param_maes_array.mean(axis=0)
    per_param_mae_std = per_sample_param_maes_array.std(axis=0)

    fig_path = OUTPUT_DIR / OUTPUT_FIG_NAME
    txt_path = OUTPUT_DIR / OUTPUT_TXT_NAME

    print("Saving parameter MAE figure...")
    plot_parameter_mae(PARAMETER_NAMES, per_param_mae_mean, per_param_mae_std, fig_path)
    print("Saving parameter MAE text...")
    save_metrics_text(txt_path, PARAMETER_NAMES, per_param_mae_mean, per_param_mae_std, overall_maes_array)

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {txt_path}")
    print(f"Overall MAE mean: {overall_maes_array.mean():.6f}")
    print("Per-parameter MAE mean:")
    for name, value in zip(PARAMETER_NAMES, per_param_mae_mean):
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
