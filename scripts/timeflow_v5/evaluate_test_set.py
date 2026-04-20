from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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


NUM_TEST_SAMPLES = 100
NUM_POSTERIOR_SAMPLES = 800
OUTPUT_DIR = Path(__file__).resolve().parent / "test_fig"
OUTPUT_FIG_NAME = "test_set_param_mae_v5.png"
OUTPUT_TXT_NAME = "test_set_metrics_v5.txt"


def save_metrics_text(
    output_path: Path,
    parameter_names: list[str],
    per_param_mae_mean: np.ndarray,
    per_param_mae_std: np.ndarray,
    overall_mae_per_sample: np.ndarray,
    nll_per_sample: np.ndarray,
) -> None:
    lines = [
        f"num_test_samples: {overall_mae_per_sample.shape[0]}",
        f"overall_mae_mean: {overall_mae_per_sample.mean():.8f}",
        f"overall_mae_std: {overall_mae_per_sample.std():.8f}",
        f"test_nll_mean: {nll_per_sample.mean():.8f}",
        f"test_nll_std: {nll_per_sample.std():.8f}",
        "",
        "parameter,mae_mean,mae_std",
    ]
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
        color="#61d9a8",
        alpha=0.88,
        capsize=4,
        edgecolor="#1f7a5c",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(parameter_names, rotation=25, ha="right")
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("V5 model per-parameter MAE on test split", fontsize=13)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


@torch.inference_mode()
def compute_test_nll(model, waveform: torch.Tensor, label_normalized: torch.Tensor, device: torch.device) -> float:
    theta = label_normalized.unsqueeze(0).to(device)
    x = waveform.unsqueeze(0).to(device)
    return float((-model.log_prob(theta, x)).item())


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_split_dataset(checkpoint, split_name="test")
    model = build_model(checkpoint, device)

    eval_count = min(NUM_TEST_SAMPLES, len(dataset))
    print(f"Evaluating {eval_count} test samples...")

    per_sample_param_maes = []
    overall_maes = []
    nll_values = []

    progress = tqdm(range(eval_count), desc="Evaluating v5 test split", unit="sample")
    for sample_index in progress:
        waveform, label_normalized = dataset[sample_index]
        label_true = inverse_label_normalization(
            label_normalized,
            checkpoint["label_mean"],
            checkpoint["label_std"],
        )

        samples_normalized = sample_posterior(model, waveform, NUM_POSTERIOR_SAMPLES, device)
        posterior_samples = inverse_label_normalization(
            samples_normalized,
            checkpoint["label_mean"],
            checkpoint["label_std"],
        )
        posterior_mean = posterior_samples.mean(axis=0)

        param_mae = np.abs(posterior_mean - label_true)
        nll = compute_test_nll(model, waveform, label_normalized, device)

        per_sample_param_maes.append(param_mae)
        overall_maes.append(param_mae.mean())
        nll_values.append(nll)
        progress.set_postfix(mae=f"{np.mean(overall_maes):.2f}", nll=f"{np.mean(nll_values):.2f}")

    per_sample_param_maes_array = np.asarray(per_sample_param_maes, dtype=np.float64)
    overall_maes_array = np.asarray(overall_maes, dtype=np.float64)
    nll_values_array = np.asarray(nll_values, dtype=np.float64)
    per_param_mae_mean = per_sample_param_maes_array.mean(axis=0)
    per_param_mae_std = per_sample_param_maes_array.std(axis=0)

    fig_path = OUTPUT_DIR / OUTPUT_FIG_NAME
    txt_path = OUTPUT_DIR / OUTPUT_TXT_NAME
    plot_parameter_mae(PARAMETER_NAMES, per_param_mae_mean, per_param_mae_std, fig_path)
    save_metrics_text(
        txt_path,
        PARAMETER_NAMES,
        per_param_mae_mean,
        per_param_mae_std,
        overall_maes_array,
        nll_values_array,
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {txt_path}")
    print(f"Overall test MAE mean: {overall_maes_array.mean():.6f}")
    print(f"Overall test NLL mean: {nll_values_array.mean():.6f}")


if __name__ == "__main__":
    main()
