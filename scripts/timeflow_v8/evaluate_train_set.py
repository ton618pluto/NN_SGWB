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


NUM_TRAIN_SAMPLES = 100
NUM_POSTERIOR_SAMPLES = 800
OUTPUT_DIR = Path(__file__).resolve().parent / "train_fig"
OUTPUT_FIG_NAME = "train_set_param_mae_v8.png"
OUTPUT_TXT_NAME = "train_set_metrics_v8.txt"

PARAMETER_RANGES = {
    "zp": (1.4, 2.4),
    "m_max": (10.0, 100.0),
    "m_min": (2.0, 10.0),
    "sigma_m": (3.52, 3.60),
}


def save_metrics_text(output_path: Path, parameter_names: list[str], parameter_range_widths: np.ndarray, per_param_mae_mean: np.ndarray, per_param_mae_std: np.ndarray, per_param_mae_range_norm_mean: np.ndarray, per_param_coverage68: np.ndarray, per_param_coverage95: np.ndarray, per_param_ci68_width_mean: np.ndarray, per_param_ci68_width_std: np.ndarray, per_param_ci68_width_range_norm_mean: np.ndarray, per_param_ci95_width_mean: np.ndarray, per_param_ci95_width_std: np.ndarray, per_param_ci95_width_range_norm_mean: np.ndarray, overall_mae_per_sample: np.ndarray, nll_per_sample: np.ndarray) -> None:
    lines = [
        f"num_train_samples: {overall_mae_per_sample.shape[0]}",
        f"overall_mae_mean: {overall_mae_per_sample.mean():.8f}",
        f"overall_mae_std: {overall_mae_per_sample.std():.8f}",
        f"train_nll_mean: {nll_per_sample.mean():.8f}",
        f"train_nll_std: {nll_per_sample.std():.8f}",
        "",
        "parameter,range_width,mae_mean,mae_std,mae_range_norm_mean,coverage68,coverage95,ci68_width_mean,ci68_width_std,ci68_width_range_norm_mean,ci95_width_mean,ci95_width_std,ci95_width_range_norm_mean",
    ]
    for name, range_width, mae_mean, mae_std, mae_range_norm_mean, coverage68, coverage95, ci68_width_mean, ci68_width_std, ci68_width_range_norm_mean, ci95_width_mean, ci95_width_std, ci95_width_range_norm_mean in zip(
        parameter_names, parameter_range_widths, per_param_mae_mean, per_param_mae_std, per_param_mae_range_norm_mean, per_param_coverage68, per_param_coverage95, per_param_ci68_width_mean, per_param_ci68_width_std, per_param_ci68_width_range_norm_mean, per_param_ci95_width_mean, per_param_ci95_width_std, per_param_ci95_width_range_norm_mean
    ):
        lines.append(
            f"{name},{range_width:.8f},{mae_mean:.8f},{mae_std:.8f},{mae_range_norm_mean:.8f},{coverage68:.8f},{coverage95:.8f},{ci68_width_mean:.8f},{ci68_width_std:.8f},{ci68_width_range_norm_mean:.8f},{ci95_width_mean:.8f},{ci95_width_std:.8f},{ci95_width_range_norm_mean:.8f}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_parameter_mae(parameter_names: list[str], per_param_mae_mean: np.ndarray, per_param_mae_std: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(parameter_names))
    ax.bar(x, per_param_mae_mean, yerr=per_param_mae_std, color="#61d9a8", alpha=0.88, capsize=4, edgecolor="#1f7a5c", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(parameter_names, rotation=20, ha="right")
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_title("V8 model per-parameter MAE on train split", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


@torch.inference_mode()
def compute_train_nll(model, waveform: torch.Tensor, label_normalized: torch.Tensor, device: torch.device) -> float:
    theta = label_normalized.unsqueeze(0).to(device)
    x = waveform.unsqueeze(0).to(device)
    return float((-model.log_prob(theta, x)).item())


def summarize_posterior_intervals(posterior_samples: np.ndarray, label_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q16 = np.percentile(posterior_samples, 16, axis=0)
    q84 = np.percentile(posterior_samples, 84, axis=0)
    q025 = np.percentile(posterior_samples, 2.5, axis=0)
    q975 = np.percentile(posterior_samples, 97.5, axis=0)
    covered68 = ((label_true >= q16) & (label_true <= q84)).astype(np.float64)
    covered95 = ((label_true >= q025) & (label_true <= q975)).astype(np.float64)
    ci68_width = q84 - q16
    ci95_width = q975 - q025
    return covered68, covered95, ci68_width, ci95_width


def get_parameter_range_widths(parameter_names: list[str]) -> np.ndarray:
    return np.asarray([PARAMETER_RANGES[name][1] - PARAMETER_RANGES[name][0] for name in parameter_names], dtype=np.float64)


def print_parameter_diagnostics(parameter_names: list[str], parameter_range_widths: np.ndarray, per_param_mae_mean: np.ndarray, per_param_mae_range_norm_mean: np.ndarray, per_param_coverage68: np.ndarray, per_param_coverage95: np.ndarray, per_param_ci68_width_mean: np.ndarray, per_param_ci68_width_range_norm_mean: np.ndarray, per_param_ci95_width_mean: np.ndarray, per_param_ci95_width_range_norm_mean: np.ndarray) -> None:
    header = f"{'param':<12}{'range':>12}{'mae_mean':>14}{'mae/rng':>12}{'cov68':>10}{'cov95':>10}{'ci68':>12}{'ci68/r':>10}{'ci95':>12}{'ci95/r':>10}"
    print("\nPer-parameter diagnostics:")
    print(header)
    print("-" * len(header))
    for name, range_width, mae_mean, mae_range_norm_mean, coverage68, coverage95, ci68_width_mean, ci68_width_range_norm_mean, ci95_width_mean, ci95_width_range_norm_mean in zip(
        parameter_names, parameter_range_widths, per_param_mae_mean, per_param_mae_range_norm_mean, per_param_coverage68, per_param_coverage95, per_param_ci68_width_mean, per_param_ci68_width_range_norm_mean, per_param_ci95_width_mean, per_param_ci95_width_range_norm_mean
    ):
        print(f"{name:<12}{range_width:>12.6f}{mae_mean:>14.6f}{mae_range_norm_mean:>12.3f}{coverage68:>10.3f}{coverage95:>10.3f}{ci68_width_mean:>12.6f}{ci68_width_range_norm_mean:>10.3f}{ci95_width_mean:>12.6f}{ci95_width_range_norm_mean:>10.3f}")


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_split_dataset(checkpoint, split_name="train")
    model = build_model(checkpoint, device)
    eval_count = min(NUM_TRAIN_SAMPLES, len(dataset))
    print(f"Evaluating {eval_count} train samples...")

    per_sample_param_maes = []
    per_sample_param_coverage68 = []
    per_sample_param_coverage95 = []
    per_sample_param_ci68_widths = []
    per_sample_param_ci95_widths = []
    overall_maes = []
    nll_values = []

    progress = tqdm(range(eval_count), desc="Evaluating v8 train split", unit="sample")
    for sample_index in progress:
        waveform, label_normalized = dataset[sample_index]
        label_true = inverse_label_normalization(label_normalized, checkpoint["label_mean"], checkpoint["label_std"])
        samples_normalized = sample_posterior(model, waveform, NUM_POSTERIOR_SAMPLES, device)
        posterior_samples = inverse_label_normalization(samples_normalized, checkpoint["label_mean"], checkpoint["label_std"])
        posterior_mean = posterior_samples.mean(axis=0)
        param_mae = np.abs(posterior_mean - label_true)
        param_coverage68, param_coverage95, param_ci68_width, param_ci95_width = summarize_posterior_intervals(posterior_samples, label_true)
        nll = compute_train_nll(model, waveform, label_normalized, device)
        per_sample_param_maes.append(param_mae)
        per_sample_param_coverage68.append(param_coverage68)
        per_sample_param_coverage95.append(param_coverage95)
        per_sample_param_ci68_widths.append(param_ci68_width)
        per_sample_param_ci95_widths.append(param_ci95_width)
        overall_maes.append(param_mae.mean())
        nll_values.append(nll)
        progress.set_postfix(mae=f"{np.mean(overall_maes):.2f}", nll=f"{np.mean(nll_values):.2f}")

    per_sample_param_maes_array = np.asarray(per_sample_param_maes, dtype=np.float64)
    per_sample_param_coverage68_array = np.asarray(per_sample_param_coverage68, dtype=np.float64)
    per_sample_param_coverage95_array = np.asarray(per_sample_param_coverage95, dtype=np.float64)
    per_sample_param_ci68_widths_array = np.asarray(per_sample_param_ci68_widths, dtype=np.float64)
    per_sample_param_ci95_widths_array = np.asarray(per_sample_param_ci95_widths, dtype=np.float64)
    overall_maes_array = np.asarray(overall_maes, dtype=np.float64)
    nll_values_array = np.asarray(nll_values, dtype=np.float64)
    parameter_range_widths = get_parameter_range_widths(PARAMETER_NAMES)
    per_param_mae_mean = per_sample_param_maes_array.mean(axis=0)
    per_param_mae_std = per_sample_param_maes_array.std(axis=0)
    per_param_mae_range_norm_mean = (per_sample_param_maes_array / parameter_range_widths).mean(axis=0)
    per_param_coverage68 = per_sample_param_coverage68_array.mean(axis=0)
    per_param_coverage95 = per_sample_param_coverage95_array.mean(axis=0)
    per_param_ci68_width_mean = per_sample_param_ci68_widths_array.mean(axis=0)
    per_param_ci68_width_std = per_sample_param_ci68_widths_array.std(axis=0)
    per_param_ci68_width_range_norm_mean = (per_sample_param_ci68_widths_array / parameter_range_widths).mean(axis=0)
    per_param_ci95_width_mean = per_sample_param_ci95_widths_array.mean(axis=0)
    per_param_ci95_width_std = per_sample_param_ci95_widths_array.std(axis=0)
    per_param_ci95_width_range_norm_mean = (per_sample_param_ci95_widths_array / parameter_range_widths).mean(axis=0)

    fig_path = OUTPUT_DIR / OUTPUT_FIG_NAME
    txt_path = OUTPUT_DIR / OUTPUT_TXT_NAME
    plot_parameter_mae(PARAMETER_NAMES, per_param_mae_mean, per_param_mae_std, fig_path)
    save_metrics_text(txt_path, PARAMETER_NAMES, parameter_range_widths, per_param_mae_mean, per_param_mae_std, per_param_mae_range_norm_mean, per_param_coverage68, per_param_coverage95, per_param_ci68_width_mean, per_param_ci68_width_std, per_param_ci68_width_range_norm_mean, per_param_ci95_width_mean, per_param_ci95_width_std, per_param_ci95_width_range_norm_mean, overall_maes_array, nll_values_array)
    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {txt_path}")
    print(f"Overall train MAE mean: {overall_maes_array.mean():.6f}")
    print(f"Overall train NLL mean: {nll_values_array.mean():.6f}")
    print_parameter_diagnostics(PARAMETER_NAMES, parameter_range_widths, per_param_mae_mean, per_param_mae_range_norm_mean, per_param_coverage68, per_param_coverage95, per_param_ci68_width_mean, per_param_ci68_width_range_norm_mean, per_param_ci95_width_mean, per_param_ci95_width_range_norm_mean)


if __name__ == "__main__":
    main()
