from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import PARAMETER_RANGES
from eval_utils import load_model_and_dataset
NUM_POSTERIOR_SAMPLES = 800
#DATA_VERSION = "v6"
#DATA_VERSION = "v11"
#DATA_VERSION = "v21"
#DATA_VERSION = "v22"
#DATA_VERSION = "v30_s"
#DATA_VERSION = "v41"
#DATA_VERSION = "v50"
#DATA_VERSION = "v61"
#DATA_VERSION = "v61_random_two_noise"
#DATA_VERSION = "v61_random_two_noise_500"
#DATA_VERSION = "v61_random_two_noise_100"
#DATA_VERSION = "v61_500"
#DATA_VERSION = "v61_200"
#DATA_VERSION = "v62"
#DATA_VERSION = "v63"
#DATA_VERSION = "v64"
#DATA_VERSION = "v80"
#DATA_VERSION = "v81"
#DATA_VERSION = "v70"
#DATA_VERSION = "v71"
#DATA_VERSION = "v100"
DATA_VERSION = "v100_random_two_noise_100"
#DATA_VERSION = "v100_test"
#DATA_VERSION = "v10_2"
GPU_INDEX = 0
#USE_MODEL2 = True
USE_MODEL2 = False
model_name = "best_flow_v1_a_model2.pt" if USE_MODEL2 else "best_flow_v1_a.pt"
#model_name = "latest_flow_v1_a_model2.pt" if USE_MODEL2 else "latest_flow_v1_a.pt"

def summarize_posterior_intervals(posterior_samples: np.ndarray, label_true: np.ndarray):
    q16 = np.percentile(posterior_samples, 16, axis=0)
    q84 = np.percentile(posterior_samples, 84, axis=0)
    q025 = np.percentile(posterior_samples, 2.5, axis=0)
    q975 = np.percentile(posterior_samples, 97.5, axis=0)
    covered68 = ((label_true >= q16) & (label_true <= q84)).astype(np.float64)
    covered95 = ((label_true >= q025) & (label_true <= q975)).astype(np.float64)
    return q16, q84, q025, q975, covered68, covered95


def main() -> None:
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    output_name = f"outputs_dataset_{DATA_VERSION}"
    if USE_MODEL2:
        output_name += "_model2"
    output_dir = Path(__file__).resolve().parent / output_name
    checkpoint_path = output_dir / model_name
    model, dataset, run_config = load_model_and_dataset("test", checkpoint_path, device)
    parameter_names = list(run_config["parameter_names"])
    parameter_ranges = {name: PARAMETER_RANGES[name] for name in parameter_names}

    dataset_parameter_names = getattr(dataset, "parameter_names", list(run_config["parameter_names"]))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint parameter_names: {run_config['parameter_names']}")
    print(f"Dataset parameter_names: {dataset_parameter_names}")
    print(f"label_mean: {run_config['label_mean']}")
    print(f"label_std: {run_config['label_std']}")

    per_sample_mae = []
    per_sample_coverage68 = []
    per_sample_coverage95 = []
    per_sample_ci68 = []
    per_sample_ci95 = []
    total_nll = 0.0

    for index in tqdm(range(len(dataset)), desc="Evaluating test split", unit="sample"):
        data, label_norm = dataset[index]
        data = data.unsqueeze(0).to(device)
        label_norm = label_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            nll = -model.log_prob(label_norm, data).item()
            posterior_norm = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)
        total_nll += nll

        posterior_norm = posterior_norm.squeeze(0).detach().cpu().numpy()
        label_true_norm = label_norm.squeeze(0).cpu().numpy()
        posterior_raw = posterior_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])
        label_true_raw = label_true_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])
        posterior_mean_raw = posterior_raw.mean(axis=0)

        q16, q84, q025, q975, cov68, cov95 = summarize_posterior_intervals(posterior_raw, label_true_raw)
        per_sample_mae.append(np.abs(posterior_mean_raw - label_true_raw))
        per_sample_coverage68.append(cov68)
        per_sample_coverage95.append(cov95)
        per_sample_ci68.append(q84 - q16)
        per_sample_ci95.append(q975 - q025)

    mae = np.asarray(per_sample_mae)
    cov68 = np.asarray(per_sample_coverage68)
    cov95 = np.asarray(per_sample_coverage95)
    ci68 = np.asarray(per_sample_ci68)
    ci95 = np.asarray(per_sample_ci95)
    range_widths = np.asarray([parameter_ranges[name][1] - parameter_ranges[name][0] for name in parameter_names], dtype=np.float64)

    mae_mean = mae.mean(axis=0)
    cov68_mean = cov68.mean(axis=0)
    cov95_mean = cov95.mean(axis=0)
    ci68_mean = ci68.mean(axis=0)
    ci95_mean = ci95.mean(axis=0)

    figure_dir = output_dir / "test_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "test_set_param_mae_v1_a.png"
    metrics_path = figure_dir / "test_set_metrics_v1_a.txt"

    plt.figure(figsize=(8, 4))
    plt.bar(parameter_names, mae_mean / range_widths)
    plt.ylabel("MAE / range")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    overall_mae = float(mae.mean())
    overall_nll = total_nll / len(dataset)
    lines = [
        f"Overall test MAE mean: {overall_mae:.6f}",
        f"Overall test NLL mean: {overall_nll:.6f}",
        "",
        "Per-parameter diagnostics:",
        f"{'param':<15}{'range':>10}{'mae_mean':>14}{'mae/rng':>12}{'cov68':>10}{'cov95':>10}{'ci68':>12}{'ci68/r':>10}{'ci95':>12}{'ci95/r':>10}",
        "-" * 118,
    ]
    for i, name in enumerate(parameter_names):
        lines.append(
            f"{name:<15}{range_widths[i]:>10.6f}{mae_mean[i]:>14.6f}{(mae_mean[i]/range_widths[i]):>12.3f}"
            f"{cov68_mean[i]:>10.3f}{cov95_mean[i]:>10.3f}{ci68_mean[i]:>12.6f}{(ci68_mean[i]/range_widths[i]):>10.3f}"
            f"{ci95_mean[i]:>12.6f}{(ci95_mean[i]/range_widths[i]):>10.3f}"
        )
    text = "\n".join(lines)
    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
