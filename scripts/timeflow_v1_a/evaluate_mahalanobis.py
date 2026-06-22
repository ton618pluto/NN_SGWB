from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval_utils import load_model_and_dataset

NUM_POSTERIOR_SAMPLES = 800
DATA_VERSION = "v40"
GPU_INDEX = 1
REG_EPS = 1e-6
USE_MODEL2 = True
#USE_MODEL2 = False


def compute_mahalanobis_distance(
    posterior_samples: np.ndarray,
    label_true: np.ndarray,
    reg_eps: float = REG_EPS,
) -> float:
    mean = posterior_samples.mean(axis=0)
    covariance = np.cov(posterior_samples, rowvar=False)
    covariance = np.atleast_2d(covariance).astype(np.float64)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * reg_eps
    delta = (label_true - mean).astype(np.float64)
    inv_covariance = np.linalg.pinv(covariance)
    distance_squared = float(delta.T @ inv_covariance @ delta)
    return float(np.sqrt(max(distance_squared, 0.0)))


def main() -> None:
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    output_name = f"outputs_dataset_{DATA_VERSION}"
    checkpoint_name = "best_flow_v1_a.pt"
    file_tag = "v1_a"
    if USE_MODEL2:
        output_name += "_model2"
        checkpoint_name = "best_flow_v1_a_model2.pt"
        file_tag = "v1_a_model2"
    output_dir = Path(__file__).resolve().parent / output_name
    checkpoint_path = output_dir / checkpoint_name
    model, dataset, run_config = load_model_and_dataset("test", checkpoint_path, device)

    distances = []

    for index in tqdm(range(len(dataset)), desc="Evaluating Mahalanobis distance", unit="sample"):
        data, label_norm = dataset[index]
        data = data.unsqueeze(0).to(device)
        label_norm = label_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            posterior_norm = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)

        posterior_norm = posterior_norm.squeeze(0).detach().cpu().numpy()
        label_true_norm = label_norm.squeeze(0).cpu().numpy()

        posterior_raw = posterior_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])
        label_true_raw = label_true_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])

        distance = compute_mahalanobis_distance(posterior_raw, label_true_raw)
        distances.append(distance)

    distances = np.asarray(distances, dtype=np.float64)
    parameter_names = list(run_config["parameter_names"])

    figure_dir = output_dir / "test_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_set_mahalanobis_hist_{file_tag}.png"
    metrics_path = figure_dir / f"test_set_mahalanobis_metrics_{file_tag}.txt"

    plt.figure(figsize=(7, 4))
    plt.hist(distances, bins=40, color="tab:blue", alpha=0.8)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    lines = [
        f"Parameter names: {parameter_names}",
        f"Posterior samples per test sample: {NUM_POSTERIOR_SAMPLES}",
        f"Regularization epsilon: {REG_EPS}",
        "",
        f"Mean Mahalanobis distance: {distances.mean():.6f}",
        f"Std Mahalanobis distance: {distances.std():.6f}",
        f"Median Mahalanobis distance: {np.median(distances):.6f}",
        f"Min Mahalanobis distance: {distances.min():.6f}",
        f"Max Mahalanobis distance: {distances.max():.6f}",
        f"68th percentile: {np.percentile(distances, 68):.6f}",
        f"95th percentile: {np.percentile(distances, 95):.6f}",
    ]
    text = "\n".join(lines)
    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
