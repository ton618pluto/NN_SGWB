from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval_utils import load_model_and_dataset

NUM_POSTERIOR_SAMPLES = 800
EVALUATION_SAMPLE_LIMIT = 17_000
DATA_VERSION = "v100_random_two_noise_100"
GPU_INDEX = 1
REG_EPS = 1e-6
SEED = 42
RESUME_PROGRESS = True
RESET_PROGRESS = False
PROGRESS_SAVE_EVERY = 1
AVERAGE_REPORT_EVERY = 100
#USE_MODEL2 = True
USE_MODEL2 = False


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


def _save_progress(
    progress_path: Path,
    distances: list[float],
    metadata: dict[str, object],
) -> None:
    temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            distances=np.asarray(distances, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_path.replace(progress_path)


def _load_progress(
    progress_path: Path,
    expected_metadata: dict[str, object],
) -> list[float]:
    try:
        with np.load(progress_path, allow_pickle=False) as saved:
            distances = np.asarray(saved["distances"], dtype=np.float64)
            saved_metadata = json.loads(str(saved["metadata_json"].item()))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read progress file {progress_path}. "
            "Set RESET_PROGRESS=True to start over."
        ) from exc

    if saved_metadata != expected_metadata:
        raise RuntimeError(
            f"Progress file is incompatible with the current evaluation settings: {progress_path}\n"
            f"Saved metadata: {saved_metadata}\n"
            f"Current metadata: {expected_metadata}\n"
            "Set RESET_PROGRESS=True to start over."
        )
    if distances.ndim != 1 or len(distances) > int(expected_metadata["dataset_size"]):
        raise RuntimeError(
            f"Invalid progress distances in {progress_path}: shape={distances.shape}. "
            "Set RESET_PROGRESS=True to start over."
        )
    return distances.tolist()


def main() -> None:
    if PROGRESS_SAVE_EVERY < 1:
        raise ValueError(f"PROGRESS_SAVE_EVERY must be at least 1, got {PROGRESS_SAVE_EVERY}")
    if AVERAGE_REPORT_EVERY < 1:
        raise ValueError(f"AVERAGE_REPORT_EVERY must be at least 1, got {AVERAGE_REPORT_EVERY}")

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
    evaluation_size = min(EVALUATION_SAMPLE_LIMIT, len(dataset))
    if evaluation_size < 1:
        raise RuntimeError("The test dataset is empty.")
    print(f"Evaluating {evaluation_size}/{len(dataset)} test samples")

    figure_dir = output_dir / "test_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_set_mahalanobis_hist_{file_tag}.png"
    metrics_path = figure_dir / f"test_set_mahalanobis_metrics_{file_tag}.txt"
    progress_path = figure_dir / f"test_set_mahalanobis_progress_{file_tag}.npz"

    checkpoint_stat = checkpoint_path.stat()
    progress_metadata = {
        "format_version": 1,
        "data_version": DATA_VERSION,
        "model_tag": file_tag,
        "checkpoint_name": checkpoint_name,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "dataset_size": len(dataset),
        "parameter_names": list(run_config["parameter_names"]),
        "label_mean": [float(value) for value in run_config["label_mean"]],
        "label_std": [float(value) for value in run_config["label_std"]],
        "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
        "regularization_epsilon": REG_EPS,
        "seed": SEED,
    }

    distances: list[float] = []
    if RESUME_PROGRESS and not RESET_PROGRESS and progress_path.exists():
        distances = _load_progress(progress_path, progress_metadata)
        print(f"Resuming from {progress_path}: {len(distances)}/{evaluation_size} target samples available")
    elif RESET_PROGRESS and progress_path.exists():
        print(f"RESET_PROGRESS=True: ignoring existing progress file {progress_path}")

    if len(distances) > evaluation_size:
        print(
            f"Progress contains {len(distances)} samples; "
            f"using only the first {evaluation_size} for this evaluation."
        )
        distances = distances[:evaluation_size]

    start_index = len(distances)
    progress = tqdm(
        range(start_index, evaluation_size),
        desc="Evaluating Mahalanobis distance",
        unit="sample",
        initial=start_index,
        total=evaluation_size,
    )
    for index in progress:
        data, label_norm = dataset[index]
        data = data.unsqueeze(0).to(device)
        label_norm = label_norm.unsqueeze(0).to(device)

        # A per-index seed makes resumed evaluation identical to an uninterrupted run.
        torch.manual_seed(SEED + index)
        with torch.no_grad():
            posterior_norm = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)

        posterior_norm = posterior_norm.squeeze(0).detach().cpu().numpy()
        label_true_norm = label_norm.squeeze(0).cpu().numpy()

        posterior_raw = posterior_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])
        label_true_raw = label_true_norm * np.asarray(run_config["label_std"]) + np.asarray(run_config["label_mean"])

        distance = compute_mahalanobis_distance(posterior_raw, label_true_raw)
        distances.append(distance)

        completed = index + 1
        if completed % PROGRESS_SAVE_EVERY == 0 or completed == evaluation_size:
            _save_progress(progress_path, distances, progress_metadata)
        progress.set_postfix_str(f"distance={distance:.6f}", refresh=True)
        if completed % AVERAGE_REPORT_EVERY == 0 or completed == evaluation_size:
            remainder = completed % AVERAGE_REPORT_EVERY
            batch_size = remainder if remainder else AVERAGE_REPORT_EVERY
            batch_start = completed - batch_size
            batch_mean = float(np.mean(distances[batch_start:completed]))
            tqdm.write(
                f"Samples {batch_start + 1}-{completed}: "
                f"mean Mahalanobis distance={batch_mean:.6f}"
            )

    distances = np.asarray(distances, dtype=np.float64)
    parameter_names = list(run_config["parameter_names"])

    plt.figure(figsize=(7, 4))
    plt.hist(distances, bins=40, color="tab:blue", alpha=0.8)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    lines = [
        f"Parameter names: {parameter_names}",
        f"Evaluated test samples: {len(distances)} / {len(dataset)}",
        f"Posterior samples per test sample: {NUM_POSTERIOR_SAMPLES}",
        f"Regularization epsilon: {REG_EPS}",
        f"Progress file: {progress_path}",
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
    print(f"Saved completed progress: {progress_path}")


if __name__ == "__main__":
    main()
