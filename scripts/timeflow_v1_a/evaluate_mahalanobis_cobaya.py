from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import PARAMETER_SPECS
from eval_utils import load_model_and_dataset

try:
    from cobaya.run import run as cobaya_run
except ImportError:
    cobaya_run = None


# This script is intentionally separate from evaluate_mahalanobis.py. It runs
# one independent Cobaya chain for each test example, so it is much slower than
# drawing posterior samples directly from the trained normalizing flow.
EVALUATION_SAMPLE_LIMIT = 100
DATA_VERSION = "v100_random_two_noise_100"
GPU_INDEX = 0
SEED = 42
REG_EPS = 1e-6

MAX_SAMPLES = 20_000
PROPOSAL_STD = 0.15
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
AVERAGE_REPORT_EVERY = 10

# A completed test example is cached immediately. On a later run, compatible
# caches are reused and Cobaya is not called again for those examples.
USE_POSTERIOR_CACHE = True
REBUILD_POSTERIOR_CACHE = False
RESUME_PROGRESS = True
RESET_PROGRESS = False

# Cobaya text chains are normally unnecessary because the compact NPZ cache is
# sufficient for the Mahalanobis calculation. Enabling this consumes more disk.
SAVE_COBAYA_CHAINS = False

#USE_MODEL2 = True
USE_MODEL2 = False


def _atomic_save_npz(path: Path, **arrays: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_path.replace(path)


def _collection_to_weighted_array(
    collection,
    parameter_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return Cobaya parameter samples and their MCMC multiplicity weights."""
    if hasattr(collection, "data"):
        data = collection.data
        if hasattr(data, "__getitem__"):
            samples = np.column_stack(
                [np.asarray(data[name], dtype=np.float64) for name in parameter_names]
            )
            try:
                weights = np.asarray(data["weight"], dtype=np.float64)
            except (KeyError, TypeError):
                weights = np.ones(samples.shape[0], dtype=np.float64)
            return samples, weights

    if hasattr(collection, "to_numpy"):
        samples = np.asarray(collection.to_numpy(), dtype=np.float64)
        if samples.ndim == 2 and samples.shape[1] >= len(parameter_names):
            samples = samples[:, : len(parameter_names)]
            return samples, np.ones(samples.shape[0], dtype=np.float64)

    raise RuntimeError("Unable to convert the Cobaya sample collection to arrays.")


def compute_weighted_mahalanobis_distance(
    posterior_samples: np.ndarray,
    weights: np.ndarray,
    label_true: np.ndarray,
    reg_eps: float = REG_EPS,
) -> float:
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    label_true = np.asarray(label_true, dtype=np.float64).reshape(-1)

    if posterior_samples.ndim != 2:
        raise ValueError(f"posterior_samples must be 2-D, got {posterior_samples.shape}")
    if posterior_samples.shape[0] != weights.size:
        raise ValueError("The number of samples and weights does not match.")
    if posterior_samples.shape[1] != label_true.size:
        raise ValueError("The posterior and truth parameter dimensions do not match.")
    if not np.all(np.isfinite(posterior_samples)) or not np.all(np.isfinite(weights)):
        raise ValueError("Posterior samples or weights contain non-finite values.")
    if np.any(weights < 0.0) or float(weights.sum()) <= 1.0:
        raise ValueError("Cobaya sample weights must be non-negative and sum to more than one.")

    weight_sum = float(weights.sum())
    mean = np.average(posterior_samples, axis=0, weights=weights)
    centered = posterior_samples - mean
    # Cobaya MCMC weights are state multiplicities. This is the covariance of
    # the weighted empirical posterior, with the usual frequency correction.
    covariance = (centered * weights[:, None]).T @ centered / (weight_sum - 1.0)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * reg_eps
    delta = label_true - mean
    distance_squared = float(delta.T @ np.linalg.pinv(covariance) @ delta)
    return float(np.sqrt(max(distance_squared, 0.0)))


def _file_signature(path_like: str | Path) -> dict[str, object]:
    path = Path(path_like)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _save_posterior_cache(
    cache_path: Path,
    posterior_norm: np.ndarray,
    weights: np.ndarray,
    metadata: dict[str, object],
) -> None:
    _atomic_save_npz(
        cache_path,
        posterior_norm=np.asarray(posterior_norm, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_posterior_cache(
    cache_path: Path,
    expected_metadata: dict[str, object],
    parameter_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as saved:
        posterior_norm = np.asarray(saved["posterior_norm"], dtype=np.float64)
        weights = np.asarray(saved["weights"], dtype=np.float64)
        saved_metadata = json.loads(str(saved["metadata_json"].item()))

    if saved_metadata != expected_metadata:
        raise ValueError("cache metadata does not match the current settings")
    if (
        posterior_norm.ndim != 2
        or posterior_norm.shape[0] == 0
        or posterior_norm.shape[1] != parameter_count
        or weights.shape != (posterior_norm.shape[0],)
    ):
        raise ValueError(
            f"invalid cached posterior/weight shapes: {posterior_norm.shape}, {weights.shape}"
        )
    if not np.all(np.isfinite(posterior_norm)) or not np.all(np.isfinite(weights)):
        raise ValueError("cached posterior or weights contain non-finite values")
    return posterior_norm, weights


def _save_progress(
    progress_path: Path,
    distances: list[float],
    chain_rows: list[int],
    weighted_samples: list[float],
    metadata: dict[str, object],
) -> None:
    _atomic_save_npz(
        progress_path,
        distances=np.asarray(distances, dtype=np.float64),
        chain_rows=np.asarray(chain_rows, dtype=np.int64),
        weighted_samples=np.asarray(weighted_samples, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_progress(
    progress_path: Path,
    expected_metadata: dict[str, object],
) -> tuple[list[float], list[int], list[float]]:
    with np.load(progress_path, allow_pickle=False) as saved:
        distances = np.asarray(saved["distances"], dtype=np.float64)
        chain_rows = np.asarray(saved["chain_rows"], dtype=np.int64)
        weighted_samples = np.asarray(saved["weighted_samples"], dtype=np.float64)
        saved_metadata = json.loads(str(saved["metadata_json"].item()))

    if saved_metadata != expected_metadata:
        raise RuntimeError(
            f"Incompatible progress file: {progress_path}. "
            "Set RESET_PROGRESS=True to start a new evaluation."
        )
    if not (distances.ndim == chain_rows.ndim == weighted_samples.ndim == 1):
        raise RuntimeError(f"Invalid progress arrays in {progress_path}.")
    if not (len(distances) == len(chain_rows) == len(weighted_samples)):
        raise RuntimeError(f"Progress array lengths do not match in {progress_path}.")
    return distances.tolist(), chain_rows.tolist(), weighted_samples.tolist()


def _normalized_training_prior_bounds(
    parameter_names: list[str],
    label_mean: np.ndarray,
    label_std: np.ndarray,
) -> np.ndarray:
    bounds = []
    for index, name in enumerate(parameter_names):
        parameter_range = PARAMETER_SPECS.get(name, {}).get("range")
        if parameter_range is None:
            raise KeyError(f"No training range is defined for parameter {name!r}.")
        lower, upper = map(float, parameter_range)
        if upper <= lower:
            raise ValueError(f"Parameter {name!r} has a degenerate training range.")
        bounds.append(
            [
                (lower - label_mean[index]) / label_std[index],
                (upper - label_mean[index]) / label_std[index],
            ]
        )
    return np.asarray(bounds, dtype=np.float64)


def main() -> None:
    if cobaya_run is None:
        raise ImportError("Cobaya is required. Install it before running this script.")
    if EVALUATION_SAMPLE_LIMIT < 1:
        raise ValueError("EVALUATION_SAMPLE_LIMIT must be at least one.")
    if AVERAGE_REPORT_EVERY < 1:
        raise ValueError("AVERAGE_REPORT_EVERY must be at least one.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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

    parameter_names = list(run_config["parameter_names"])
    normalized_names = [f"norm_{name}" for name in parameter_names]
    label_mean = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std = np.asarray(run_config["label_std"], dtype=np.float64)
    prior_bounds = _normalized_training_prior_bounds(parameter_names, label_mean, label_std)

    figure_dir = output_dir / "test_fig"
    cache_dir = figure_dir / f"cobaya_mahalanobis_cache_{file_tag}"
    chain_dir = cache_dir / "chains"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_COBAYA_CHAINS:
        chain_dir.mkdir(parents=True, exist_ok=True)

    progress_path = figure_dir / f"test_set_mahalanobis_progress_cobaya_{file_tag}.npz"
    metrics_path = figure_dir / f"test_set_mahalanobis_metrics_cobaya_{file_tag}.txt"
    figure_path = figure_dir / f"test_set_mahalanobis_hist_cobaya_{file_tag}.png"

    checkpoint_signature = _file_signature(checkpoint_path)
    progress_metadata = {
        "format_version": 1,
        "data_version": DATA_VERSION,
        "model_tag": file_tag,
        "checkpoint": checkpoint_signature,
        "dataset_size": len(dataset),
        "evaluation_size": evaluation_size,
        "parameter_names": parameter_names,
        "label_mean": label_mean.tolist(),
        "label_std": label_std.tolist(),
        "prior_bounds_norm": prior_bounds.tolist(),
        "regularization_epsilon": REG_EPS,
        "max_samples": MAX_SAMPLES,
        "proposal_std": PROPOSAL_STD,
        "rminus1_stop": RMINUS1_STOP,
        "rminus1_cl_stop": RMINUS1_CL_STOP,
        "seed": SEED,
    }

    distances: list[float] = []
    chain_rows: list[int] = []
    weighted_samples: list[float] = []
    if RESUME_PROGRESS and not RESET_PROGRESS and progress_path.exists():
        distances, chain_rows, weighted_samples = _load_progress(
            progress_path, progress_metadata
        )
        if len(distances) > evaluation_size:
            raise RuntimeError(
                f"Progress has {len(distances)} entries but evaluation_size={evaluation_size}."
            )
        print(
            f"Resuming Cobaya Mahalanobis evaluation: "
            f"{len(distances)}/{evaluation_size} samples completed."
        )
    elif RESET_PROGRESS and progress_path.exists():
        print(f"RESET_PROGRESS=True: ignoring {progress_path}")

    print(f"Device: {device}")
    print(f"Evaluating {evaluation_size}/{len(dataset)} test samples with Cobaya")
    print("Normalized training-prior bounds:")
    for name, (lower, upper) in zip(parameter_names, prior_bounds):
        print(f"  {name}: [{lower:.8f}, {upper:.8f}]")
    print(f"Completed-sample cache directory: {cache_dir}")

    progress = tqdm(
        range(len(distances), evaluation_size),
        desc="Cobaya Mahalanobis",
        unit="sample",
        initial=len(distances),
        total=evaluation_size,
    )
    for index in progress:
        data, label_true_norm_tensor = dataset[index]
        data = data.unsqueeze(0).to(device)
        label_true_norm = label_true_norm_tensor.detach().cpu().numpy().astype(np.float64)
        label_true = label_true_norm * label_std + label_mean

        # Encoding the time series once is critical: MCMC then evaluates only
        # the comparatively cheap normalizing-flow density at every proposal.
        with torch.inference_mode():
            frame_context = model.encode(data)

        record = dataset.records[index]
        sample_metadata = dict(progress_metadata)
        sample_metadata.update(
            {
                "test_index": index,
                "h1_file": _file_signature(record["h1"]),
                "l1_file": _file_signature(record["l1"]),
                "label_true_norm": label_true_norm.tolist(),
            }
        )
        cache_path = cache_dir / f"sample_{index:05d}.npz"

        posterior_norm = None
        weights = None
        if USE_POSTERIOR_CACHE and cache_path.exists() and not REBUILD_POSTERIOR_CACHE:
            try:
                posterior_norm, weights = _load_posterior_cache(
                    cache_path, sample_metadata, len(parameter_names)
                )
                tqdm.write(
                    f"sample={index:05d}: loaded cache with "
                    f"{posterior_norm.shape[0]} chain rows"
                )
            except Exception as exc:
                tqdm.write(f"sample={index:05d}: ignoring incompatible cache: {exc}")

        if posterior_norm is None or weights is None:
            np.random.seed(SEED + index)
            torch.manual_seed(SEED + index)
            with torch.inference_mode():
                initial_norm = (
                    model.flow.sample(num_samples=1, context=frame_context)
                    .reshape(-1, len(parameter_names))[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            def flow_target_logp(**params):
                theta_array = np.asarray(
                    [params[name] for name in normalized_names], dtype=np.float32
                )
                theta = torch.from_numpy(theta_array[None, :]).to(device)
                with torch.inference_mode():
                    value = model.flow.log_prob(inputs=theta, context=frame_context)
                return float(value.item())

            params_info = {}
            for parameter_index, normalized_name in enumerate(normalized_names):
                lower, upper = prior_bounds[parameter_index]
                # Cobaya requires the reference point to have non-zero prior.
                # Keep it strictly inside the uniform-prior bounds instead of
                # clipping exactly onto a boundary, which Cobaya may reject.
                margin = max(1e-6, float(upper - lower) * 1e-6)
                initial_value = float(initial_norm[parameter_index])
                if np.isfinite(initial_value):
                    reference = float(
                        np.clip(initial_value, lower + margin, upper - margin)
                    )
                else:
                    reference = float((lower + upper) / 2.0)
                params_info[normalized_name] = {
                    "prior": {"min": float(lower), "max": float(upper)},
                    # Start from a native Flow draw rather than the known truth,
                    # so the calibration label does not initialize the chain.
                    "ref": reference,
                    "proposal": PROPOSAL_STD,
                }

            info = {
                "force": True,
                "resume": False,
                "params": params_info,
                "likelihood": {
                    "flow_posterior_single_frame": {
                        "external": flow_target_logp,
                        "input_params": normalized_names,
                    }
                },
                "sampler": {
                    "mcmc": {
                        "max_samples": MAX_SAMPLES,
                        "learn_proposal": True,
                        "Rminus1_stop": RMINUS1_STOP,
                        "Rminus1_cl_stop": RMINUS1_CL_STOP,
                        "measure_speeds": False,
                    }
                },
            }
            if SAVE_COBAYA_CHAINS:
                info["output"] = str(chain_dir / f"sample_{index:05d}")

            _, sampler = cobaya_run(info, force=True)
            posterior_norm, weights = _collection_to_weighted_array(
                sampler.products()["sample"], normalized_names
            )
            if USE_POSTERIOR_CACHE:
                _save_posterior_cache(
                    cache_path, posterior_norm, weights, sample_metadata
                )
                tqdm.write(f"sample={index:05d}: saved posterior cache {cache_path}")

        posterior = posterior_norm * label_std[None, :] + label_mean[None, :]
        distance = compute_weighted_mahalanobis_distance(
            posterior, weights, label_true, REG_EPS
        )
        distances.append(distance)
        chain_rows.append(int(posterior_norm.shape[0]))
        weighted_samples.append(float(weights.sum()))
        _save_progress(
            progress_path,
            distances,
            chain_rows,
            weighted_samples,
            progress_metadata,
        )
        progress.set_postfix_str(
            f"distance={distance:.6f}, rows={posterior_norm.shape[0]}", refresh=True
        )
        tqdm.write(
            f"Completed {index + 1}/{evaluation_size}: "
            f"Mahalanobis distance={distance:.6f}"
        )
        completed = index + 1
        if completed % AVERAGE_REPORT_EVERY == 0:
            batch_start = completed - AVERAGE_REPORT_EVERY
            batch_mean = float(np.mean(distances[batch_start:completed]))
            tqdm.write(
                f"Samples {batch_start + 1}-{completed}: "
                f"mean Mahalanobis distance={batch_mean:.6f}"
            )

    distances_array = np.asarray(distances, dtype=np.float64)
    dimension = len(parameter_names)
    theoretical_mean = math.sqrt(2.0) * math.gamma((dimension + 1.0) / 2.0) / math.gamma(
        dimension / 2.0
    )

    plt.figure(figsize=(7.2, 4.4))
    plt.hist(distances_array, bins=min(20, max(5, evaluation_size // 5)), color="#6267A5", alpha=0.82)
    plt.axvline(
        distances_array.mean(), color="#C9795D", linewidth=1.8, label="Cobaya sample mean"
    )
    plt.axvline(
        theoretical_mean,
        color="#4F7F73",
        linewidth=1.6,
        linestyle="--",
        label=f"Chi({dimension}) theoretical mean",
    )
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=500, bbox_inches="tight")
    plt.close()

    lines = [
        f"Parameter names: {parameter_names}",
        f"Evaluated test samples: {len(distances_array)} / {len(dataset)}",
        "Posterior source: independent Cobaya MCMC per test sample",
        "Cobaya sample weights used: Yes",
        "Prior: per-parameter physical training ranges transformed to normalized space",
        f"Maximum MCMC samples per test sample: {MAX_SAMPLES}",
        f"Rminus1_stop: {RMINUS1_STOP}",
        f"Rminus1_cl_stop: {RMINUS1_CL_STOP}",
        f"Regularization epsilon: {REG_EPS}",
        f"Progress file: {progress_path}",
        f"Posterior cache directory: {cache_dir}",
        "",
        f"Mean Mahalanobis distance: {distances_array.mean():.6f}",
        f"Std Mahalanobis distance: {distances_array.std():.6f}",
        f"Median Mahalanobis distance: {np.median(distances_array):.6f}",
        f"Min Mahalanobis distance: {distances_array.min():.6f}",
        f"Max Mahalanobis distance: {distances_array.max():.6f}",
        f"68th percentile: {np.percentile(distances_array, 68):.6f}",
        f"95th percentile: {np.percentile(distances_array, 95):.6f}",
        f"Chi({dimension}) theoretical mean: {theoretical_mean:.6f}",
        f"Mean stored chain rows: {np.mean(chain_rows):.1f}",
        f"Mean sum of Cobaya weights: {np.mean(weighted_samples):.1f}",
    ]
    text = "\n".join(lines)
    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Saved histogram: {figure_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved progress: {progress_path}")


if __name__ == "__main__":
    main()
