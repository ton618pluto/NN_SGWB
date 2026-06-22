from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

from eval_utils import load_model_and_dataset

try:
    from cobaya.run import run as cobaya_run
except ImportError:
    cobaya_run = None


NUM_POSTERIOR_SAMPLES = 800
DATA_VERSION = "v40"
GPU_INDEX = 1
USE_MODEL2 = True
# USE_MODEL2 = False
KDE_REG_EPS = 1e-8
USE_COBAYA_MCMC = False
COBAYA_MAX_SAMPLES = 4000
COBAYA_PROPOSAL_SCALE_FRAC = 0.08
NORMALIZED_PRIOR_ABS_MAX = 8.0


def _safe_kde(samples_2d: np.ndarray):
    try:
        return gaussian_kde(samples_2d.T)
    except Exception:
        jitter = np.random.default_rng(42).normal(scale=KDE_REG_EPS, size=samples_2d.shape)
        try:
            return gaussian_kde((samples_2d + jitter).T)
        except Exception:
            return None


def _joint_coverage_flags(
    posterior_2d: np.ndarray,
    true_2d: np.ndarray,
) -> tuple[float, float]:
    kde = _safe_kde(posterior_2d)
    if kde is None:
        return float("nan"), float("nan")

    sample_density = kde(posterior_2d.T)
    true_density = float(kde(true_2d[:, None])[0])

    level_68 = float(np.percentile(sample_density, 32.0))
    level_95 = float(np.percentile(sample_density, 5.0))

    covered68 = 1.0 if true_density >= level_68 else 0.0
    covered95 = 1.0 if true_density >= level_95 else 0.0
    return covered68, covered95


def _collection_to_array(collection, parameter_names: list[str]) -> np.ndarray:
    if hasattr(collection, "data"):
        data = collection.data
        if hasattr(data, "__getitem__"):
            return np.column_stack([np.asarray(data[name], dtype=np.float64) for name in parameter_names])
    if hasattr(collection, "to_numpy"):
        arr = np.asarray(collection.to_numpy(), dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= len(parameter_names):
            return arr[:, : len(parameter_names)]
    if hasattr(collection, "samples"):
        samples = collection.samples
        if callable(samples):
            samples = samples()
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= len(parameter_names):
            return arr[:, : len(parameter_names)]
    raise RuntimeError("Unable to convert Cobaya sample collection to numpy array.")


def main() -> None:
    if USE_COBAYA_MCMC and cobaya_run is None:
        raise ImportError("Cobaya is not installed. Please install `cobaya` or set USE_COBAYA_MCMC = False.")

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

    parameter_names = list(run_config["parameter_names"])
    norm_parameter_names = [f"norm_{name}" for name in parameter_names]
    label_mean = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std = np.asarray(run_config["label_std"], dtype=np.float64)
    pair_indices = list(combinations(range(len(parameter_names)), 2))
    pair_names = [f"{parameter_names[i]} vs {parameter_names[j]}" for i, j in pair_indices]

    cov68_values = [[] for _ in pair_indices]
    cov95_values = [[] for _ in pair_indices]
    total_nll = 0.0

    for index in tqdm(range(len(dataset)), desc="Evaluating 2D joint coverage", unit="sample"):
        data, label_norm = dataset[index]
        data = data.unsqueeze(0).to(device)
        label_norm = label_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            nll = -model.log_prob(label_norm, data).item()
        total_nll += nll
        label_true_norm = label_norm.squeeze(0).cpu().numpy()
        label_true_raw = label_true_norm * label_std + label_mean

        if USE_COBAYA_MCMC:
            def flow_target_logp(**params):
                norm = np.asarray([params[name] for name in norm_parameter_names], dtype=np.float32)
                theta = torch.tensor(norm[None, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    return float(model.log_prob(theta, data).item())

            params_info = {}
            for i, norm_name in enumerate(norm_parameter_names):
                params_info[norm_name] = {
                    "prior": {"min": -NORMALIZED_PRIOR_ABS_MAX, "max": NORMALIZED_PRIOR_ABS_MAX},
                    "ref": float(label_true_norm[i]),
                    "proposal": max(COBAYA_PROPOSAL_SCALE_FRAC, 1e-5),
                }

            info = {
                "force": True,
                "resume": False,
                "params": params_info,
                "likelihood": {
                    "flow_posterior": {
                        "external": flow_target_logp,
                        "input_params": norm_parameter_names,
                    }
                },
                "sampler": {
                    "mcmc": {
                        "max_samples": COBAYA_MAX_SAMPLES,
                        "learn_proposal": True,
                        "Rminus1_stop": 0.03,
                        "Rminus1_cl_stop": 0.2,
                        "measure_speeds": False,
                    }
                },
            }
            _, sampler = cobaya_run(info, force=True)
            posterior_norm = _collection_to_array(sampler.products()["sample"], norm_parameter_names)
            posterior_raw = posterior_norm * label_std + label_mean
        else:
            with torch.no_grad():
                posterior_norm = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)
            posterior_norm = posterior_norm.squeeze(0).detach().cpu().numpy()
            posterior_raw = posterior_norm * label_std + label_mean

        for pair_id, (i, j) in enumerate(pair_indices):
            posterior_2d = posterior_raw[:, [i, j]]
            true_2d = label_true_raw[[i, j]]
            covered68, covered95 = _joint_coverage_flags(posterior_2d, true_2d)
            cov68_values[pair_id].append(covered68)
            cov95_values[pair_id].append(covered95)

    cov68_mean = np.asarray([np.nanmean(values) for values in cov68_values], dtype=np.float64)
    cov95_mean = np.asarray([np.nanmean(values) for values in cov95_values], dtype=np.float64)
    valid_counts = np.asarray([np.sum(~np.isnan(values)) for values in cov68_values], dtype=np.int64)
    overall_nll = total_nll / len(dataset)

    figure_dir = output_dir / "test_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_set_joint_coverage_2d_{file_tag}.png"
    metrics_path = figure_dir / f"test_set_joint_coverage_2d_{file_tag}.txt"

    x = np.arange(len(pair_names))
    width = 0.35
    plt.figure(figsize=(max(8, 1.8 * len(pair_names)), 4.5))
    plt.bar(x - width / 2, cov68_mean, width=width, label="2D cov68")
    plt.bar(x + width / 2, cov95_mean, width=width, label="2D cov95")
    plt.axhline(0.68, color="tab:green", linestyle="--", linewidth=1.0)
    plt.axhline(0.95, color="tab:red", linestyle="--", linewidth=1.0)
    plt.xticks(x, pair_names, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Coverage")
    plt.title("2D joint coverage by parameter pair")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    lines = [
        f"Parameter names: {parameter_names}",
        f"Sampler: {'cobaya_mcmc' if USE_COBAYA_MCMC else 'flow_direct'}",
        f"Posterior samples per test sample: {COBAYA_MAX_SAMPLES if USE_COBAYA_MCMC else NUM_POSTERIOR_SAMPLES}",
        f"Overall test NLL mean: {overall_nll:.6f}",
        "",
        "Per-pair 2D joint coverage:",
        f"{'pair':<28}{'valid':>8}{'cov68_2d':>12}{'cov95_2d':>12}",
        "-" * 60,
    ]
    for idx, pair_name in enumerate(pair_names):
        lines.append(
            f"{pair_name:<28}{valid_counts[idx]:>8d}{cov68_mean[idx]:>12.3f}{cov95_mean[idx]:>12.3f}"
        )
    text = "\n".join(lines)

    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
