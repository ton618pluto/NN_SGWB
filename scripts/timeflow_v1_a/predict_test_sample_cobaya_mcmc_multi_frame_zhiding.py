from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from dataset import GWDatasetV1A, PARAMETER_SPECS
from eval_utils import load_run_config
from model import GWFlowModelV1A
from model2 import GWFlowModelV1A_Model2

try:
    from dataset import get_parameter_ranges as _dataset_get_parameter_ranges
except ImportError:
    _dataset_get_parameter_ranges = None

try:
    from cobaya.run import run as cobaya_run
except ImportError:
    cobaya_run = None


#BASE_VERSION = "v61"
#BASE_VERSION = "v64"
#BASE_VERSION = "v80"
#BASE_VERSION = "v81"
#BASE_VERSION = "v70"
#BASE_VERSION = "v71"
#BASE_VERSION = "v100"
BASE_VERSION = "v100_random_two_noise_100"
TRAIN_DATA_VERSION = BASE_VERSION

#input_folder="processed_data_test" # this will be changed,processed_data_test is "signal",processed_data_test_superimposed is "signal+noise"
input_folder="processed_data_test_superimposed" # this will be changed,processed_data_test is "signal",processed_data_test_superimposed is "signal+noise"

#TEST_DATA_VERSION = f"fixed_one_test_{BASE_VERSION}_fixed_noise"
TEST_DATA_VERSION = f"fixed_one_test_{BASE_VERSION}"

USE_MODEL2 = False
TEST_SAMPLE_INDEX = 0
# Select the exact `sampleXXXXX` records and `_pN` split used in the joint
# likelihood. All selected records must belong to TEST_SAMPLE_INDEX (pop index)
# and share the same population hyperparameters.
FRAME_SAMPLE_INDICES = [0, 2]
FRAME_SPLIT_INDEX = 0
GPU_INDEX = 1
SEED = 42
#MAX_SAMPLES = 8000
MAX_SAMPLES = 20000
NORMALIZED_PRIOR_ABS_MAX = 8.0
#PROPOSAL_STD = 0.15
#PROPOSAL_STD = 0.10
PROPOSAL_STD = 0.08
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
SAVE_TEXT_SUMMARY = False
SAVE_COBAYA_CHAIN = False
#USE_INPUT_SCALE = True
USE_INPUT_SCALE = False
INPUT_SCALE = 1e23
#USE_PER_SAMPLE_DEMEAN = True
USE_PER_SAMPLE_DEMEAN = False


DATA_VERSION_RANGE_OVERRIDES = {
    "v21": {
        "alpha_m": (-3.0, 7.0),
        "delta_m": (0.0, 10.0),
        "mu_m": (25.0, 100.0),
        "sigma_m": (0.01, 5.0),
        "zp": (1.85, 2.9),
        "alpha_z": (2.7, 2.7),
        "beta_z": (2.9, 2.9),
        "m_max": (100.0, 100.0),
        "m_min": (2.0, 2.0),
        "lambda_peak": (0.04, 0.04),
        "beta_q": (1.08, 1.08),
    }
}

FULL_PARAMETER_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)


def get_parameter_ranges(
    data_version: str | None,
    parameter_names: list[str],
) -> dict[str, tuple[float, float] | None]:
    if _dataset_get_parameter_ranges is not None:
        return _dataset_get_parameter_ranges(data_version, parameter_names)

    ranges = {name: spec["range"] for name, spec in PARAMETER_SPECS.items()}
    if data_version in DATA_VERSION_RANGE_OVERRIDES:
        ranges.update(DATA_VERSION_RANGE_OVERRIDES[data_version])
    return {name: ranges[name] for name in parameter_names}


def _expand_if_degenerate(lo: float, hi: float, frac: float = 0.02) -> tuple[float, float]:
    if hi > lo:
        return lo, hi
    pad = max(abs(lo) * frac, 1e-6)
    return lo - pad, hi + pad


def _adaptive_limits(
    samples: np.ndarray,
    true_value: float,
    mean_value: float,
    q_low: float = 0.5,
    q_high: float = 99.5,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    lo = float(np.percentile(samples, q_low))
    hi = float(np.percentile(samples, q_high))
    lo = min(lo, float(true_value), float(mean_value))
    hi = max(hi, float(true_value), float(mean_value))
    lo, hi = _expand_if_degenerate(lo, hi)
    width = hi - lo
    pad = max(width * pad_frac, 1e-6)
    return lo - pad, hi + pad


def _range_based_limits(
    samples: np.ndarray,
    true_value: float,
    mean_value: float,
    parameter_range: tuple[float, float] | None,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    if parameter_range is None:
        return _adaptive_limits(samples, true_value, mean_value)

    lo, hi = map(float, parameter_range)
    lo, hi = _expand_if_degenerate(lo, hi)
    width = hi - lo
    sample_lo = float(np.min(samples))
    sample_hi = float(np.max(samples))
    sample_width = max(sample_hi - sample_lo, 1e-6)
    pad = max(width * pad_frac, sample_width * 0.08, 1e-6)
    lo -= pad
    hi += pad

    lo = min(lo, sample_lo, float(true_value), float(mean_value))
    hi = max(hi, sample_hi, float(true_value), float(mean_value))
    return lo, hi


def _safe_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
):
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return None, None, None
    x_lo, x_hi = _expand_if_degenerate(*x_limits)
    y_lo, y_hi = _expand_if_degenerate(*y_limits)
    x_grid = np.linspace(x_lo, x_hi, 160)
    y_grid = np.linspace(y_lo, y_hi, 160)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _safe_kde_1d(samples: np.ndarray, x_limits: tuple[float, float]):
    try:
        kde = gaussian_kde(samples)
    except Exception:
        return None, None
    x_lo, x_hi = _expand_if_degenerate(*x_limits)
    x_grid = np.linspace(x_lo, x_hi, 240)
    y_grid = kde(x_grid)
    return x_grid, y_grid


def _credible_levels_from_density(zz: np.ndarray, probs: list[float]) -> list[float]:
    flat = np.ravel(zz)
    order = np.argsort(flat)[::-1]
    sorted_density = flat[order]
    cumsum = np.cumsum(sorted_density)
    cumsum = cumsum / cumsum[-1]

    levels = []
    for prob in probs:
        idx = np.searchsorted(cumsum, prob, side="left")
        idx = min(idx, len(sorted_density) - 1)
        levels.append(sorted_density[idx])
    return levels


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


def _build_output_dir() -> Path:
    output_name = f"outputs_dataset_{TRAIN_DATA_VERSION}"
    if USE_MODEL2:
        output_name += "_model2"
    return Path(__file__).resolve().parent / output_name


def _load_model(device: torch.device):
    output_dir = _build_output_dir()
    checkpoint_name = "best_flow_v1_a_model2.pt" if USE_MODEL2 else "best_flow_v1_a.pt"
    checkpoint_path = output_dir / checkpoint_name
    run_config = load_run_config(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_variant = run_config.get("model_variant", "model1")
    model_cls = GWFlowModelV1A_Model2 if model_variant == "model2" else GWFlowModelV1A
    model = model_cls(
        param_dim=len(run_config["parameter_names"]),
        in_channels=2,
        context_dim=run_config["context_dim"],
        flow_layers=run_config["flow_layers"],
        flow_hidden_features=run_config["flow_hidden"],
        flow_bins=run_config["flow_bins"],
        flow_tail_bound=run_config["flow_tail_bound"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, run_config


def _label_to_full_hyperparameters(label) -> dict[str, float]:
    if isinstance(label, dict):
        return {name: float(label[name]) for name in FULL_PARAMETER_NAMES}
    label_array = np.asarray(label, dtype=np.float64).reshape(-1)
    if label_array.size != len(FULL_PARAMETER_NAMES):
        raise ValueError(
            f"Unexpected label length: {label_array.size}. "
            f"Expected {len(FULL_PARAMETER_NAMES)}"
        )
    return {name: float(value) for name, value in zip(FULL_PARAMETER_NAMES, label_array)}


def _load_multi_frame_data(
    processed_test_version: str,
    test_sample_index: int,
    frame_sample_indices: list[int],
    frame_split_index: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    label_mean: np.ndarray,
    label_std: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, dict[str, float], list[Path], list[Path]]:
    base_dir = Path(__file__).resolve().parents[1] / input_folder / processed_test_version
    h1_dir = base_dir / "H1_splits"
    l1_dir = base_dir / "L1_splits"
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 split directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 split directory not found: {l1_dir}")

    if not frame_sample_indices:
        raise ValueError("FRAME_SAMPLE_INDICES must contain at least one sample index.")
    if len(set(frame_sample_indices)) != len(frame_sample_indices):
        raise ValueError(f"FRAME_SAMPLE_INDICES contains duplicates: {frame_sample_indices}")
    if any(index < 0 for index in frame_sample_indices):
        raise ValueError(f"FRAME_SAMPLE_INDICES must be non-negative: {frame_sample_indices}")
    if frame_split_index < 0:
        raise ValueError(f"FRAME_SPLIT_INDEX must be non-negative: {frame_split_index}")

    h1_files = [
        h1_dir
        / (
            f"H1-SUPERIMPOSED-pop{test_sample_index:05d}_"
            f"sample{sample_index:05d}_p{frame_split_index}.npy"
        )
        for sample_index in frame_sample_indices
    ]
    missing_h1 = [path for path in h1_files if not path.exists()]
    if missing_h1:
        raise FileNotFoundError(f"Missing selected H1 files: {missing_h1}")
    l1_files = [l1_dir / path.name.replace("H1-", "L1-", 1) for path in h1_files]
    missing_l1 = [path for path in l1_files if not path.exists()]
    if missing_l1:
        raise FileNotFoundError(f"Missing matching L1 files: {missing_l1}")

    features_list = []
    full_hyperparameters = None
    for h1_path, l1_path in zip(h1_files, l1_files):
        h1_sample = np.load(h1_path, allow_pickle=True).item()
        l1_sample = np.load(l1_path, allow_pickle=True).item()
        h1 = np.asarray(h1_sample["data"], dtype=np.float32)
        l1 = np.asarray(l1_sample["data"], dtype=np.float32)
        if USE_INPUT_SCALE:
            h1 = h1 * INPUT_SCALE
            l1 = l1 * INPUT_SCALE
        if USE_PER_SAMPLE_DEMEAN:
            h1 = h1 - h1.mean(dtype=np.float32)
            l1 = l1 - l1.mean(dtype=np.float32)
        features = np.stack([h1, l1], axis=0)
        features = (features - feature_mean[:, None]) / feature_std[:, None]
        features_list.append(features.astype(np.float32))

        current_full_hyperparameters = _label_to_full_hyperparameters(h1_sample["label"])
        if full_hyperparameters is None:
            full_hyperparameters = current_full_hyperparameters
        else:
            inconsistent_names = [
                name
                for name in FULL_PARAMETER_NAMES
                if not np.isclose(
                    current_full_hyperparameters[name],
                    full_hyperparameters[name],
                    rtol=1e-7,
                    atol=1e-10,
                )
            ]
            if inconsistent_names:
                raise ValueError(
                    "Selected frames do not share the same hyperparameters. "
                    f"File {h1_path} differs in: {inconsistent_names}"
                )

    if full_hyperparameters is None:
        raise RuntimeError("Failed to load hyperparameters from processed test files.")

    selected_names = list(label_mean.keys()) if isinstance(label_mean, dict) else None
    label_true = np.asarray(
        [full_hyperparameters[name] for name in PARAMETER_NAMES_RUNTIME],
        dtype=np.float64,
    )
    label_true_norm = (label_true - label_mean_array_runtime) / label_std_array_runtime
    data_tensor = torch.from_numpy(np.stack(features_list, axis=0))
    return data_tensor, label_true_norm, full_hyperparameters, h1_files, l1_files


PARAMETER_NAMES_RUNTIME: list[str] = []
label_mean_array_runtime: np.ndarray
label_std_array_runtime: np.ndarray


def main() -> None:
    global PARAMETER_NAMES_RUNTIME, label_mean_array_runtime, label_std_array_runtime

    if cobaya_run is None:
        raise ImportError("Cobaya is not installed. Please install `cobaya` before running this script.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    model, run_config = _load_model(device)
    PARAMETER_NAMES_RUNTIME = list(run_config["parameter_names"])
    norm_parameter_names = [f"norm_{name}" for name in PARAMETER_NAMES_RUNTIME]
    parameter_ranges = get_parameter_ranges(TRAIN_DATA_VERSION, PARAMETER_NAMES_RUNTIME)

    label_mean_array_runtime = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std_array_runtime = np.asarray(run_config["label_std"], dtype=np.float64)
    feature_mean = np.asarray(run_config["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(run_config["feature_std"], dtype=np.float32)

    data, label_true_norm, full_hyperparameters, h1_files, l1_files = _load_multi_frame_data(
        processed_test_version=TEST_DATA_VERSION,
        test_sample_index=TEST_SAMPLE_INDEX,
        frame_sample_indices=FRAME_SAMPLE_INDICES,
        frame_split_index=FRAME_SPLIT_INDEX,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_mean=label_mean_array_runtime,
        label_std=label_std_array_runtime,
    )
    data = data.to(device)
    label_true = label_true_norm * label_std_array_runtime + label_mean_array_runtime

    figure_dir = Path(__file__).resolve().parent / "result_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "a_model2" if USE_MODEL2 else "a"
    num_frames = len(FRAME_SAMPLE_INDICES)
    sample_tag = "-".join(str(index) for index in FRAME_SAMPLE_INDICES)

    def flow_target_logp(**params):
        theta_norm = np.asarray([params[name] for name in norm_parameter_names], dtype=np.float32)
        theta_batch = np.repeat(theta_norm[None, :], repeats=data.shape[0], axis=0)
        theta = torch.tensor(theta_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            log_probs = model.log_prob(theta, data)
            return float(log_probs.sum().item())

    params_info = {}
    for i, norm_name in enumerate(norm_parameter_names):
        params_info[norm_name] = {
            "prior": {"min": -NORMALIZED_PRIOR_ABS_MAX, "max": NORMALIZED_PRIOR_ABS_MAX},
            "ref": float(label_true_norm[i]),
            "proposal": PROPOSAL_STD,
        }

    chain_prefix = str(
        figure_dir
        / f"cobaya_chain_multiframe_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_pop{TEST_SAMPLE_INDEX:05d}_samples{sample_tag}_p{FRAME_SPLIT_INDEX}"
    )
    info = {
        "force": True,
        "resume": False,
        "params": params_info,
        "likelihood": {
            "flow_posterior_multi_frame": {
                "external": flow_target_logp,
                "input_params": norm_parameter_names,
            }
        },
        "sampler": {
            "mcmc": {
                "max_samples": MAX_SAMPLES,
                "learn_proposal": True,
                "Rminus1_stop": RMINUS1_STOP,
                "Rminus1_cl_stop": RMINUS1_CL_STOP,
                "measure_speeds": False,
                #"proposal_scale": 0.5  # this will be changed
            }
        },
    }
    if SAVE_COBAYA_CHAIN:
        info["output"] = chain_prefix

    _, sampler = cobaya_run(info, force=True)
    posterior_norm = _collection_to_array(sampler.products()["sample"], norm_parameter_names)
    posterior = posterior_norm * label_std_array_runtime + label_mean_array_runtime

    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q50 = np.percentile(posterior, 50, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(PARAMETER_NAMES_RUNTIME)

    figure_path = (
        figure_dir
        / f"test_multiframe_posterior_cobaya_mcmc_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_pop{TEST_SAMPLE_INDEX:05d}_samples{sample_tag}_p{FRAME_SPLIT_INDEX}.png"
    )
    metrics_path = (
        figure_dir
        / f"test_multiframe_posterior_cobaya_mcmc_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_pop{TEST_SAMPLE_INDEX:05d}_samples{sample_tag}_p{FRAME_SPLIT_INDEX}.txt"
    )

    fig, axes = plt.subplots(n_params, n_params, figsize=(3.2 * n_params, 3.2 * n_params))
    if n_params == 1:
        axes = np.asarray([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                x_limits = _range_based_limits(
                    samples,
                    label_true[col],
                    means[col],
                    parameter_ranges.get(PARAMETER_NAMES_RUNTIME[col]),
                )
                x_grid, y_grid = _safe_kde_1d(samples, x_limits)
                if x_grid is not None:
                    ax.plot(x_grid, y_grid, color="#4DAA57", linewidth=1.8)
                else:
                    ax.hist(samples, bins=40, density=True, color="#4DAA57", alpha=0.35)
                ax.axvline(label_true[col], color="red", linestyle="--", linewidth=1.0)
                ax.set_title(PARAMETER_NAMES_RUNTIME[col])
                ax.set_xlim(*x_limits)
                ax.set_yticks([])
                summary_text = (
                    f"true={label_true[col]:.3f}\n"
                    f"mean={means[col]:.3f}\n"
                    f"{q50[col]:.3f}"
                    f"^(+{(q84[col] - q50[col]):.3f})"
                    f"_(-{(q50[col] - q16[col]):.3f})\n"
                    f"68% CI"
                )
                ax.text(
                    0.03,
                    0.97,
                    summary_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8.5,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#BDBDBD",
                        "alpha": 0.9,
                    },
                )
            elif row > col:
                x = posterior[:, col]
                y = posterior[:, row]
                x_limits = _range_based_limits(
                    x,
                    label_true[col],
                    means[col],
                    parameter_ranges.get(PARAMETER_NAMES_RUNTIME[col]),
                )
                y_limits = _range_based_limits(
                    y,
                    label_true[row],
                    means[row],
                    parameter_ranges.get(PARAMETER_NAMES_RUNTIME[row]),
                )
                xx, yy, zz = _safe_kde_2d(x, y, x_limits, y_limits)
                if zz is not None:
                    ax.contourf(xx, yy, zz, levels=14, cmap="Greens")
                    level_68, level_95 = _credible_levels_from_density(zz, [0.68, 0.95])
                    contour_levels = sorted(set([level_95, level_68]))
                    ax.contour(
                        xx,
                        yy,
                        zz,
                        levels=contour_levels,
                        colors=["#3F8F4A", "#1E5F2B"][: len(contour_levels)],
                        linewidths=[1.0, 1.4][: len(contour_levels)],
                    )
                else:
                    ax.scatter(x, y, s=4, alpha=0.15, color="#4DAA57")
                ax.scatter(label_true[col], label_true[row], color="red", marker="x", s=60, linewidths=2)
                ax.scatter(means[col], means[row], color="black", marker="o", s=18)
                ax.set_xlim(*_expand_if_degenerate(*x_limits))
                ax.set_ylim(*_expand_if_degenerate(*y_limits))
            else:
                ax.axis("off")

            if row == n_params - 1 and col <= row:
                ax.set_xlabel(PARAMETER_NAMES_RUNTIME[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(PARAMETER_NAMES_RUNTIME[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])

    fig.suptitle(
        f"Posterior Cobaya MCMC corner plot (multi-frame test, train={TRAIN_DATA_VERSION}, test={TEST_DATA_VERSION}, n={num_frames})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Train data version: {TRAIN_DATA_VERSION}",
        f"Test data version: {TEST_DATA_VERSION}",
        f"Model variant: {'model2' if USE_MODEL2 else 'model1'}",
        f"Test sample index (pop index): {TEST_SAMPLE_INDEX}",
        f"Frame sample indices: {FRAME_SAMPLE_INDICES}",
        f"Frame split index: {FRAME_SPLIT_INDEX}",
        f"Number of frames: {num_frames}",
        "Sampler: cobaya mcmc",
        "Target distribution: joint pure flow posterior over multiple frames",
        f"Max samples: {MAX_SAMPLES}",
        f"Normalized prior range: [-{NORMALIZED_PRIOR_ABS_MAX}, {NORMALIZED_PRIOR_ABS_MAX}]",
        f"Proposal std (normalized space): {PROPOSAL_STD}",
        f"Rminus1_stop: {RMINUS1_STOP}",
        f"Rminus1_cl_stop: {RMINUS1_CL_STOP}",
        f"Chain prefix: {chain_prefix}",
        "H1 files:",
        *[f"  {path}" for path in h1_files],
        "L1 files:",
        *[f"  {path}" for path in l1_files],
        "",
        "All hyperparameters from source label:",
    ]
    for name in FULL_PARAMETER_NAMES:
        lines.append(f"  {name}: {full_hyperparameters[name]}")
    lines.append("")
    for i, name in enumerate(PARAMETER_NAMES_RUNTIME):
        in68 = "Y" if q16[i] <= label_true[i] <= q84[i] else "N"
        lines.append(
            f"{name:<12} true={label_true[i]:.6f} median={q50[i]:.6f} "
            f"plus={q84[i]-q50[i]:.6f} minus={q50[i]-q16[i]:.6f} "
            f"ci16={q16[i]:.6f} ci84={q84[i]:.6f} in68={in68}"
        )
    text = "\n".join(lines)

    print(text)
    if SAVE_TEXT_SUMMARY:
        metrics_path.write_text(text, encoding="utf-8")
        print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
