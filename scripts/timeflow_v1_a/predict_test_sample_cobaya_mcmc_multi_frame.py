from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
#BASE_VERSION = "v61_random_two_noise_100"
BASE_VERSION = "v100_random_two_noise_100"
TRAIN_DATA_VERSION = BASE_VERSION

#input_folder="processed_data_test" # this will be changed,processed_data_test is "signal",processed_data_test_superimposed is "signal+noise"
input_folder="processed_data_test_superimposed" # this will be changed,processed_data_test is "signal",processed_data_test_superimposed is "signal+noise"

#TEST_DATA_VERSION = f"fixed_one_test_{BASE_VERSION}_fixed_noise"
TEST_DATA_VERSION = f"fixed_one_test_{BASE_VERSION}"

USE_MODEL2 = False
TEST_SAMPLE_INDEX = 0
SEED = 42
# Select sample00000, then sample00009, sample00019, ..., sample00099.
FRAME_SAMPLE_INDICES = [0, 9,19,29,39,49,59,79,89,99]
NUM_FRAMES = len(FRAME_SAMPLE_INDICES)
FRAME_SPLIT_INDEX = 0
GPU_INDEX = 1
#MAX_SAMPLES = 8000
MAX_SAMPLES = 20000
#PROPOSAL_STD = 0.15
#PROPOSAL_STD = 0.10
PROPOSAL_STD = 0.08
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
SAVE_TEXT_SUMMARY = False
SAVE_COBAYA_CHAIN = False
AUTO_RESUME_COBAYA_CHAIN = False
FORCE_NEW_COBAYA_CHAIN = False
USE_POSTERIOR_CACHE = True
REBUILD_POSTERIOR_CACHE = False
PRINT_FRAME_DELTA_DIAGNOSTICS = True
#USE_INPUT_SCALE = True
USE_INPUT_SCALE = False
INPUT_SCALE = 1e23
#USE_PER_SAMPLE_DEMEAN = True
USE_PER_SAMPLE_DEMEAN = False


DATA_VERSION_RANGE_OVERRIDES = {
    "v21": {
        "alpha_m": (-3.0, 7.0),
        "delta_m": (0.0, 10.0),
        "mu_m": (25.0, 50.0),
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
    pad_frac: float = 0.15,
) -> tuple[float, float]:
    # Keep adaptive axes while leaving enough room for the 95% contour and
    # the tails of the one-dimensional KDE curves.
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


def _save_posterior_cache(
    cache_path: Path,
    posterior_norm: np.ndarray,
    metadata: dict[str, object],
) -> None:
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            posterior_norm=np.asarray(posterior_norm, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_path.replace(cache_path)


def _load_posterior_cache(
    cache_path: Path,
    expected_metadata: dict[str, object],
    expected_parameter_count: int,
) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as saved:
        posterior_norm = np.asarray(saved["posterior_norm"], dtype=np.float64)
        saved_metadata_json = str(saved["metadata_json"].item())

    if saved_metadata_json != json.dumps(expected_metadata, sort_keys=True):
        raise ValueError("posterior cache metadata does not match the current run")
    if (
        posterior_norm.ndim != 2
        or posterior_norm.shape[0] == 0
        or posterior_norm.shape[1] != expected_parameter_count
        or not np.all(np.isfinite(posterior_norm))
    ):
        raise ValueError(f"invalid cached posterior shape or values: {posterior_norm.shape}")
    return posterior_norm


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

    normalized_prior_bounds: dict[str, tuple[float, float]] = {}
    for i, parameter_name in enumerate(PARAMETER_NAMES_RUNTIME):
        physical_bounds = parameter_ranges.get(parameter_name)
        if physical_bounds is None:
            raise ValueError(f"Missing training range for parameter: {parameter_name}")
        label_std = float(label_std_array_runtime[i])
        if not np.isfinite(label_std) or label_std <= 0.0:
            raise ValueError(
                f"Invalid label std for parameter {parameter_name}: {label_std}"
            )
        physical_min, physical_max = map(float, physical_bounds)
        normalized_min = (physical_min - label_mean_array_runtime[i]) / label_std
        normalized_max = (physical_max - label_mean_array_runtime[i]) / label_std
        if not normalized_min < normalized_max:
            raise ValueError(
                f"Invalid training range for parameter {parameter_name}: "
                f"{physical_bounds}"
            )
        normalized_prior_bounds[parameter_name] = (
            float(normalized_min),
            float(normalized_max),
        )

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
    frame_selection_tag = f"spaced0_9to99_n{NUM_FRAMES}"
    run_tag = (
        f"multiframe_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_"
        f"pop{TEST_SAMPLE_INDEX:05d}_{frame_selection_tag}_"
        f"p{FRAME_SPLIT_INDEX}_trainprior"
    )
    run_result_dir = figure_dir / run_tag
    run_result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run outputs directory: {run_result_dir}")

    # The observed frames stay fixed throughout MCMC. Encoding them inside
    # flow_target_logp would rerun the CNN for every proposal, so cache the
    # context once and only evaluate the much cheaper conditional flow below.
    with torch.inference_mode():
        frame_context = model.encode(data)
    num_likelihood_frames = frame_context.shape[0]
    print(
        "Precomputed fixed frame context: "
        f"{tuple(frame_context.shape)} for {num_likelihood_frames} frames."
    )

    def flow_target_logp(**params):
        theta = torch.as_tensor(
            [[params[name] for name in norm_parameter_names]],
            dtype=torch.float32,
            device=device,
        ).expand(num_likelihood_frames, -1)
        with torch.inference_mode():
            log_probs = model.flow.log_prob(inputs=theta, context=frame_context)
            return float(log_probs.sum().item())

    params_info = {}
    for i, (parameter_name, norm_name) in enumerate(
        zip(PARAMETER_NAMES_RUNTIME, norm_parameter_names)
    ):
        prior_min, prior_max = normalized_prior_bounds[parameter_name]
        if not prior_min <= label_true_norm[i] <= prior_max:
            raise ValueError(
                f"True value for {parameter_name} is outside its training prior: "
                f"normalized truth={label_true_norm[i]:.6g}, "
                f"bounds=[{prior_min:.6g}, {prior_max:.6g}]"
            )
        params_info[norm_name] = {
            "prior": {"min": prior_min, "max": prior_max},
            "ref": float(label_true_norm[i]),
            "proposal": PROPOSAL_STD,
        }

    print("Normalized training-prior bounds:")
    for parameter_name in PARAMETER_NAMES_RUNTIME:
        prior_min, prior_max = normalized_prior_bounds[parameter_name]
        print(f"  {parameter_name}: [{prior_min:.6f}, {prior_max:.6f}]")

    chain_prefix_path = run_result_dir / "cobaya_chain"
    chain_prefix = str(chain_prefix_path)
    posterior_cache_path = run_result_dir / "posterior_samples_norm.npz"
    checkpoint_name = "best_flow_v1_a_model2.pt" if USE_MODEL2 else "best_flow_v1_a.pt"
    checkpoint_path = _build_output_dir() / checkpoint_name
    checkpoint_stat = checkpoint_path.stat()
    input_file_signatures = [
        {
            "name": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in [*h1_files, *l1_files]
    ]
    posterior_cache_metadata = {
        "format_version": 1,
        "train_data_version": TRAIN_DATA_VERSION,
        "test_data_version": TEST_DATA_VERSION,
        "input_folder": input_folder,
        "model_variant": run_config.get("model_variant", "model1"),
        "checkpoint_name": checkpoint_name,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "test_sample_index": TEST_SAMPLE_INDEX,
        "frame_sample_indices": list(FRAME_SAMPLE_INDICES),
        "frame_split_index": FRAME_SPLIT_INDEX,
        "input_files": input_file_signatures,
        "parameter_names": list(PARAMETER_NAMES_RUNTIME),
        "label_mean": label_mean_array_runtime.tolist(),
        "label_std": label_std_array_runtime.tolist(),
        "feature_mean": feature_mean.astype(np.float64).tolist(),
        "feature_std": feature_std.astype(np.float64).tolist(),
        "normalized_prior_bounds": normalized_prior_bounds,
        "max_samples": MAX_SAMPLES,
        "proposal_std": PROPOSAL_STD,
        "rminus1_stop": RMINUS1_STOP,
        "rminus1_cl_stop": RMINUS1_CL_STOP,
        "seed": SEED,
    }

    posterior_norm = None
    loaded_posterior_cache = False
    resume_chain = False
    if USE_POSTERIOR_CACHE and posterior_cache_path.exists() and not REBUILD_POSTERIOR_CACHE:
        try:
            posterior_norm = _load_posterior_cache(
                posterior_cache_path,
                posterior_cache_metadata,
                len(PARAMETER_NAMES_RUNTIME),
            )
            loaded_posterior_cache = True
            print(
                f"Loaded posterior cache: {posterior_cache_path} "
                f"({posterior_norm.shape[0]} samples); skipping Cobaya sampling."
            )
        except Exception as exc:
            print(f"Ignoring incompatible posterior cache {posterior_cache_path}: {exc}")

    if posterior_norm is None:
        if cobaya_run is None:
            raise ImportError(
                "Cobaya is not installed and no compatible posterior cache is available. "
                "Please install `cobaya` before sampling."
            )

        existing_chain_files = sorted(
            chain_prefix_path.parent.glob(f"{chain_prefix_path.name}.*")
        )
        resume_chain = bool(
            SAVE_COBAYA_CHAIN
            and AUTO_RESUME_COBAYA_CHAIN
            and existing_chain_files
            and not FORCE_NEW_COBAYA_CHAIN
        )
        force_chain = bool(SAVE_COBAYA_CHAIN and FORCE_NEW_COBAYA_CHAIN)

        if resume_chain:
            print(f"Resuming Cobaya chain from prefix: {chain_prefix}")
        elif force_chain and existing_chain_files:
            print(f"FORCE_NEW_COBAYA_CHAIN=True: overwriting chain prefix: {chain_prefix}")
        elif SAVE_COBAYA_CHAIN:
            print(f"Starting a new Cobaya chain with prefix: {chain_prefix}")

        info = {
            "force": force_chain,
            "resume": resume_chain,
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
                }
            },
        }
        if SAVE_COBAYA_CHAIN:
            info["output"] = chain_prefix

        _, sampler = cobaya_run(info)
        posterior_norm = _collection_to_array(
            sampler.products()["sample"], norm_parameter_names
        )
        if USE_POSTERIOR_CACHE:
            _save_posterior_cache(
                posterior_cache_path,
                posterior_norm,
                posterior_cache_metadata,
            )
            print(f"Saved posterior cache before plotting: {posterior_cache_path}")

    posterior = posterior_norm * label_std_array_runtime + label_mean_array_runtime

    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q50 = np.percentile(posterior, 50, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(PARAMETER_NAMES_RUNTIME)

    if PRINT_FRAME_DELTA_DIAGNOSTICS:
        joint_median_norm = np.percentile(posterior_norm, 50, axis=0)
        theta_true_batch = torch.as_tensor(
            label_true_norm,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).expand(num_likelihood_frames, -1)
        theta_joint_batch = torch.as_tensor(
            joint_median_norm,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0).expand(num_likelihood_frames, -1)

        with torch.inference_mode():
            frame_logp_true = model.flow.log_prob(
                inputs=theta_true_batch,
                context=frame_context,
            )
            frame_logp_joint = model.flow.log_prob(
                inputs=theta_joint_batch,
                context=frame_context,
            )

        frame_logp_true_np = frame_logp_true.detach().cpu().numpy()
        frame_logp_joint_np = frame_logp_joint.detach().cpu().numpy()
        frame_deltas = frame_logp_joint_np - frame_logp_true_np

        print("\nPer-frame log-probability preference diagnostics:")
        print("  delta_i = log q(theta_joint | frame_i) - log q(theta_true | frame_i)")
        print("  delta_i > 0 favors the joint posterior median; delta_i < 0 favors truth.")
        for frame_index, (h1_path, logp_true_i, logp_joint_i, delta_i) in enumerate(
            zip(h1_files, frame_logp_true_np, frame_logp_joint_np, frame_deltas),
            start=1,
        ):
            print(
                f"  frame={frame_index:03d} file={h1_path.name} "
                f"logp_true={logp_true_i:.6f} "
                f"logp_joint={logp_joint_i:.6f} delta={delta_i:+.6f}"
            )

        finite_deltas = frame_deltas[np.isfinite(frame_deltas)]
        if finite_deltas.size == 0:
            print("Frame delta summary: no finite delta values.")
        else:
            print("Frame delta summary:")
            print(f"  positive_count: {np.count_nonzero(finite_deltas > 0)}")
            print(f"  negative_count: {np.count_nonzero(finite_deltas < 0)}")
            print(f"  zero_count: {np.count_nonzero(finite_deltas == 0)}")
            print(f"  nonfinite_count: {frame_deltas.size - finite_deltas.size}")
            print(f"  mean_delta: {finite_deltas.mean():+.6f}")
            print(f"  median_delta: {np.median(finite_deltas):+.6f}")
            print(f"  total_delta: {finite_deltas.sum():+.6f}")
            print(f"  min_delta: {finite_deltas.min():+.6f}")
            print(f"  max_delta: {finite_deltas.max():+.6f}")

    figure_path = (
        run_result_dir
        / f"test_multiframe_posterior_cobaya_mcmc_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_pop{TEST_SAMPLE_INDEX:05d}_n{NUM_FRAMES}.pdf"
    )
    metrics_path = (
        run_result_dir
        / f"test_multiframe_posterior_cobaya_mcmc_{TRAIN_DATA_VERSION}_{TEST_DATA_VERSION}_{model_tag}_pop{TEST_SAMPLE_INDEX:05d}_n{NUM_FRAMES}.txt"
    )

    posterior_color = "#565A9E"
    posterior_outer_color = "#9295C8"
    truth_color = "#C9795D"
    parameter_limits = [
        _adaptive_limits(posterior[:, i], label_true[i], means[i])
        for i in range(n_params)
    ]

    with plt.rc_context(
        {
            "font.family": "serif",
            "figure.facecolor": "#FBFAFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#777780",
            "axes.linewidth": 0.9,
            "xtick.color": "#50505A",
            "ytick.color": "#50505A",
        }
    ):
        fig, axes = plt.subplots(
            n_params,
            n_params,
            figsize=(2.15 * n_params, 2.15 * n_params),
            squeeze=False,
        )

        for row in range(n_params):
            for col in range(n_params):
                ax = axes[row, col]
                if row < col:
                    ax.axis("off")
                    continue

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                x_limits = parameter_limits[col]

                if row == col:
                    samples = posterior[:, col]
                    x_grid, y_grid = _safe_kde_1d(samples, x_limits)
                    if x_grid is not None:
                        ax.plot(x_grid, y_grid, color=posterior_color, linewidth=1.8)
                    else:
                        ax.hist(
                            samples,
                            bins=40,
                            density=True,
                            histtype="step",
                            color=posterior_color,
                            linewidth=1.5,
                        )
                    ax.axvline(
                        label_true[col],
                        color=truth_color,
                        linestyle="--",
                        linewidth=1.1,
                        zorder=0,
                    )
                    ax.set_xlim(*x_limits)
                    ax.set_yticks([])
                else:
                    x = posterior[:, col]
                    y = posterior[:, row]
                    y_limits = parameter_limits[row]
                    xx, yy, zz = _safe_kde_2d(x, y, x_limits, y_limits)
                    if zz is not None:
                        level_68, level_95 = _credible_levels_from_density(zz, [0.68, 0.95])
                        contour_levels = sorted(set([level_95, level_68]))
                        ax.contour(
                            xx,
                            yy,
                            zz,
                            levels=contour_levels,
                            colors=[posterior_outer_color, posterior_color][
                                : len(contour_levels)
                            ],
                            linewidths=[1.3, 1.8][: len(contour_levels)],
                        )
                    else:
                        ax.scatter(x, y, s=3, alpha=0.12, color=posterior_color)

                    ax.axvline(
                        label_true[col],
                        color=truth_color,
                        linestyle="--",
                        linewidth=0.9,
                        zorder=0,
                    )
                    ax.axhline(
                        label_true[row],
                        color=truth_color,
                        linestyle="--",
                        linewidth=0.9,
                        zorder=0,
                    )
                    ax.scatter(
                        label_true[col],
                        label_true[row],
                        color=truth_color,
                        marker="+",
                        s=34,
                        linewidths=1.2,
                        zorder=3,
                    )
                    ax.set_xlim(*x_limits)
                    ax.set_ylim(*y_limits)

                if row == n_params - 1:
                    ax.set_xlabel(PARAMETER_NAMES_RUNTIME[col], fontsize=10)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col == 0 and row > 0:
                    ax.set_ylabel(PARAMETER_NAMES_RUNTIME[row], fontsize=10)
                elif col > 0:
                    ax.tick_params(axis="y", labelleft=False)
                ax.tick_params(labelsize=8, length=3)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                if row > col:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

        legend_handles = [
            Line2D([0], [0], color=posterior_color, linewidth=2.0, label="Posterior"),
            Line2D(
                [0],
                [0],
                color=truth_color,
                linestyle="--",
                linewidth=1.2,
                label="Truth",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            frameon=False,
            fontsize=11,
        )
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.98, wspace=0.08, hspace=0.08)
    fig.savefig(figure_path, dpi=800)
    plt.close(fig)

    lines = [
        f"Train data version: {TRAIN_DATA_VERSION}",
        f"Test data version: {TEST_DATA_VERSION}",
        f"Model variant: {'model2' if USE_MODEL2 else 'model1'}",
        f"Test sample index (pop index): {TEST_SAMPLE_INDEX}",
        f"Frame sample indices: {FRAME_SAMPLE_INDICES}",
        f"Frame split index: {FRAME_SPLIT_INDEX}",
        f"Number of frames: {NUM_FRAMES}",
        "Sampler: cobaya mcmc",
        "Target distribution: joint pure flow posterior over multiple frames",
        f"Max samples: {MAX_SAMPLES}",
        "Prior: per-parameter training ranges transformed to normalized space",
        "Normalized training-prior bounds:",
        *[
            f"  {name}: [{normalized_prior_bounds[name][0]:.8f}, "
            f"{normalized_prior_bounds[name][1]:.8f}]"
            for name in PARAMETER_NAMES_RUNTIME
        ],
        f"Proposal std (normalized space): {PROPOSAL_STD}",
        f"Rminus1_stop: {RMINUS1_STOP}",
        f"Rminus1_cl_stop: {RMINUS1_CL_STOP}",
        f"Use posterior cache: {USE_POSTERIOR_CACHE}",
        f"Loaded posterior cache: {loaded_posterior_cache}",
        f"Posterior cache: {posterior_cache_path}",
        f"Save Cobaya chain: {SAVE_COBAYA_CHAIN}",
        f"Auto-resume Cobaya chain: {AUTO_RESUME_COBAYA_CHAIN}",
        f"Resumed existing chain: {resume_chain}",
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
