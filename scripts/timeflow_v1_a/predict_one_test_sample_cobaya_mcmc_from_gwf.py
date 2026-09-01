from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gwpy.timeseries import TimeSeries

from dataset import PARAMETER_SPECS, USE_INPUT_SCALE, INPUT_SCALE, USE_PER_SAMPLE_DEMEAN
from eval_utils import load_run_config, resolve_split_manifest_path
from model import GWFlowModelV1A
from model2 import GWFlowModelV1A_Model2
from predict_one_test_sample_cobaya_mcmc import (
    _collection_to_array,
    _credible_levels_from_density,
    _expand_if_degenerate,
    _range_based_limits,
    _safe_kde_1d,
    _safe_kde_2d,
    cobaya_run,
    get_parameter_ranges,
)


# Model checkpoint and its original test split manifest.
MODEL_DATA_VERSION = "v100"
USE_MODEL2 = False

# Source GWF location. Set this to the training_set version that generated the
# processed_data_train version used to train the checkpoint.
GWF_DATA_VERSION = MODEL_DATA_VERSION
GWF_FOLDER_NAME = "training_set0"
GWF_NUM_SPLITS = 1  # Must equal processData.py NUM_SPLITS used for training data.
SPLIT_INDEX_OVERRIDE: int | None = None  # None: recover _p<index> from the old NPY name.

START_SAMPLE_INDEX = 10   # this will be changed
NUM_TEST_SAMPLES = 1 # this will be changed
GPU_INDEX = 1
SEED = 42
MAX_SAMPLES = 20000
NORMALIZED_PRIOR_ABS_MAX = 8.0
PROPOSAL_STD = 0.15
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
SAVE_TEXT_SUMMARY = False
SAVE_COBAYA_CHAIN = False
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"


SPLIT_PATTERN = re.compile(r"_p(\d+)$")


def load_model_and_test_records(checkpoint_path: Path, device: torch.device):
    output_dir = checkpoint_path.parent
    run_config = load_run_config(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split_manifest_path = resolve_split_manifest_path(run_config, output_dir)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

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
    return model, list(split_manifest["test_records"]), run_config, split_manifest_path


def parse_gwf_stem(npy_path: str, detector: str) -> tuple[str, int]:
    stem = Path(npy_path).stem
    match = SPLIT_PATTERN.search(stem)
    split_index = int(match.group(1)) if match is not None else 0
    gwf_stem = stem[: match.start()] if match is not None else stem
    prefix = f"{detector}-"
    if not gwf_stem.startswith(prefix):
        raise ValueError(f"Expected {detector} NPY filename, got: {Path(npy_path).name}")
    return gwf_stem, split_index


def build_gwf_path(scripts_dir: Path, detector: str, gwf_stem: str) -> Path:
    return (
        scripts_dir
        / "training_set"
        / GWF_DATA_VERSION
        / GWF_FOLDER_NAME
        / detector
        / f"{gwf_stem}.gwf"
    )


def load_gwf_split(path: Path, detector: str, split_index: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"GWF file not found: {path}")
    if not 0 <= split_index < GWF_NUM_SPLITS:
        raise ValueError(
            f"Split index {split_index} is invalid for GWF_NUM_SPLITS={GWF_NUM_SPLITS}."
        )

    series = TimeSeries.read(str(path), channel=f"{detector}:{STRAIN_CHANNEL_SUFFIX}")
    data = np.asarray(series.value, dtype=np.float32)
    if data.size % GWF_NUM_SPLITS != 0:
        raise ValueError(
            f"GWF length {data.size} is not divisible by GWF_NUM_SPLITS={GWF_NUM_SPLITS}: {path}"
        )
    split_length = data.size // GWF_NUM_SPLITS
    start = split_index * split_length
    return data[start : start + split_length]


def preprocess_channel(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if USE_INPUT_SCALE:
        data = data * INPUT_SCALE
    if USE_PER_SAMPLE_DEMEAN:
        data = data - data.mean(dtype=np.float32)
    return data


def load_label(label_path: Path, sample_id: str, parameter_names: list[str]) -> np.ndarray:
    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")
    labels = pd.read_csv(label_path)
    for row in labels.itertuples(index=False):
        if str(row[0]) == sample_id:
            full_label = np.asarray(row[2:], dtype=np.float32)
            indices = [PARAMETER_SPECS[name]["index"] for name in parameter_names]
            return full_label[indices]
    raise KeyError(f"Label not found for sample ID '{sample_id}' in {label_path}")


def draw_corner_plot(
    posterior: np.ndarray,
    label_true: np.ndarray,
    parameter_names: list[str],
    parameter_ranges: dict,
    figure_path: Path,
    title: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q50 = np.percentile(posterior, 50, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(parameter_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(3.2 * n_params, 3.2 * n_params))
    if n_params == 1:
        axes = np.asarray([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                limits = _range_based_limits(samples, label_true[col], means[col], parameter_ranges.get(parameter_names[col]))
                x_grid, y_grid = _safe_kde_1d(samples, limits)
                if x_grid is None:
                    ax.hist(samples, bins=40, density=True, color="#4DAA57", alpha=0.35)
                else:
                    ax.plot(x_grid, y_grid, color="#4DAA57", linewidth=1.8)
                ax.axvline(label_true[col], color="red", linestyle="--", linewidth=1.0)
                ax.set_xlim(*limits)
                ax.set_yticks([])
                ax.set_title(parameter_names[col])
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
                x_limits = _range_based_limits(x, label_true[col], means[col], parameter_ranges.get(parameter_names[col]))
                y_limits = _range_based_limits(y, label_true[row], means[row], parameter_ranges.get(parameter_names[row]))
                xx, yy, zz = _safe_kde_2d(x, y, x_limits, y_limits)
                if zz is None:
                    ax.scatter(x, y, s=4, alpha=0.15, color="#4DAA57")
                else:
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
                ax.scatter(label_true[col], label_true[row], color="red", marker="x", s=60, linewidths=2)
                ax.scatter(means[col], means[row], color="black", marker="o", s=18)
                ax.set_xlim(*_expand_if_degenerate(*x_limits))
                ax.set_ylim(*_expand_if_degenerate(*y_limits))
            else:
                ax.axis("off")

            if row == n_params - 1 and col <= row:
                ax.set_xlabel(parameter_names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(parameter_names[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return means, q16, q50, q84


def main() -> None:
    if cobaya_run is None:
        raise ImportError("Cobaya is not installed. Please install 'cobaya' before running this script.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")

    output_name = f"outputs_dataset_{MODEL_DATA_VERSION}"
    checkpoint_name = "best_flow_v1_a.pt"
    model_tag = "a"
    if USE_MODEL2:
        output_name += "_model2"
        checkpoint_name = "best_flow_v1_a_model2.pt"
        model_tag = "a_model2"

    version_dir = Path(__file__).resolve().parent
    scripts_dir = version_dir.parent
    output_dir = version_dir / output_name
    checkpoint_path = output_dir / checkpoint_name
    model, test_records, run_config, split_manifest_path = load_model_and_test_records(checkpoint_path, device)
    parameter_names = list(run_config["parameter_names"])
    parameter_ranges = get_parameter_ranges(MODEL_DATA_VERSION, parameter_names)
    label_mean = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std = np.asarray(run_config["label_std"], dtype=np.float64)
    feature_mean = np.asarray(run_config["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(run_config["feature_std"], dtype=np.float32)
    label_path = scripts_dir / "training_set" / GWF_DATA_VERSION / GWF_FOLDER_NAME / "train_idx.csv"
    figure_dir = version_dir / "result_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test split manifest: {split_manifest_path}")
    print(f"GWF root: {scripts_dir / 'training_set' / GWF_DATA_VERSION / GWF_FOLDER_NAME}")
    print(f"GWF_NUM_SPLITS: {GWF_NUM_SPLITS}")

    end_sample_index = min(len(test_records), START_SAMPLE_INDEX + NUM_TEST_SAMPLES)
    for sample_index in range(START_SAMPLE_INDEX, end_sample_index):
        record = test_records[sample_index]
        h1_stem, record_split_index = parse_gwf_stem(record["h1"], "H1")
        l1_stem, l1_record_split_index = parse_gwf_stem(record["l1"], "L1")
        split_index = record_split_index if SPLIT_INDEX_OVERRIDE is None else SPLIT_INDEX_OVERRIDE
        if SPLIT_INDEX_OVERRIDE is None and l1_record_split_index != split_index:
            raise ValueError("H1 and L1 records refer to different split indices.")

        h1_path = build_gwf_path(scripts_dir, "H1", h1_stem)
        l1_path = build_gwf_path(scripts_dir, "L1", l1_stem)
        print(f"Original test H1 NPY record: {record['h1']}")
        print(f"Original test L1 NPY record: {record['l1']}")
        print(f"Test sample H1 GWF path: {h1_path}")
        print(f"Test sample L1 GWF path: {l1_path}")
        print(f"Using GWF split index: {split_index}")

        h1 = preprocess_channel(load_gwf_split(h1_path, "H1", split_index))
        l1 = preprocess_channel(load_gwf_split(l1_path, "L1", split_index))
        if h1.shape != l1.shape:
            raise ValueError(f"H1/L1 feature length mismatch: {h1.shape} versus {l1.shape}")
        features = np.stack([h1, l1], axis=0)
        features = (features - feature_mean[:, None]) / feature_std[:, None]
        data = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)

        sample_id = h1_stem.removeprefix("H1-")
        label_true = load_label(label_path, sample_id, parameter_names).astype(np.float64)
        label_true_norm = (label_true - label_mean) / label_std
        norm_parameter_names = [f"norm_{name}" for name in parameter_names]
        print(f"Parameter names: {parameter_names}")
        print(f"Label true: {label_true}")
        print(f"Label true norm: {label_true_norm}")

        def flow_target_logp(**params):
            theta_norm = np.asarray([params[name] for name in norm_parameter_names], dtype=np.float32)
            theta = torch.tensor(theta_norm[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                return float(model.log_prob(theta, data).item())

        params_info = {
            name: {
                "prior": {"min": -NORMALIZED_PRIOR_ABS_MAX, "max": NORMALIZED_PRIOR_ABS_MAX},
                "ref": float(label_true_norm[index]),
                "proposal": PROPOSAL_STD,
            }
            for index, name in enumerate(norm_parameter_names)
        }
        chain_prefix = str(figure_dir / f"cobaya_chain_{MODEL_DATA_VERSION}_{model_tag}_from_gwf_{sample_index:05d}")
        info = {
            "force": True,
            "resume": False,
            "params": params_info,
            "likelihood": {"flow_posterior": {"external": flow_target_logp, "input_params": norm_parameter_names}},
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

        _, sampler = cobaya_run(info, force=True)
        posterior_norm = _collection_to_array(sampler.products()["sample"], norm_parameter_names)
        posterior = posterior_norm * label_std + label_mean
        figure_path = figure_dir / f"test_sample_posterior_cobaya_mcmc_{MODEL_DATA_VERSION}_{model_tag}_from_gwf_{sample_index:05d}.png"
        means, q16, q50, q84 = draw_corner_plot(
            posterior,
            label_true,
            parameter_names,
            parameter_ranges,
            figure_path,
            f"Posterior Cobaya MCMC (GWF input, sample {sample_index})",
        )

        lines = [
            f"Sample index: {sample_index}",
            f"Model data version: {MODEL_DATA_VERSION}",
            f"GWF data version: {GWF_DATA_VERSION}",
            f"H1 GWF: {h1_path}",
            f"L1 GWF: {l1_path}",
            f"GWF split index: {split_index}",
            "Sampler: Cobaya MCMC over the trained flow posterior",
            "",
        ]
        for index, name in enumerate(parameter_names):
            in68 = "Y" if q16[index] <= label_true[index] <= q84[index] else "N"
            lines.append(
                f"{name:<12} true={label_true[index]:.6f} mean={means[index]:.6f} "
                f"median={q50[index]:.6f} ci16={q16[index]:.6f} ci84={q84[index]:.6f} "
                f"in68={in68}"
            )
        summary = "\n".join(lines)
        print(summary)
        if SAVE_TEXT_SUMMARY:
            summary_path = figure_dir / f"test_sample_posterior_cobaya_mcmc_{MODEL_DATA_VERSION}_{model_tag}_from_gwf_{sample_index:05d}.txt"
            summary_path.write_text(summary, encoding="utf-8")
            print(f"Saved summary: {summary_path}")
        print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
