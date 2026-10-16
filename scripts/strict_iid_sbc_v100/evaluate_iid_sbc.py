from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from config import (
    CONDITION_ON_NONEMPTY_POPULATION,
    GPU_INDEX,
    H1_DIR,
    L1_DIR,
    MODEL_CHECKPOINT_PATH,
    MODEL_DATA_VERSION,
    MODEL_RUN_CONFIG_PATH,
    NUM_POSTERIOR_SAMPLES,
    NUM_RANK_BINS,
    NUM_SBC_SAMPLES,
    OUTPUT_DPI,
    OUTPUT_ROOT,
    POSTERIOR_SEED_BASE,
    PROGRESS_SAVE_EVERY,
    RESET_PROGRESS,
    RESUME_PROGRESS,
    SBC_FIGURE_PATH,
    SBC_METRICS_PATH,
    SBC_PROGRESS_PATH,
    TIMEFLOW_DIR,
    UNIFORM_ENVELOPE_PROBABILITY,
    USE_MODEL2,
    posterior_seed,
    validate_config,
)


if str(TIMEFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(TIMEFLOW_DIR))

from dataset import GWDatasetV1A  # noqa: E402
from model import GWFlowModelV1A  # noqa: E402
from model2 import GWFlowModelV1A_Model2  # noqa: E402


def load_model_and_iid_dataset(device: torch.device):
    run_config = json.loads(MODEL_RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    records = GWDatasetV1A.build_records(
        h1_dir=H1_DIR,
        l1_dir=L1_DIR,
        sample_step=1,
        dataset_name="strict_iid_sbc",
    )
    if len(records) != NUM_SBC_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SBC_SAMPLES} IID records, found {len(records)}. "
            "Complete data generation before evaluation."
        )

    dataset_kwargs = {
        "records": records,
        "dataset_name": "strict_iid_sbc",
        "label_mean": torch.tensor(run_config["label_mean"], dtype=torch.float32),
        "label_std": torch.tensor(run_config["label_std"], dtype=torch.float32),
        "feature_mean": torch.tensor(run_config["feature_mean"], dtype=torch.float32),
        "feature_std": torch.tensor(run_config["feature_std"], dtype=torch.float32),
    }
    if "selected_parameters" in inspect.signature(GWDatasetV1A.__init__).parameters:
        dataset_kwargs["selected_parameters"] = list(run_config["parameter_names"])
    dataset = GWDatasetV1A(**dataset_kwargs)

    configured_variant = run_config.get("model_variant", "model1")
    expected_variant = "model2" if USE_MODEL2 else "model1"
    if configured_variant != expected_variant:
        raise RuntimeError(
            f"Run config model variant is {configured_variant}, expected {expected_variant}."
        )
    model_class = GWFlowModelV1A_Model2 if USE_MODEL2 else GWFlowModelV1A
    model = model_class(
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
    return model, dataset, run_config, records


def record_digest(records: list[dict]) -> str:
    payload = "\n".join(
        f"{record['h1']}|{record['l1']}" for record in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_progress(ranks: list[np.ndarray], metadata: dict[str, object]) -> None:
    temporary_path = SBC_PROGRESS_PATH.with_suffix(".npz.tmp")
    rank_array = np.asarray(ranks, dtype=np.int32)
    if not ranks:
        rank_array = np.empty((0, len(metadata["parameter_names"])), dtype=np.int32)
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ranks=rank_array,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_path.replace(SBC_PROGRESS_PATH)


def load_progress(expected_metadata: dict[str, object]) -> list[np.ndarray]:
    try:
        with np.load(SBC_PROGRESS_PATH, allow_pickle=False) as saved:
            ranks = np.asarray(saved["ranks"], dtype=np.int32)
            metadata = json.loads(str(saved["metadata_json"].item()))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load {SBC_PROGRESS_PATH}; set RESET_PROGRESS=True."
        ) from exc
    if metadata != expected_metadata:
        raise RuntimeError(
            "Saved strict-IID SBC progress is incompatible with the current run. "
            "Set RESET_PROGRESS=True to restart."
        )
    expected_shape_tail = (len(expected_metadata["parameter_names"]),)
    if (
        ranks.ndim != 2
        or ranks.shape[1:] != expected_shape_tail
        or ranks.shape[0] > int(expected_metadata["evaluation_size"])
        or np.any(ranks < 0)
        or np.any(ranks > NUM_POSTERIOR_SAMPLES)
    ):
        raise RuntimeError(f"Invalid rank cache shape or values: {ranks.shape}")
    return [row.copy() for row in ranks]


def rank_bin_edges() -> np.ndarray:
    states = NUM_POSTERIOR_SAMPLES + 1
    states_per_bin = states // NUM_RANK_BINS
    return np.arange(0, states + states_per_bin, states_per_bin, dtype=np.float64)


def uniform_envelope(sample_count: int) -> tuple[float, float, float]:
    probability = 1.0 / NUM_RANK_BINS
    expected = sample_count * probability
    std = np.sqrt(sample_count * probability * (1.0 - probability))
    tail = (1.0 - UNIFORM_ENVELOPE_PROBABILITY) / 2.0
    z_value = NormalDist().inv_cdf(1.0 - tail)
    return expected, max(0.0, expected - z_value * std), expected + z_value * std


def display_name(name: str) -> str:
    return {
        "zp": r"$z_p$",
        "alpha_z": r"$\alpha_z$",
        "beta_z": r"$\beta_z$",
        "alpha_m": r"$\alpha_m$",
        "delta_m": r"$\delta_m$",
        "mu_m": r"$\mu_m$",
        "sigma_m": r"$\sigma_m$",
        "beta_q": r"$\beta_q$",
    }.get(name, name)


def plot_rank_histograms(ranks: np.ndarray, parameter_names: list[str]) -> np.ndarray:
    edges = rank_bin_edges()
    counts = np.stack(
        [np.histogram(ranks[:, column], bins=edges)[0] for column in range(ranks.shape[1])]
    )
    expected, lower, upper = uniform_envelope(len(ranks))
    normalized_edges = edges / (NUM_POSTERIOR_SAMPLES + 1)
    centers = 0.5 * (normalized_edges[:-1] + normalized_edges[1:])
    width = 0.88 * (normalized_edges[1] - normalized_edges[0])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(11.0, 6.5), sharex=True, sharey=True)
    colors = (
        "#557BA6", "#C66D62", "#64956D", "#826EA2",
        "#B98643", "#4D9292", "#A96180", "#73796D",
    )
    y_upper = 1.10 * max(float(np.max(counts)), upper)
    for column, (axis, name) in enumerate(zip(axes.ravel(), parameter_names)):
        axis.axhspan(lower, upper, color="#B8BCC5", alpha=0.28, linewidth=0)
        axis.axhline(expected, color="#555555", linestyle=(0, (3, 2.4)), linewidth=0.9)
        axis.bar(
            centers,
            counts[column],
            width=width,
            color=colors[column],
            edgecolor="white",
            linewidth=0.25,
        )
        axis.set_title(display_name(name), pad=5)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, y_upper)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", color="#C8CAD0", linestyle="--", linewidth=0.52, alpha=0.48)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.legend(
        handles=[
            Patch(facecolor=colors[0], label="Observed SBC ranks"),
            Line2D([0], [0], color="#555555", linestyle=(0, (3, 2.4)), label="Uniform expectation"),
            Patch(facecolor="#B8BCC5", alpha=0.28, label="99% pointwise envelope"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    figure.supxlabel("Normalized SBC rank", y=0.035)
    figure.supylabel("Count per rank bin", x=0.018)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.89, wspace=0.20, hspace=0.31)
    figure.savefig(SBC_FIGURE_PATH, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return counts


def format_metrics(
    ranks: np.ndarray,
    counts: np.ndarray,
    parameter_names: list[str],
) -> str:
    expected, lower, upper = uniform_envelope(len(ranks))
    lines = [
        "Strict IID simulation-based calibration rank diagnostics",
        "=" * 58,
        f"Model data version: {MODEL_DATA_VERSION}",
        "Parameter design: independent draws from the exact uniform prior",
        "Observation design: one independently simulated H1/L1 record per theta",
        f"Conditioned on non-empty CBC population: {CONDITION_ON_NONEMPTY_POPULATION}",
        f"Evaluated IID populations: {len(ranks)}",
        f"Posterior samples per population: {NUM_POSTERIOR_SAMPLES}",
        f"Possible ranks: 0-{NUM_POSTERIOR_SAMPLES}",
        f"Rank histogram bins: {NUM_RANK_BINS}",
        f"Expected count per bin: {expected:.6f}",
        f"99% pointwise count envelope: [{lower:.6f}, {upper:.6f}]",
        f"Hyperparameter seed: see generated metadata",
        f"Posterior seed base: {POSTERIOR_SEED_BASE}",
        f"Progress file: {SBC_PROGRESS_PATH}",
        "",
        "Per-parameter uniform-rank summaries:",
        f"{'parameter':<15}{'chi2':>14}{'chi2/dof':>14}{'min_count':>14}{'max_count':>14}{'max_rel_dev':>16}",
        "-" * 87,
    ]
    degrees_of_freedom = NUM_RANK_BINS - 1
    for column, name in enumerate(parameter_names):
        values = counts[column].astype(np.float64)
        chi_square = float(np.sum((values - expected) ** 2 / expected))
        max_relative_deviation = float(np.max(np.abs(values - expected)) / expected)
        lines.append(
            f"{name:<15}{chi_square:>14.6f}{chi_square / degrees_of_freedom:>14.6f}"
            f"{int(values.min()):>14d}{int(values.max()):>14d}{max_relative_deviation:>16.6f}"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "  Flat ranks: globally calibrated marginal posterior.",
            "  U-shaped ranks: posterior is typically too narrow (overconfident).",
            "  Centre-heavy ranks: posterior is typically too wide (underconfident).",
            "  Monotonic slope: systematic posterior-location bias.",
            "  Marginal rank uniformity is not proof of complete joint calibration.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    validate_config()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for required_path in (MODEL_RUN_CONFIG_PATH, MODEL_CHECKPOINT_PATH, H1_DIR, L1_DIR):
        if not required_path.exists():
            raise FileNotFoundError(f"Required input not found: {required_path}")

    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    model, dataset, run_config, records = load_model_and_iid_dataset(device)
    parameter_names = list(run_config["parameter_names"])
    checkpoint_stat = MODEL_CHECKPOINT_PATH.stat()
    metadata = {
        "format_version": 1,
        "strict_iid": True,
        "evaluation_size": len(dataset),
        "record_digest": record_digest(records),
        "parameter_names": parameter_names,
        "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
        "num_rank_bins": NUM_RANK_BINS,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "condition_on_nonempty_population": CONDITION_ON_NONEMPTY_POPULATION,
        "posterior_seed_base": POSTERIOR_SEED_BASE,
    }

    ranks: list[np.ndarray] = []
    if RESUME_PROGRESS and not RESET_PROGRESS and SBC_PROGRESS_PATH.exists():
        ranks = load_progress(metadata)
        print(f"Resuming strict-IID SBC: {len(ranks)}/{len(dataset)}")
    elif RESET_PROGRESS and SBC_PROGRESS_PATH.exists():
        print(f"RESET_PROGRESS=True: ignoring {SBC_PROGRESS_PATH}")

    progress = tqdm(
        range(len(ranks), len(dataset)),
        initial=len(ranks),
        total=len(dataset),
        desc="Strict IID SBC",
        unit="population",
    )
    for index in progress:
        data, truth_normalized = dataset[index]
        data = data.unsqueeze(0).to(device)
        truth_normalized = truth_normalized.cpu().numpy()
        seed = posterior_seed(index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            posterior = model.sample(data, num_samples=NUM_POSTERIOR_SAMPLES)
        posterior = posterior.squeeze(0).detach().cpu().numpy()
        rank = np.sum(posterior < truth_normalized[None, :], axis=0).astype(np.int32)
        ranks.append(rank)
        completed = index + 1
        progress.set_postfix_str(
            "rank=" + ",".join(str(int(value)) for value in rank), refresh=True
        )
        if completed % PROGRESS_SAVE_EVERY == 0 or completed == len(dataset):
            save_progress(ranks, metadata)

    rank_array = np.asarray(ranks, dtype=np.int32)
    counts = plot_rank_histograms(rank_array, parameter_names)
    metrics = format_metrics(rank_array, counts, parameter_names)
    SBC_METRICS_PATH.write_text(metrics, encoding="utf-8")
    print(metrics)
    print(f"Saved 800-dpi strict-IID SBC PDF: {SBC_FIGURE_PATH}")
    print(f"Saved strict-IID SBC metrics: {SBC_METRICS_PATH}")
    print(f"Saved strict-IID SBC progress: {SBC_PROGRESS_PATH}")


if __name__ == "__main__":
    main()
