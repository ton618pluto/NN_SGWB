from __future__ import annotations

import hashlib
import json
import re
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

from eval_utils import load_model_and_dataset


# ---------------------------- User controls ----------------------------
DATA_VERSION = "v100_random_two_noise_100"
GPU_INDEX = 1
USE_MODEL2 = False

# One TimeFlow record is selected for each population before this limit is
# applied. Set to None to evaluate every distinct test-set population.
# EVALUATION_SAMPLE_LIMIT: int | None = 17_000
EVALUATION_SAMPLE_LIMIT: int | None = 1_000

# M posterior samples produce M + 1 possible ranks. Here 999 + 1 = 1000,
# which divides exactly into 20 equal-probability histogram bins.
NUM_POSTERIOR_SAMPLES = 999
NUM_RANK_BINS = 20

SEED = 42
RESUME_PROGRESS = True
RESET_PROGRESS = False
PROGRESS_SAVE_EVERY = 1

OUTPUT_DPI = 800
FIGURE_SIZE_INCHES = (11.0, 6.5)

# Pointwise envelope for the expected count in each rank bin under uniformity.
UNIFORM_ENVELOPE_PROBABILITY = 0.99

POP_PATTERN = re.compile(r"pop(\d+)")


def _select_one_record_per_pop(
    records: list[dict],
    sample_limit: int | None,
    seed: int,
) -> tuple[list[int], list[str], int]:
    """Select one deterministic noise realization from each randomly chosen pop."""
    grouped: dict[str, list[tuple[str, int]]] = {}
    for record_index, record in enumerate(records):
        h1_path = str(record.get("h1", ""))
        match = POP_PATTERN.search(Path(h1_path).name)
        if match is None:
            raise ValueError(f"Unable to parse pop id from test record: {h1_path}")
        pop_id = f"pop{int(match.group(1)):05d}"
        grouped.setdefault(pop_id, []).append((Path(h1_path).name, record_index))

    # Sorting by file name selects sample00000 when both random-noise versions
    # are present, while still working if that record is missing.
    one_record_per_pop = [
        (pop_id, min(grouped[pop_id], key=lambda item: item[0])[1])
        for pop_id in sorted(grouped)
    ]
    available_pop_count = len(one_record_per_pop)
    if available_pop_count == 0:
        raise ValueError("No population IDs were found in the test records.")

    if sample_limit is None:
        selected = one_record_per_pop
    else:
        if sample_limit < 1:
            raise ValueError(
                f"EVALUATION_SAMPLE_LIMIT must be positive or None, got {sample_limit}."
            )
        selection_size = min(sample_limit, available_pop_count)
        generator = np.random.default_rng(seed)
        selected_positions = generator.choice(
            available_pop_count,
            size=selection_size,
            replace=False,
        )
        selected = [one_record_per_pop[int(position)] for position in selected_positions]

    selected_pop_ids = [pop_id for pop_id, _ in selected]
    selected_indices = [record_index for _, record_index in selected]
    return selected_indices, selected_pop_ids, available_pop_count


def _selection_digest(
    records: list[dict],
    selected_indices: list[int],
) -> str:
    selected_paths = [str(records[index].get("h1", "")) for index in selected_indices]
    payload = "\n".join(selected_paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_progress(
    progress_path: Path,
    ranks: list[np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Atomically save all completed per-sample rank vectors."""
    temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    if ranks:
        rank_array = np.asarray(ranks, dtype=np.int32)
    else:
        rank_array = np.empty(
            (0, len(metadata["parameter_names"])), dtype=np.int32
        )
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ranks=rank_array,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary_path.replace(progress_path)


def _load_progress(
    progress_path: Path,
    expected_metadata: dict[str, object],
) -> list[np.ndarray]:
    try:
        with np.load(progress_path, allow_pickle=False) as saved:
            ranks = np.asarray(saved["ranks"], dtype=np.int32)
            saved_metadata = json.loads(str(saved["metadata_json"].item()))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read SBC progress file {progress_path}. "
            "Set RESET_PROGRESS=True to start over."
        ) from exc

    if saved_metadata != expected_metadata:
        raise RuntimeError(
            f"SBC progress file is incompatible with the current settings: "
            f"{progress_path}\n"
            f"Saved metadata: {saved_metadata}\n"
            f"Current metadata: {expected_metadata}\n"
            "Set RESET_PROGRESS=True to start over."
        )

    expected_columns = len(expected_metadata["parameter_names"])
    evaluation_size = int(expected_metadata["evaluation_size"])
    maximum_rank = int(expected_metadata["num_posterior_samples"])
    if (
        ranks.ndim != 2
        or ranks.shape[1] != expected_columns
        or ranks.shape[0] > evaluation_size
        or np.any(ranks < 0)
        or np.any(ranks > maximum_rank)
    ):
        raise RuntimeError(
            f"Invalid SBC ranks in {progress_path}: shape={ranks.shape}. "
            "Set RESET_PROGRESS=True to start over."
        )
    return [row.copy() for row in ranks]


def _rank_bin_edges() -> np.ndarray:
    rank_state_count = NUM_POSTERIOR_SAMPLES + 1
    if rank_state_count % NUM_RANK_BINS != 0:
        raise ValueError(
            "NUM_POSTERIOR_SAMPLES + 1 must be divisible by NUM_RANK_BINS. "
            f"Got {rank_state_count} rank states and {NUM_RANK_BINS} bins."
        )
    states_per_bin = rank_state_count // NUM_RANK_BINS
    return np.arange(
        0,
        rank_state_count + states_per_bin,
        states_per_bin,
        dtype=np.float64,
    )


def _uniform_count_envelope(sample_count: int) -> tuple[float, float, float]:
    probability = 1.0 / NUM_RANK_BINS
    expected = sample_count * probability
    standard_deviation = np.sqrt(sample_count * probability * (1.0 - probability))
    tail_probability = (1.0 - UNIFORM_ENVELOPE_PROBABILITY) / 2.0
    z_value = NormalDist().inv_cdf(1.0 - tail_probability)
    lower = max(0.0, expected - z_value * standard_deviation)
    upper = expected + z_value * standard_deviation
    return expected, lower, upper


def _parameter_display_name(name: str) -> str:
    labels = {
        "zp": r"$z_p$",
        "alpha_z": r"$\alpha_z$",
        "beta_z": r"$\beta_z$",
        "alpha_m": r"$\alpha_m$",
        "delta_m": r"$\delta_m$",
        "mu_m": r"$\mu_m$",
        "sigma_m": r"$\sigma_m$",
        "beta_q": r"$\beta_q$",
    }
    return labels.get(name, name.replace("_", r"\_"))


def _compute_histogram_counts(ranks: np.ndarray) -> np.ndarray:
    edges = _rank_bin_edges()
    return np.stack(
        [np.histogram(ranks[:, index], bins=edges)[0] for index in range(ranks.shape[1])],
        axis=0,
    )


def _plot_rank_histograms(
    ranks: np.ndarray,
    parameter_names: list[str],
    figure_path: Path,
) -> np.ndarray:
    counts = _compute_histogram_counts(ranks)
    expected, envelope_lower, envelope_upper = _uniform_count_envelope(len(ranks))

    rank_state_count = NUM_POSTERIOR_SAMPLES + 1
    integer_edges = _rank_bin_edges()
    normalized_edges = integer_edges / rank_state_count
    centers = 0.5 * (normalized_edges[:-1] + normalized_edges[1:])
    bar_width = 0.88 * (normalized_edges[1] - normalized_edges[0])

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
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=FIGURE_SIZE_INCHES,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    colors = (
        "#557BA6",
        "#C66D62",
        "#64956D",
        "#826EA2",
        "#B98643",
        "#4D9292",
        "#A96180",
        "#73796D",
    )

    maximum_count = max(float(np.max(counts)), envelope_upper)
    y_upper = max(1.0, 1.10 * maximum_count)
    for parameter_index, (axis, name) in enumerate(zip(axes_flat, parameter_names)):
        axis.axhspan(
            envelope_lower,
            envelope_upper,
            color="#B8BCC5",
            alpha=0.28,
            linewidth=0.0,
            zorder=0,
        )
        axis.axhline(
            expected,
            color="#555555",
            linestyle=(0, (3.0, 2.4)),
            linewidth=0.9,
            zorder=2,
        )
        axis.bar(
            centers,
            counts[parameter_index],
            width=bar_width,
            color=colors[parameter_index % len(colors)],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.91,
            zorder=1,
        )
        axis.set_title(_parameter_display_name(name), pad=5.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, y_upper)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(
            axis="y",
            color="#C8CAD0",
            linestyle="--",
            linewidth=0.52,
            alpha=0.48,
            zorder=0,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(colors="#3F3F3F", length=3.5, width=0.75)

    # The current TimeFlow v1_a setup uses eight inferred parameters. Keep a
    # graceful fallback if a checkpoint contains fewer parameters.
    for axis in axes_flat[len(parameter_names) :]:
        axis.set_visible(False)

    figure.legend(
        handles=[
            Patch(facecolor="#557BA6", alpha=0.91, label="Observed SBC ranks"),
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=(0, (3.0, 2.4)),
                linewidth=0.9,
                label="Uniform expectation",
            ),
            Patch(
                facecolor="#B8BCC5",
                alpha=0.28,
                label=f"{100.0 * UNIFORM_ENVELOPE_PROBABILITY:.0f}% pointwise envelope",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    figure.supxlabel("Normalized SBC rank", y=0.035, fontsize=11.5)
    figure.supylabel("Count per rank bin", x=0.018, fontsize=11.5)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.89,
        wspace=0.20,
        hspace=0.31,
    )
    figure.savefig(
        figure_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    return counts


def _format_metrics(
    ranks: np.ndarray,
    counts: np.ndarray,
    parameter_names: list[str],
    progress_path: Path,
    available_pop_count: int,
) -> str:
    expected, envelope_lower, envelope_upper = _uniform_count_envelope(len(ranks))
    lines = [
        "Global simulation-based calibration rank diagnostics",
        "=" * 54,
        f"Data version: {DATA_VERSION}",
        "Record selection: one input record per distinct population",
        f"Available distinct test populations: {available_pop_count}",
        f"Evaluated distinct populations: {len(ranks)}",
        f"Posterior samples per test sample: {NUM_POSTERIOR_SAMPLES}",
        f"Possible ranks: 0-{NUM_POSTERIOR_SAMPLES}",
        f"Rank histogram bins: {NUM_RANK_BINS}",
        f"Expected count per bin: {expected:.6f}",
        f"{100.0 * UNIFORM_ENVELOPE_PROBABILITY:.0f}% pointwise count envelope: "
        f"[{envelope_lower:.6f}, {envelope_upper:.6f}]",
        f"Progress file: {progress_path}",
        "",
        "Per-parameter uniform-rank summaries:",
        f"{'parameter':<15}{'chi2':>14}{'chi2/dof':>14}{'min_count':>14}"
        f"{'max_count':>14}{'max_rel_dev':>16}",
        "-" * 87,
    ]
    degrees_of_freedom = NUM_RANK_BINS - 1
    for parameter_index, name in enumerate(parameter_names):
        parameter_counts = counts[parameter_index].astype(np.float64)
        chi_square = float(np.sum((parameter_counts - expected) ** 2 / expected))
        maximum_relative_deviation = float(
            np.max(np.abs(parameter_counts - expected)) / expected
        )
        lines.append(
            f"{name:<15}{chi_square:>14.6f}"
            f"{(chi_square / degrees_of_freedom):>14.6f}"
            f"{int(np.min(parameter_counts)):>14d}"
            f"{int(np.max(parameter_counts)):>14d}"
            f"{maximum_relative_deviation:>16.6f}"
        )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "  Flat ranks: globally calibrated marginal posterior.",
            "  U-shaped ranks: posterior is typically too narrow (overconfident).",
            "  Centre-heavy ranks: posterior is typically too wide (underconfident).",
            "  Monotonic slope: systematic posterior-location bias.",
            "  This is a marginal rank check for each parameter, not a proof of "
            "complete joint calibration.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if PROGRESS_SAVE_EVERY < 1:
        raise ValueError(
            f"PROGRESS_SAVE_EVERY must be at least 1, got {PROGRESS_SAVE_EVERY}."
        )
    _rank_bin_edges()

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
    model.eval()

    parameter_names = list(run_config["parameter_names"])
    dataset_size = len(dataset)
    evaluation_indices, selected_pop_ids, available_pop_count = (
        _select_one_record_per_pop(
            dataset.records,
            sample_limit=EVALUATION_SAMPLE_LIMIT,
            seed=SEED,
        )
    )
    evaluation_size = len(evaluation_indices)
    if evaluation_size < 1:
        raise ValueError("The SBC evaluation set is empty.")
    if len(parameter_names) > 8:
        raise ValueError(
            "The 2x4 paper figure supports at most eight inferred parameters, "
            f"but the checkpoint contains {len(parameter_names)}."
        )

    figure_dir = output_dir / "test_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_set_sbc_rank_hist_{file_tag}.pdf"
    metrics_path = figure_dir / f"test_set_sbc_metrics_{file_tag}.txt"
    progress_path = (
        figure_dir
        / f"test_set_sbc_progress_{file_tag}_one_per_pop_n{evaluation_size}.npz"
    )

    checkpoint_stat = checkpoint_path.stat()
    progress_metadata: dict[str, object] = {
        "data_version": DATA_VERSION,
        "use_model2": USE_MODEL2,
        "checkpoint_name": checkpoint_name,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "dataset_size": dataset_size,
        "one_record_per_pop": True,
        "available_pop_count": available_pop_count,
        "evaluation_size": evaluation_size,
        "selection_seed": SEED,
        "selection_digest": _selection_digest(dataset.records, evaluation_indices),
        "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
        "num_rank_bins": NUM_RANK_BINS,
        "parameter_names": parameter_names,
    }

    ranks: list[np.ndarray] = []
    if RESUME_PROGRESS and not RESET_PROGRESS and progress_path.exists():
        ranks = _load_progress(progress_path, progress_metadata)
        print(
            f"Resuming SBC from {progress_path}: "
            f"{len(ranks)}/{evaluation_size} completed samples"
        )
    elif RESET_PROGRESS and progress_path.exists():
        print(f"RESET_PROGRESS=True: ignoring existing SBC cache {progress_path}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Data version: {DATA_VERSION}")
    print(f"Parameter names: {parameter_names}")
    print(f"Test records: {dataset_size}")
    print(f"Distinct test populations: {available_pop_count}")
    print(f"Evaluated populations (one record each): {evaluation_size}")
    print(f"Posterior samples per test sample: {NUM_POSTERIOR_SAMPLES}")
    print(f"Rank bins: {NUM_RANK_BINS}")
    print(f"Progress file: {progress_path}")

    progress = tqdm(
        range(len(ranks), evaluation_size),
        initial=len(ranks),
        total=evaluation_size,
        desc="Global SBC",
        unit="sample",
    )
    for evaluation_position in progress:
        dataset_index = evaluation_indices[evaluation_position]
        pop_id = selected_pop_ids[evaluation_position]
        data, label_norm = dataset[dataset_index]
        data = data.unsqueeze(0).to(device)
        label_norm = label_norm.to(device)

        with torch.inference_mode():
            posterior_norm = model.sample(
                data,
                num_samples=NUM_POSTERIOR_SAMPLES,
            )

        posterior_norm = posterior_norm.squeeze(0)
        if posterior_norm.ndim != 2:
            raise RuntimeError(
                f"Unexpected posterior shape at dataset index {dataset_index}: "
                f"{tuple(posterior_norm.shape)}"
            )
        if posterior_norm.shape[1] != len(parameter_names):
            raise RuntimeError(
                f"Posterior parameter dimension {posterior_norm.shape[1]} does not "
                f"match checkpoint parameter_names ({len(parameter_names)})."
            )

        rank_vector = torch.sum(
            posterior_norm < label_norm.unsqueeze(0),
            dim=0,
        ).to(dtype=torch.int32)
        rank_numpy = rank_vector.cpu().numpy()
        ranks.append(rank_numpy)

        if len(ranks) % PROGRESS_SAVE_EVERY == 0 or len(ranks) == evaluation_size:
            _save_progress(progress_path, ranks, progress_metadata)
        progress.set_postfix_str(
            f"{pop_id} rank_min={int(rank_numpy.min())} "
            f"rank_max={int(rank_numpy.max())}",
            refresh=False,
        )

    if not ranks:
        raise RuntimeError("No SBC ranks were evaluated.")
    rank_array = np.asarray(ranks, dtype=np.int32)
    counts = _plot_rank_histograms(rank_array, parameter_names, figure_path)
    metrics_text = _format_metrics(
        rank_array,
        counts,
        parameter_names,
        progress_path,
        available_pop_count,
    )
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print(metrics_text)
    print(f"Saved 800-dpi SBC PDF: {figure_path}")
    print(f"Saved SBC metrics: {metrics_path}")
    print(f"Saved completed SBC progress: {progress_path}")


if __name__ == "__main__":
    main()
