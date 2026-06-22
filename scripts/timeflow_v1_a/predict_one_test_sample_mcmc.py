from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from dataset import PARAMETER_SPECS
from eval_utils import load_model_and_dataset

try:
    from dataset import get_parameter_ranges as _dataset_get_parameter_ranges
except ImportError:
    _dataset_get_parameter_ranges = None


SAMPLE_INDEX = 0
NUM_MCMC_SAMPLES = 2000
BURN_IN = 1000
PROPOSAL_STD = 0.15
NUM_CHAINS = 4
DATA_VERSION = "v21"
GPU_INDEX = 1
SEED = 42
USE_MODEL2 = True


DATA_VERSION_RANGE_OVERRIDES = {
    "v21": {
        "alpha_m": (-3.0, 7.0),
        "delta_m": (0.0, 10.0),
        "mu_m": (25.0, 100.0),
        "sigma_m": (0.01, 5.0),
        "zp": (1.9, 1.9),
        "alpha_z": (2.7, 2.7),
        "beta_z": (2.9, 2.9),
        "m_max": (100.0, 100.0),
        "m_min": (2.0, 2.0),
        "lambda_peak": (0.04, 0.04),
        "beta_q": (1.08, 1.08),
    }
}


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


def log_prob_theta(model, theta: torch.Tensor, data: torch.Tensor) -> float:
    with torch.no_grad():
        return float(model.log_prob(theta, data).item())


def metropolis_hastings(
    model,
    data: torch.Tensor,
    init_theta: torch.Tensor,
    num_samples: int,
    burn_in: int,
    proposal_std: float,
) -> tuple[np.ndarray, float]:
    current = init_theta.clone()
    current_log_prob = log_prob_theta(model, current, data)
    samples = []
    accepted = 0
    total_steps = burn_in + num_samples

    for step in range(total_steps):
        proposal = current + torch.randn_like(current) * proposal_std
        proposal_log_prob = log_prob_theta(model, proposal, data)
        log_alpha = proposal_log_prob - current_log_prob

        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1

        if step >= burn_in:
            samples.append(current.squeeze(0).detach().cpu().numpy().copy())

    acceptance_rate = accepted / total_steps
    return np.asarray(samples, dtype=np.float32), acceptance_rate


def draw_flow_chain_initial_states(
    model,
    data: torch.Tensor,
    num_chains: int,
    device: torch.device,
) -> list[torch.Tensor]:
    with torch.no_grad():
        init_thetas = model.sample(data, num_samples=num_chains).squeeze(0)
    return [init_thetas[i : i + 1].to(device) for i in range(num_chains)]


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    output_name = f"outputs_dataset_{DATA_VERSION}"
    checkpoint_name = "best_flow_v1_a.pt"
    model_tag = "a"
    if USE_MODEL2:
        output_name += "_model2"
        checkpoint_name = "best_flow_v1_a_model2.pt"
        model_tag = "a_model2"
    output_dir = Path(__file__).resolve().parent / output_name
    checkpoint_path = output_dir / checkpoint_name
    model, dataset, run_config = load_model_and_dataset("test", checkpoint_path, device)
    parameter_names = list(run_config["parameter_names"])

    data, label_norm = dataset[SAMPLE_INDEX]
    data = data.unsqueeze(0).to(device)
    label_norm = label_norm.unsqueeze(0).to(device)

    init_thetas = draw_flow_chain_initial_states(model=model, data=data, num_chains=NUM_CHAINS, device=device)

    chain_samples = []
    acceptance_rates = []
    for chain_idx, init_theta in enumerate(init_thetas):
        torch.manual_seed(SEED + chain_idx)
        np.random.seed(SEED + chain_idx)
        samples_norm, acceptance_rate = metropolis_hastings(
            model=model,
            data=data,
            init_theta=init_theta,
            num_samples=NUM_MCMC_SAMPLES,
            burn_in=BURN_IN,
            proposal_std=PROPOSAL_STD,
        )
        chain_samples.append(samples_norm)
        acceptance_rates.append(acceptance_rate)

    posterior_norm = chain_samples[0]
    acceptance_rate = float(np.mean(acceptance_rates))

    label_true_norm = label_norm.squeeze(0).cpu().numpy()
    label_mean = np.asarray(run_config["label_mean"], dtype=np.float64)
    label_std = np.asarray(run_config["label_std"], dtype=np.float64)
    posterior = posterior_norm * label_std + label_mean
    label_true = label_true_norm * label_std + label_mean

    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)
    n_params = len(parameter_names)

    figure_dir = Path(__file__).resolve().parent / "result_fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"test_sample_posterior_mcmc_{DATA_VERSION}_{model_tag}_sample{SAMPLE_INDEX:05d}.png"
    metrics_path = figure_dir / f"test_sample_posterior_mcmc_{DATA_VERSION}_{model_tag}_sample{SAMPLE_INDEX:05d}.txt"

    fig, axes = plt.subplots(n_params, n_params, figsize=(3.2 * n_params, 3.2 * n_params))
    if n_params == 1:
        axes = np.asarray([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                ax.hist(samples, bins=40, density=True, color="tab:green", alpha=0.7)
                ax.axvline(label_true[col], color="red", linestyle="-", linewidth=1.5)
                ax.axvline(means[col], color="black", linestyle="--", linewidth=1.2)
                ax.axvspan(q16[col], q84[col], color="tab:green", alpha=0.15)
                ax.set_title(parameter_names[col])
                x_limits = _adaptive_limits(samples, label_true[col], means[col])
                x_grid, y_grid = _safe_kde_1d(samples, x_limits)
                ax.cla()
                if x_grid is not None:
                    ax.plot(x_grid, y_grid, color="#4DAA57", linewidth=1.8)
                else:
                    ax.hist(samples, bins=40, density=True, color="#4DAA57", alpha=0.35)
                ax.axvline(label_true[col], color="#BDBDBD", linestyle="--", linewidth=1.0)
                ax.set_title(parameter_names[col])
                ax.set_xlim(*x_limits)
                ax.set_yticks([])
            elif row > col:
                x = posterior[:, col]
                y = posterior[:, row]
                x_limits = _adaptive_limits(x, label_true[col], means[col])
                y_limits = _adaptive_limits(y, label_true[row], means[row])
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
                ax.set_xlabel(parameter_names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(parameter_names[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])

    fig.suptitle(f"Posterior MCMC corner plot (test sample, {DATA_VERSION})", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    lines = [
        f"Sample index: {SAMPLE_INDEX}",
        f"Posterior samples per chain: {NUM_MCMC_SAMPLES}",
        f"Number of chains: {NUM_CHAINS}",
        "Plotted chain index: 0",
        f"Total posterior samples (plotted chain): {posterior_norm.shape[0]}",
        "Proposal type: Gaussian random walk",
        f"Burn-in: {BURN_IN}",
        f"Proposal std (normalized space): {PROPOSAL_STD}",
        f"Mean acceptance rate: {acceptance_rate:.6f}",
        f"Per-chain acceptance rates: {[round(rate, 6) for rate in acceptance_rates]}",
        "",
    ]
    for i, name in enumerate(parameter_names):
        in68 = "Y" if q16[i] <= label_true[i] <= q84[i] else "N"
        lines.append(
            f"{name:<12} true={label_true[i]:.6f} mean={means[i]:.6f} "
            f"err={means[i]-label_true[i]:.6f} ci16={q16[i]:.6f} ci84={q84[i]:.6f} in68={in68}"
        )
    text = "\n".join(lines)

    print(text)
    metrics_path.write_text(text, encoding="utf-8")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
