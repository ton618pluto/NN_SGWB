from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gwpy.timeseries import TimeSeries
from scipy.optimize import minimize

from crosscorr_8param_common import (
    CrossCorrEmulator,
    PARAMETER_NAMES,
    PARAMETER_RANGES,
    coarse_cross_spectrum,
)
from predict_one_test_sample_cobaya_mcmc import (
    _credible_levels_from_density,
    _expand_if_degenerate,
    _range_based_limits,
    _safe_kde_1d,
    _safe_kde_2d,
)

try:
    from cobaya.run import run as cobaya_run
except ImportError:
    cobaya_run = None


BASE_VERSION = "v100"
NOISE_SCALE = 100
DATA_VERSION = f"{BASE_VERSION}_random_two_noise_{NOISE_SCALE}"
TEST_VERSION = f"fixed_one_test_{DATA_VERSION}"
TEST_SAMPLE_INDEX = 0
POP_INDEX = 0
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
SEED = 42
MAX_SAMPLES = 20_000
RMINUS1_STOP = 0.03
RMINUS1_CL_STOP = 0.2
SAVE_COBAYA_CHAIN = False
MIN_EMULATOR_TEST_R2 = 0.0

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(__file__).resolve().parent
EMULATOR_DIR = VERSION_DIR / f"outputs_crosscorr_emulator_{DATA_VERSION}"
EMULATOR_PATH = EMULATOR_DIR / "best_crosscorr_emulator_8params.pt"
TEST_ROOT = SCRIPTS_DIR / "training_set_superimposed_test" / TEST_VERSION
LABEL_ROOT = SCRIPTS_DIR / "training_set_test" / TEST_VERSION / "training_set0"
RESULT_DIR = VERSION_DIR / "result_fig"


def test_paths(detector: str) -> Path:
    return TEST_ROOT / detector / (
        f"{detector}-SUPERIMPOSED-pop{POP_INDEX:05d}_sample{TEST_SAMPLE_INDEX:05d}.gwf"
    )


def load_emulator(device: torch.device) -> tuple[CrossCorrEmulator, dict]:
    if not EMULATOR_PATH.exists():
        raise FileNotFoundError(
            f"Emulator checkpoint not found: {EMULATOR_PATH}\n"
            "Run train_crosscorr_emulator_8params.py first."
        )
    checkpoint = torch.load(EMULATOR_PATH, map_location=device, weights_only=False)
    if list(checkpoint["parameter_names"]) != PARAMETER_NAMES:
        raise ValueError(f"Checkpoint parameters do not match: {checkpoint['parameter_names']}")
    model = CrossCorrEmulator(
        input_dim=len(PARAMETER_NAMES),
        output_dim=int(checkpoint["num_coarse_bins"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_observed_spectrum() -> tuple[np.ndarray, np.ndarray, Path, Path]:
    h1_path = test_paths("H1")
    l1_path = test_paths("L1")
    if not h1_path.exists() or not l1_path.exists():
        raise FileNotFoundError(f"Missing test GWF pair: {h1_path}, {l1_path}")
    h1 = TimeSeries.read(str(h1_path), channel=f"H1:{STRAIN_CHANNEL_SUFFIX}")
    l1 = TimeSeries.read(str(l1_path), channel=f"L1:{STRAIN_CHANNEL_SUFFIX}")
    if len(h1) != len(l1) or h1.sample_rate.value != l1.sample_rate.value:
        raise ValueError("Test H1/L1 length or sample-rate mismatch")
    freqs, spectrum = coarse_cross_spectrum(
        np.asarray(h1.value, dtype=np.float64),
        np.asarray(l1.value, dtype=np.float64),
        float(h1.sample_rate.value),
    )
    return freqs, spectrum, h1_path, l1_path


def load_true_parameters_after_inference() -> np.ndarray | None:
    csv_path = LABEL_ROOT / "train_idx.csv"
    if not csv_path.exists():
        print(f"Truth CSV not found; plots will omit truth markers: {csv_path}")
        return None
    sample_id = f"pop{POP_INDEX:05d}_sample{TEST_SAMPLE_INDEX:05d}"
    frame = pd.read_csv(csv_path)
    if "sample_id" not in frame.columns:
        print(f"Truth CSV has no sample_id column: {csv_path}")
        return None
    selected = frame.loc[frame["sample_id"].astype(str) == sample_id]
    if selected.empty:
        print(f"Truth row not found for {sample_id}: {csv_path}")
        return None
    row = selected.iloc[0]
    missing = [name for name in PARAMETER_NAMES if name not in frame.columns]
    if missing:
        print(f"Truth CSV is missing columns {missing}; plots will omit truth markers")
        return None
    return np.asarray([float(row[name]) for name in PARAMETER_NAMES], dtype=np.float64)


def collection_to_array(collection) -> np.ndarray:
    data = collection.data
    return np.column_stack([np.asarray(data[name], dtype=np.float64) for name in PARAMETER_NAMES])


def draw_corner(posterior: np.ndarray, truth: np.ndarray | None, path: Path) -> None:
    count = len(PARAMETER_NAMES)
    if posterior.ndim != 2 or posterior.shape[1] != count:
        raise ValueError(
            f"Posterior shape {posterior.shape} does not match {count} parameters: {PARAMETER_NAMES}"
        )
    means = posterior.mean(axis=0)
    q16 = np.percentile(posterior, 16, axis=0)
    q50 = np.percentile(posterior, 50, axis=0)
    q84 = np.percentile(posterior, 84, axis=0)

    fig, axes = plt.subplots(count, count, figsize=(3.2 * count, 3.2 * count))
    for row in range(count):
        for col in range(count):
            ax = axes[row, col]
            if row == col:
                samples = posterior[:, col]
                reference = means[col] if truth is None else truth[col]
                limits = _range_based_limits(
                    samples,
                    reference,
                    means[col],
                    PARAMETER_RANGES[PARAMETER_NAMES[col]],
                )
                x_grid, y_grid = _safe_kde_1d(samples, limits)
                if x_grid is None:
                    ax.hist(samples, bins=40, density=True, color="#4DAA57", alpha=0.35)
                else:
                    ax.plot(x_grid, y_grid, color="#4DAA57", linewidth=1.8)
                if truth is not None:
                    ax.axvline(truth[col], color="red", linestyle="--", linewidth=1.0)
                ax.set_title(PARAMETER_NAMES[col])
                ax.set_xlim(*limits)
                ax.set_yticks([])
                summary_text = (
                    (f"true={truth[col]:.3f}\n" if truth is not None else "")
                    + f"mean={means[col]:.3f}\n"
                    + f"{q50[col]:.3f}^(+{q84[col]-q50[col]:.3f})_(-{q50[col]-q16[col]:.3f})\n"
                    + "68% CI"
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
                x_reference = means[col] if truth is None else truth[col]
                y_reference = means[row] if truth is None else truth[row]
                x_limits = _range_based_limits(
                    x,
                    x_reference,
                    means[col],
                    PARAMETER_RANGES[PARAMETER_NAMES[col]],
                )
                y_limits = _range_based_limits(
                    y,
                    y_reference,
                    means[row],
                    PARAMETER_RANGES[PARAMETER_NAMES[row]],
                )
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
                if truth is not None:
                    ax.scatter(truth[col], truth[row], color="red", marker="x", s=60, linewidths=2)
                ax.scatter(means[col], means[row], color="black", marker="o", s=18)
                ax.set_xlim(*_expand_if_degenerate(*x_limits))
                ax.set_ylim(*_expand_if_degenerate(*y_limits))
            else:
                ax.axis("off")
                continue

            if row == count - 1 and col <= row:
                ax.set_xlabel(PARAMETER_NAMES[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(PARAMETER_NAMES[row])
            elif col == 0 and row == 0:
                ax.set_ylabel("density")
            elif col != 0 and row >= col:
                ax.set_yticklabels([])
    fig.suptitle("Posterior Cobaya MCMC corner plot (8-parameter cross-correlation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if cobaya_run is None:
        raise ImportError("Cobaya is not installed in the current Python environment")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model, checkpoint = load_emulator(device)
    test_r2 = float(checkpoint.get("test_r2", float("nan")))
    if not np.isfinite(test_r2) or test_r2 <= MIN_EMULATOR_TEST_R2:
        raise RuntimeError(
            f"Emulator test R2={test_r2} is not above {MIN_EMULATOR_TEST_R2}; "
            "do not run MCMC with an unvalidated forward model."
        )
    freqs, observed, h1_path, l1_path = load_observed_spectrum()
    checkpoint_freqs = np.asarray(checkpoint["freqs"], dtype=np.float64)
    if not np.allclose(freqs, checkpoint_freqs, rtol=0.0, atol=1e-12):
        raise ValueError("Test and emulator frequency grids differ")

    parameter_mean = np.asarray(checkpoint["parameter_mean"], dtype=np.float64)
    parameter_std = np.asarray(checkpoint["parameter_std"], dtype=np.float64)
    spectrum_mean = np.asarray(checkpoint["spectrum_mean"], dtype=np.float64)
    spectrum_std = np.asarray(checkpoint["spectrum_std"], dtype=np.float64)
    observed_normalized = (observed - spectrum_mean) / spectrum_std
    covariance = np.asarray(checkpoint["residual_covariance"], dtype=np.float64)
    covariance_inverse = np.linalg.inv(covariance)

    def predict_normalized(parameters: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(parameters, dtype=np.float64) - parameter_mean) / parameter_std
        tensor = torch.from_numpy(normalized.astype(np.float32))[None, :]
        with torch.no_grad():
            return model(tensor).cpu().numpy()[0].astype(np.float64)

    def negative_loglike(parameters: np.ndarray) -> float:
        residual = observed_normalized - predict_normalized(parameters)
        return float(0.5 * residual @ covariance_inverse @ residual)

    bounds = [PARAMETER_RANGES[name] for name in PARAMETER_NAMES]
    rng = np.random.default_rng(SEED)
    starts = [np.asarray([(lo + hi) * 0.5 for lo, hi in bounds], dtype=np.float64)]
    for _ in range(7):
        starts.append(np.asarray([rng.uniform(lo, hi) for lo, hi in bounds], dtype=np.float64))
    optimizations = [minimize(negative_loglike, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    finite = [result for result in optimizations if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("Could not find a finite noisy-data best-fit point")
    best = min(finite, key=lambda result: result.fun)
    best_fit = np.asarray(best.x, dtype=np.float64)

    def crosscorr_loglike(zp, alpha_z, beta_z, alpha_m, delta_m, mu_m, sigma_m, beta_q):
        parameters = np.asarray([zp, alpha_z, beta_z, alpha_m, delta_m, mu_m, sigma_m, beta_q])
        return -negative_loglike(parameters)

    params = {}
    for index, name in enumerate(PARAMETER_NAMES):
        lo, hi = PARAMETER_RANGES[name]
        params[name] = {
            "prior": {"min": lo, "max": hi},
            "ref": float(best_fit[index]),
            "proposal": float(max((hi - lo) * 0.005, 1e-4)),
        }
    info = {
        "force": True,
        "resume": False,
        "params": params,
        "likelihood": {
            "crosscorr_8param_posterior": {
                "external": crosscorr_loglike,
                "input_params": PARAMETER_NAMES,
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
        info["output"] = str(RESULT_DIR / f"crosscorr_8param_chain_{BASE_VERSION}_{TEST_SAMPLE_INDEX:05d}")

    print(f"Emulator: {EMULATOR_PATH}")
    print(f"Observed H1: {h1_path}")
    print(f"Observed L1: {l1_path}")
    print(f"Posterior parameters ({len(PARAMETER_NAMES)}): {PARAMETER_NAMES}")
    print(f"Training/validation/test sizes: {checkpoint['train_size']}/{checkpoint['val_size']}/{checkpoint['test_size']}")
    print(
        f"Emulator validation/test MSE: {checkpoint['best_val_mse']}/{checkpoint['test_mse']}; "
        f"test R2={test_r2}"
    )
    print("Noisy-data best fit: " + ", ".join(f"{name}={best_fit[i]:.6g}" for i, name in enumerate(PARAMETER_NAMES)))

    _, sampler = cobaya_run(info, force=True)
    posterior = collection_to_array(sampler.products()["sample"])

    # Truth is loaded only after inference and never enters the likelihood or initialization.
    truth = load_true_parameters_after_inference()
    figure_path = RESULT_DIR / f"crosscorr_8param_corner_{BASE_VERSION}_{TEST_SAMPLE_INDEX:05d}.png"
    draw_corner(posterior, truth, figure_path)

    lines = [
        f"Sample index: {TEST_SAMPLE_INDEX}",
        f"Population index: {POP_INDEX}",
        f"Emulator: {EMULATOR_PATH}",
        f"H1 observed: {h1_path}",
        f"L1 observed: {l1_path}",
        "",
    ]
    for index, name in enumerate(PARAMETER_NAMES):
        mean = float(posterior[:, index].mean())
        q16, q50, q84 = np.percentile(posterior[:, index], [16, 50, 84])
        line = f"{name:<10} mean={mean:.6f} median={q50:.6f} ci16={q16:.6f} ci84={q84:.6f} width68={q84-q16:.6f}"
        if truth is not None:
            in68 = "Y" if q16 <= truth[index] <= q84 else "N"
            line += f" true={truth[index]:.6f} in68={in68}"
        lines.append(line)
    summary = "\n".join(lines)
    print(summary)
    summary_path = RESULT_DIR / f"crosscorr_8param_summary_{BASE_VERSION}_{TEST_SAMPLE_INDEX:05d}.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved corner: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
