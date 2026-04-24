from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from eval_utils import (
        CHECKPOINT_PATH,
        PARAMETER_NAMES,
        build_model,
        build_split_dataset,
        inverse_label_normalization,
        load_checkpoint,
        sample_posterior,
    )
else:
    from .eval_utils import (
        CHECKPOINT_PATH,
        PARAMETER_NAMES,
        build_model,
        build_split_dataset,
        inverse_label_normalization,
        load_checkpoint,
        sample_posterior,
    )


NUM_POSTERIOR_SAMPLES = 800


@torch.inference_mode()
def compute_test_nll(model, waveform: torch.Tensor, label_normalized: torch.Tensor, device: torch.device) -> float:
    theta = label_normalized.unsqueeze(0).to(device)
    x = waveform.unsqueeze(0).to(device)
    return float((-model.log_prob(theta, x)).item())


def main() -> None:
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    dataset = build_split_dataset(checkpoint, split_name="test")
    model = build_model(checkpoint, device)

    print(f"Evaluating {len(dataset)} test samples...")

    nll_values = []
    per_sample_mse_values = []
    per_sample_mae_values = []

    progress = tqdm(range(len(dataset)), desc="Evaluating v5 test NLL/MSE", unit="sample")
    for sample_index in progress:
        waveform, label_normalized = dataset[sample_index]

        label_true = inverse_label_normalization(
            label_normalized,
            checkpoint["label_mean"],
            checkpoint["label_std"],
        )

        samples_normalized = sample_posterior(model, waveform, NUM_POSTERIOR_SAMPLES, device)
        posterior_samples = inverse_label_normalization(
            samples_normalized,
            checkpoint["label_mean"],
            checkpoint["label_std"],
        )
        posterior_mean = posterior_samples.mean(axis=0)

        nll = compute_test_nll(model, waveform, label_normalized, device)
        error = posterior_mean - label_true
        per_param_mse = error ** 2
        per_param_mae = np.abs(error)

        nll_values.append(nll)
        per_sample_mse_values.append(per_param_mse)
        per_sample_mae_values.append(per_param_mae)
        current_mse = float(np.mean(np.asarray(per_sample_mse_values, dtype=np.float64)))
        current_mae = float(np.mean(np.asarray(per_sample_mae_values, dtype=np.float64)))
        progress.set_postfix(
            nll=f"{np.mean(nll_values):.4f}",
            mse=f"{current_mse:.4f}",
            mae=f"{current_mae:.4f}",
        )

    nll_values = np.asarray(nll_values, dtype=np.float64)
    per_sample_mse_values = np.asarray(per_sample_mse_values, dtype=np.float64)
    per_sample_mae_values = np.asarray(per_sample_mae_values, dtype=np.float64)
    per_param_mse_mean = per_sample_mse_values.mean(axis=0)
    per_param_mse_std = per_sample_mse_values.std(axis=0)
    per_param_mae_mean = per_sample_mae_values.mean(axis=0)
    per_param_mae_std = per_sample_mae_values.std(axis=0)

    print(f"test_nll_mean={nll_values.mean():.6f}")
    print(f"test_nll_std={nll_values.std():.6f}")
    print("per_parameter_mse:")
    for name, mse_mean, mse_std in zip(PARAMETER_NAMES, per_param_mse_mean, per_param_mse_std):
        print(f"  {name}: mean={mse_mean:.6f}, std={mse_std:.6f}")
    print("per_parameter_mae:")
    for name, mae_mean, mae_std in zip(PARAMETER_NAMES, per_param_mae_mean, per_param_mae_std):
        print(f"  {name}: mean={mae_mean:.6f}, std={mae_std:.6f}")


if __name__ == "__main__":
    main()
