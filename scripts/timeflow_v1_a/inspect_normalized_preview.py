from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_VERSION = "v8"
PREVIEW_NUM_SAMPLES = 500
PREVIEW_NUM_FEATURES = 500


def main() -> None:
    base_dir = Path(__file__).resolve().parent / f"outputs_dataset_{DATA_VERSION}"
    features_path = base_dir / f"normalized_preview_features_{PREVIEW_NUM_SAMPLES}x{PREVIEW_NUM_FEATURES}_v1_a.npy"
    labels_path = base_dir / f"normalized_preview_labels_{PREVIEW_NUM_SAMPLES}_v1_a.npy"

    if not features_path.exists():
        raise FileNotFoundError(f"Preview features not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Preview labels not found: {labels_path}")

    features = np.load(features_path)
    labels = np.load(labels_path)

    print(f"Features path: {features_path}")
    print(f"Labels path: {labels_path}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Features mean: {features.mean():.6f}")
    print(f"Features std: {features.std():.6f}")
    print(f"Features min: {features.min():.6f}")
    print(f"Features max: {features.max():.6f}")

    sample_means = features.mean(axis=(1, 2))
    sample_stds = features.std(axis=(1, 2))
    print(f"Per-sample mean range: [{sample_means.min():.6f}, {sample_means.max():.6f}]")
    print(f"Per-sample std range: [{sample_stds.min():.6f}, {sample_stds.max():.6f}]")

    feature_mean_map = features.mean(axis=0)
    feature_std_map = features.std(axis=0)
    flat_features = features.reshape(features.shape[0], -1)
    mean_feature = flat_features.mean(axis=0)
    distances_to_mean = np.linalg.norm(flat_features - mean_feature[None, :], axis=1)
    print(
        "Distance-to-mean range: "
        f"[{distances_to_mean.min():.6f}, {distances_to_mean.max():.6f}], "
        f"mean={distances_to_mean.mean():.6f}"
    )

    num_compare = min(25, features.shape[0])
    compare_block = flat_features[:num_compare]
    diff_matrix = np.linalg.norm(compare_block[:, None, :] - compare_block[None, :, :], axis=2)
    ref_index = 0
    ref_diff_map = features[ref_index] - feature_mean_map

    figure_path = base_dir / f"normalized_preview_inspection_{PREVIEW_NUM_SAMPLES}x{PREVIEW_NUM_FEATURES}_v1_a.png"
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    im0 = axes[0, 0].imshow(features[0], aspect="auto", origin="lower", cmap="coolwarm")
    axes[0, 0].set_title("Sample 0 Features")
    axes[0, 0].set_ylabel("Channel")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(feature_mean_map, aspect="auto", origin="lower", cmap="coolwarm")
    axes[0, 1].set_title("Mean Across Samples")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(feature_std_map, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_title("Std Across Samples")
    axes[1, 0].set_xlabel("Feature Index")
    axes[1, 0].set_ylabel("Channel")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(ref_diff_map, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1, 1].set_title("Sample 0 Minus Mean Map")
    axes[1, 1].set_xlabel("Feature Index")
    axes[1, 1].set_ylabel("Channel")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    axes[2, 0].plot(sample_means, label="sample mean")
    axes[2, 0].plot(sample_stds, label="sample std")
    axes[2, 0].plot(distances_to_mean, label="L2 to mean")
    axes[2, 0].set_title("Per-sample Summary")
    axes[2, 0].set_xlabel("Sample Index")
    axes[2, 0].legend()

    im4 = axes[2, 1].imshow(diff_matrix, origin="lower", cmap="magma")
    axes[2, 1].set_title(f"L2 Distance Matrix (first {num_compare} samples)")
    axes[2, 1].set_xlabel("Sample Index")
    axes[2, 1].set_ylabel("Sample Index")
    fig.colorbar(im4, ax=axes[2, 1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
