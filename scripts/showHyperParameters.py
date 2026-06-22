import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")


PARAMETER_NAMES = {
    "alpha_z": "Redshift Ascent Index (alpha_z)",
    "beta_z": "Redshift Descent Index (beta_z)",
    "zp": "Peak Redshift (zp)",
    "alpha_m": "Mass Power-law Index (alpha_m)",
    "m_max": "Max BH Mass Cutoff",
    "delta_m": "Low Mass Smoothing (delta_m)",
    "m_min": "Min BH Mass Cutoff",
    "lambda_peak": "Gaussian Peak Fraction (lambda_peak)",
    "mu_m": "Gaussian Peak Mean (mu_m)",
    "sigma_m": "Gaussian Peak Std Dev (sigma_m)",
    "beta_q": "Mass Ratio Index (beta_q)",
}

EXCLUDED_PARAMETERS = set()


def print_param_ranges(path):
    try:
        data = np.load(path)
        print("Parameter summary:")
        print(
            f"{'param':<12} | {'mean':<10} | {'std':<10} | "
            f"{'min':<10} | {'max':<10} | {'range':<24} | description"
        )
        print("-" * 128)

        for key in data.files:
            if key in EXCLUDED_PARAMETERS:
                continue
            values = data[key]
            v_mean = float(np.mean(values))
            v_std = float(np.std(values))
            v_min = float(np.min(values))
            v_max = float(np.max(values))
            range_text = f"[{v_min:.6f}, {v_max:.6f}]"
            name = PARAMETER_NAMES.get(key, "Unknown")
            print(
                f"{key:<12} | {v_mean:<10.6f} | {v_std:<10.6f} | "
                f"{v_min:<10.6f} | {v_max:<10.6f} | {range_text:<24} | {name}"
            )

        data.close()
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
    except Exception as exc:
        print(f"Error: {exc}")


def save_individual_distributions(file_path, output_dir="hyperparam_distributions"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        data = np.load(file_path)
        keys = [key for key in data.files if key not in EXCLUDED_PARAMETERS]

        for key in keys:
            values = data[key]
            plt.figure(figsize=(8, 5))
            sns.histplot(values, kde=True, color="royalblue", edgecolor="white")
            plt.title(f"Distribution of {PARAMETER_NAMES.get(key, key)}", fontsize=14, fontweight="bold")
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.grid(axis="y", alpha=0.3)

            save_path = os.path.join(output_dir, f"{key}_distribution.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"{key:<12} | saved to {save_path}")

        data.close()
        print(f"\n[Done] Saved {len(keys)} individual parameter plots to '{output_dir}/'")
    except Exception as exc:
        print(f"Error while generating plots: {exc}")


def plot_param_distributions(path, save_name="./hyperparam_distributions/joint_hyperparam_distributions.png"):
    try:
        data = np.load(path)
        keys = sorted([key for key in data.files if key not in EXCLUDED_PARAMETERS])
        num_params = len(keys)

        cols = 3
        rows = math.ceil(num_params / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        print("\nBuilding joint summary plot...")
        print(f"{'param':<12} | {'mean':<10} | {'std':<10} | {'min':<10} | {'max':<10}")
        print("-" * 65)

        for i, key in enumerate(keys):
            values = data[key]
            v_mean = float(np.mean(values))
            v_std = float(np.std(values))
            v_min = float(np.min(values))
            v_max = float(np.max(values))
            print(f"{key:<12} | {v_mean:<10.6f} | {v_std:<10.6f} | {v_min:<10.6f} | {v_max:<10.6f}")

            sns.histplot(values, kde=True, ax=axes[i], color="royalblue", edgecolor="white")
            axes[i].set_title(PARAMETER_NAMES.get(key, key), fontsize=10, fontweight="bold")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"\n[Done] Saved joint distribution plot to: {save_name}")
        data.close()
    except Exception as exc:
        print(f"Execution failed: {exc}")

#version = "v50"
#version = "v44"
#version = "v45"
version = "v63"
file_path = f"./joint_hyperparams_train/{version}/joint_hyperparams_train_{version}.npz"

print_param_ranges(file_path)
save_individual_distributions(file_path)
plot_param_distributions(file_path)
