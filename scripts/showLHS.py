from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VERSION = "v42"
#SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m"]
#SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m","lambda_peak","mu_m","sigma_m","beta_q"]
SELECTED_PARAMETERS = ["zp","delta_m","beta_q"]
#SELECTED_PARAMETERS = ["alpha_m","delta_m","mu_m","sigma_m"]
POINT_SIZE = 12
ALPHA = 0.65


def load_samples(version: str) -> dict[str, np.ndarray]:
    sample_path = (
        Path(__file__).parent
        / "joint_hyperparams_train"
        / version
        / f"joint_hyperparams_train_{version}.npz"
    )
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    loaded = np.load(sample_path)
    return {name: loaded[name] for name in loaded.files}


def build_figure_path(version: str, parameter_names: list[str]) -> Path:
    suffix = "_".join(parameter_names)
    return Path(__file__).parent / "joint_hyperparams_train" / version / f"showLHS_{suffix}.png"


def validate_parameter_names(samples: dict[str, np.ndarray], parameter_names: list[str]) -> None:
    missing = [name for name in parameter_names if name not in samples]
    if missing:
        raise KeyError(f"Unknown parameter names: {missing}")


def print_basic_summary(samples: dict[str, np.ndarray], parameter_names: list[str]) -> None:
    for name in parameter_names:
        values = samples[name]
        print(
            f"{name}: "
            f"min={values.min():.6f} "
            f"max={values.max():.6f} "
            f"mean={values.mean():.6f} "
            f"std={values.std():.6f}"
        )


def pca_project(data: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = data - data.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    explained_variance = singular_values**2
    explained_ratio = explained_variance / explained_variance.sum()
    return projected, explained_ratio[:n_components]


def show_2d(samples: dict[str, np.ndarray], parameter_names: list[str], version: str) -> None:
    x_name, y_name = parameter_names
    x = samples[x_name]
    y = samples[y_name]
    figure_path = build_figure_path(version, parameter_names)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=POINT_SIZE, alpha=ALPHA)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"LHS samples ({version}): {x_name} vs {y_name}")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    print(f"Saved figure: {figure_path}")
    plt.show()


def show_3d(samples: dict[str, np.ndarray], parameter_names: list[str], version: str) -> None:
    x_name, y_name, z_name = parameter_names
    x = samples[x_name]
    y = samples[y_name]
    z = samples[z_name]
    figure_path = build_figure_path(version, parameter_names)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=POINT_SIZE, alpha=ALPHA)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title(f"LHS samples ({version}): {x_name}, {y_name}, {z_name}")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    print(f"Saved figure: {figure_path}")
    plt.show()


def show_pca_2d(samples: dict[str, np.ndarray], parameter_names: list[str], version: str) -> None:
    figure_path = build_figure_path(version, parameter_names)
    data = np.stack([samples[name] for name in parameter_names], axis=1)
    projected, explained_ratio = pca_project(data, n_components=2)

    plt.figure(figsize=(7, 6))
    plt.scatter(projected[:, 0], projected[:, 1], s=POINT_SIZE, alpha=ALPHA)
    plt.xlabel(f"PC1 ({explained_ratio[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({explained_ratio[1] * 100:.2f}%)")
    plt.title(f"LHS PCA ({version}): {', '.join(parameter_names)}")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    print(f"Saved figure: {figure_path}")
    plt.show()


def main() -> None:
    if len(SELECTED_PARAMETERS) < 2:
        raise ValueError("SELECTED_PARAMETERS must contain at least 2 parameter names.")

    samples = load_samples(VERSION)
    validate_parameter_names(samples, SELECTED_PARAMETERS)

    print(f"Version: {VERSION}")
    print(f"Selected parameters: {SELECTED_PARAMETERS}")
    print_basic_summary(samples, SELECTED_PARAMETERS)

    if len(SELECTED_PARAMETERS) == 2:
        show_2d(samples, SELECTED_PARAMETERS, VERSION)
    elif len(SELECTED_PARAMETERS) == 3:
        show_3d(samples, SELECTED_PARAMETERS, VERSION)
    else:
        show_pca_2d(samples, SELECTED_PARAMETERS, VERSION)


if __name__ == "__main__":
    main()
