from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PARAMETER_RANGES = {
    "zp": (1.4, 2.4),
    "alpha_z": (2.1, 3.6),
    "alpha_m": (3.0, 3.5),
    "m_max": (10.0, 100.0),
    "delta_m": (0.0, 10.0),
    "m_min": (2.0, 10.0),
    "lambda_peak": (0.02, 0.06),
    "mu_m": (33.2, 35.4),
    "sigma_m": (3.52, 3.60),
    "beta_q": (-5.0, 5.0),
}

VERSION_DIRS = [
    "timeflow_v5",
    "timeflow_v6",
    "timeflow_v7",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_config(version_dir: Path) -> Path | None:
    outputs_dir = version_dir / "outputs"
    resume_path = outputs_dir / "run_config_resume.json"
    legacy_path = outputs_dir / "run_config.json"
    if resume_path.exists():
        return resume_path
    if legacy_path.exists():
        return legacy_path
    return None


def infer_parameter_names(version_name: str, config: dict) -> list[str]:
    if "parameter_names" in config:
        return list(config["parameter_names"])
    if version_name == "timeflow_v5":
        return ["zp", "alpha_z", "alpha_m", "m_max", "delta_m", "m_min", "lambda_peak", "mu_m", "sigma_m", "beta_q"]
    if version_name == "timeflow_v6":
        return ["zp", "alpha_m", "m_max", "delta_m", "m_min", "lambda_peak", "mu_m", "sigma_m", "beta_q"]
    if version_name == "timeflow_v7":
        return ["alpha_m", "delta_m", "lambda_peak", "mu_m", "beta_q"]
    raise KeyError(f"Cannot infer parameter names for {version_name}")


def print_version_summary(version_dir: Path) -> None:
    config_path = find_run_config(version_dir)
    if config_path is None:
        print(f"\n[{version_dir.name}] run_config not found under {version_dir / 'outputs'}")
        return

    config = load_json(config_path)
    parameter_names = infer_parameter_names(version_dir.name, config)
    label_mean = np.asarray(config["label_mean"], dtype=np.float64)
    label_std = np.asarray(config["label_std"], dtype=np.float64)

    print(f"\n[{version_dir.name}]")
    print(f"Config: {config_path}")
    print(
        f"{'param':<12}"
        f"{'raw_min':>12}"
        f"{'raw_max':>12}"
        f"{'mean':>12}"
        f"{'std':>12}"
        f"{'norm_min':>12}"
        f"{'norm_max':>12}"
    )
    print("-" * 84)

    for index, name in enumerate(parameter_names):
        if name not in PARAMETER_RANGES:
            print(
                f"{name:<12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{label_mean[index]:>12.6f}"
                f"{label_std[index]:>12.6f}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
            )
            continue

        raw_min, raw_max = PARAMETER_RANGES[name]
        std = label_std[index]
        if std == 0:
            norm_min = float("nan")
            norm_max = float("nan")
        else:
            norm_min = (raw_min - label_mean[index]) / std
            norm_max = (raw_max - label_mean[index]) / std

        print(
            f"{name:<12}"
            f"{raw_min:>12.6f}"
            f"{raw_max:>12.6f}"
            f"{label_mean[index]:>12.6f}"
            f"{label_std[index]:>12.6f}"
            f"{norm_min:>12.6f}"
            f"{norm_max:>12.6f}"
        )


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    print("Normalized label range summary")
    print("norm_min / norm_max are computed as (raw_bound - label_mean) / label_std")

    for version_name in VERSION_DIRS:
        print_version_summary(scripts_dir / version_name)


if __name__ == "__main__":
    main()
