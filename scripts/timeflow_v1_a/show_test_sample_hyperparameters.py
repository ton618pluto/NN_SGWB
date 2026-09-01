from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

#DATA_VERSION = "v22"
#DATA_VERSION = "v64"
#DATA_VERSION = "v80"
#DATA_VERSION = "v81"
#DATA_VERSION = "v70"
#DATA_VERSION = "v71"
#DATA_VERSION = "v61"
#DATA_VERSION = "v61_random_two_noise"
#DATA_VERSION = "v61_random_two_noise_1000"
#DATA_VERSION = "v61_random_two_noise_500"
#DATA_VERSION = "v61_random_two_noise_100"
#DATA_VERSION = "v100"
DATA_VERSION = "v100_random_two_noise_100"
USE_MODEL2 = False
TEST_SAMPLE_INDEX = 0  # this will be changed

FULL_PARAMETER_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)
SPLIT_PATTERN = re.compile(r"_p\d+$")


def load_run_config(output_dir: Path) -> dict:
    config_path = output_dir / "run_config_resume.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_split_manifest_path(run_config: dict, output_dir: Path) -> Path:
    configured_path = Path(run_config["split_manifest"])
    if configured_path.exists():
        return configured_path
    fallback_path = output_dir / "data_split_v1_a.json"
    if fallback_path.exists():
        return fallback_path
    return configured_path


def resolve_record_path(path_str: str) -> Path:
    configured_path = Path(path_str)
    if configured_path.exists():
        return configured_path

    parts = configured_path.parts
    repo_scripts_dir = Path(__file__).resolve().parent.parent
    for folder_name in (
        "processed_data_train",
        "processed_data_superimposed",
        "processed_data_test_superimposed",
        "processed_data_test",
    ):
        if folder_name in parts:
            anchor = parts.index(folder_name)
            fallback_path = repo_scripts_dir.joinpath(*parts[anchor:])
            if fallback_path.exists():
                return fallback_path
    return configured_path


def build_output_dir() -> Path:
    output_name = f"outputs_dataset_{DATA_VERSION}"
    if USE_MODEL2:
        output_name += "_model2"
    return Path(__file__).resolve().parent / output_name


def label_to_hyperparameters(label) -> dict[str, float]:
    if isinstance(label, dict):
        return {name: float(label[name]) for name in FULL_PARAMETER_NAMES}

    label_array = np.asarray(label, dtype=np.float64).reshape(-1)
    if label_array.size != len(FULL_PARAMETER_NAMES):
        raise ValueError(
            f"Unexpected label length: {label_array.size}. "
            f"Expected {len(FULL_PARAMETER_NAMES)}"
        )
    return {
        name: float(value)
        for name, value in zip(FULL_PARAMETER_NAMES, label_array)
    }


def infer_gwf_data_version(data_version: str) -> str:
    for marker in ("_random_two_noise", "_shifted", "_float64"):
        if marker in data_version:
            return data_version.split(marker, 1)[0]
    return data_version


def sample_id_from_record_path(path_str: str, detector: str) -> str:
    stem = Path(path_str).stem
    prefix = f"{detector}-"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    stem = SPLIT_PATTERN.sub("", stem)
    return stem


def load_label_from_csv(label_path: Path, sample_id: str) -> dict[str, float]:
    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row and row[0] == sample_id:
                return label_to_hyperparameters(row[2:])
    raise KeyError(f"Label not found for sample ID '{sample_id}' in {label_path}")


def main() -> None:
    output_dir = build_output_dir()
    run_config = load_run_config(output_dir)

    split_manifest_path = resolve_split_manifest_path(run_config, output_dir)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    test_records = split_manifest["test_records"]

    if TEST_SAMPLE_INDEX < 0 or TEST_SAMPLE_INDEX >= len(test_records):
        raise IndexError(
            f"TEST_SAMPLE_INDEX out of range: {TEST_SAMPLE_INDEX}. "
            f"Available test indices: 0 to {len(test_records) - 1}"
        )

    record = test_records[TEST_SAMPLE_INDEX]
    h1_path = resolve_record_path(record["h1"])
    l1_path = resolve_record_path(record["l1"])
    selected_names = list(run_config["parameter_names"])
    sample_source = "processed_npy"
    label_path: Path | None = None
    gwf_h1_path: Path | None = None
    gwf_l1_path: Path | None = None

    if h1_path.exists():
        sample = np.load(h1_path, allow_pickle=True).item()
        hyperparameters = label_to_hyperparameters(sample["label"])
    else:
        scripts_dir = Path(__file__).resolve().parent.parent
        gwf_data_version = infer_gwf_data_version(DATA_VERSION)
        sample_id = sample_id_from_record_path(record["h1"], "H1")
        label_path = scripts_dir / "training_set" / gwf_data_version / "training_set0" / "train_idx.csv"
        gwf_h1_path = scripts_dir / "training_set" / gwf_data_version / "training_set0" / "H1" / f"H1-{sample_id}.gwf"
        gwf_l1_path = scripts_dir / "training_set" / gwf_data_version / "training_set0" / "L1" / f"L1-{sample_id}.gwf"
        hyperparameters = load_label_from_csv(label_path, sample_id)
        sample_source = "training_set_label_csv"

    print(f"Data version: {DATA_VERSION}")
    print(f"Model variant: {'model2' if USE_MODEL2 else 'model1'}")
    print(f"Output dir: {output_dir}")
    print(f"Split manifest: {split_manifest_path}")
    print(f"Test sample index: {TEST_SAMPLE_INDEX}")
    print(f"Sample source: {sample_source}")
    print(f"H1 file: {h1_path}")
    print(f"L1 file: {l1_path}")
    if label_path is not None:
        print(f"Label CSV: {label_path}")
    if gwf_h1_path is not None and gwf_l1_path is not None:
        print(f"H1 GWF file: {gwf_h1_path}")
        print(f"L1 GWF file: {gwf_l1_path}")
    print()
    print("All hyperparameters:")
    for name in FULL_PARAMETER_NAMES:
        print(f"  {name}: {hyperparameters[name]}")
    print()
    print("Training target parameters for this checkpoint:")
    for name in selected_names:
        print(f"  {name}: {hyperparameters[name]}")


if __name__ == "__main__":
    main()
