from pathlib import Path
import json

import numpy as np


DATA_VERSION = "v100"
USE_MODEL2 = False
NUM_SAMPLES_TO_SHOW = 10
NUM_TIMEPOINTS_TO_SHOW = 10


def build_output_dir() -> Path:
    base_dir = Path(__file__).resolve().parent
#    output_name = f"outputs_dataset_{DATA_VERSION}_old_correct"
    output_name = f"outputs_dataset_{DATA_VERSION}_new"
#    output_name = f"outputs_dataset_{DATA_VERSION}"
    if USE_MODEL2:
        output_name += "_model2"
    return base_dir / output_name


def load_train_records(output_dir: Path) -> list[dict]:
    split_manifest_path = output_dir / "data_split_v1_a.json"
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    return split_manifest["train_records"]


def main() -> None:
    output_dir = build_output_dir()
    features_path = output_dir / "normalized_preview_features_50x500_v1_a.npy"
    labels_path = output_dir / "normalized_preview_labels_50_v1_a.npy"

    if not features_path.exists():
        raise FileNotFoundError(f"Features preview file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels preview file not found: {labels_path}")

    features = np.load(features_path)
    labels = np.load(labels_path)
    train_records = load_train_records(output_dir)

    print(f"Output dir: {output_dir}")
    print(f"Features file: {features_path}")
    print(f"Labels file:   {labels_path}")
    print()

    print(f"features.shape = {features.shape}, dtype = {features.dtype}")
    print(f"labels.shape   = {labels.shape}, dtype = {labels.dtype}")
    print()

    sample_count = min(NUM_SAMPLES_TO_SHOW, features.shape[0], labels.shape[0])
    time_count = min(NUM_TIMEPOINTS_TO_SHOW, features.shape[2])

    for sample_idx in range(sample_count):
        record = train_records[sample_idx]
        h1_sample = np.load(record["h1"], allow_pickle=True).item()
        l1_sample = np.load(record["l1"], allow_pickle=True).item()
        raw_label = np.asarray(h1_sample["label"])
        raw_h1 = np.asarray(h1_sample["data"])
        raw_l1 = np.asarray(l1_sample["data"])

        print(f"sample[{sample_idx}]")
        print(f"  preview_label(raw selected) = {labels[sample_idx]}")
        print(f"  source_label(full raw)      = {raw_label}")
        print(f"  raw_H1[:{time_count}]       = {raw_h1[:time_count]}")
        print(f"  raw_L1[:{time_count}]       = {raw_l1[:time_count]}")
        print(f"  norm_H1[:{time_count}]      = {features[sample_idx, 0, :time_count]}")
        print(f"  norm_L1[:{time_count}]      = {features[sample_idx, 1, :time_count]}")
        print()


if __name__ == "__main__":
    main()
