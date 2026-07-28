import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries

NUM_SPLITS = 1
# NUM_SPLITS = 8
#version = "v61"  # this will be changed
#version = "v61_random_two_noise_1000"  # this will be changed
#version = "v61_random_two_noise_500"  # this will be changed
version = "v61_random_two_noise_200"

POP_PATTERN = re.compile(r"(pop\d{5})")


def get_labels_by_pop(file_path):
    df = pd.read_csv(file_path)
    labels_by_pop = {}

    for row in df.itertuples(index=False):
        sample_id = str(row[0])
        match = POP_PATTERN.search(sample_id)
        if match is None:
            continue

        pop_id = match.group(1)
        if pop_id not in labels_by_pop:
            labels_by_pop[pop_id] = list(row[2:])

    return labels_by_pop


def extract_pop_id(filename, detector_name):
    sample_id = filename.replace(f"{detector_name}-SUPERIMPOSED-", "").replace(".gwf", "")
    match = POP_PATTERN.search(sample_id)
    if match is None:
        return None
    return match.group(1)


def get_superimposed_data_by_pop(detector_name, base_path, save_path, labels_by_pop, num_splits=NUM_SPLITS):
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(base_path):
        print(f"Input directory not found: {base_path}")
        return

    files = sorted(f for f in os.listdir(base_path) if f.endswith(".gwf"))
    print(
        f"Start processing detector {detector_name}: {len(files)} files, "
        f"{num_splits} split(s) per file."
    )

    count = 0
    for filename in files:
        pop_id = extract_pop_id(filename, detector_name)
        if pop_id is None:
            print(f"Skip {filename}: unable to parse pop id.")
            continue

        if pop_id not in labels_by_pop:
            print(f"Skip {filename}: label not found for {pop_id}.")
            continue

        label = labels_by_pop[pop_id]
        file_full_path = os.path.join(base_path, filename)
        base_name = os.path.splitext(filename)[0]

        all_splits_exist = all(
            os.path.exists(os.path.join(save_path, f"{base_name}_p{i}.npy"))
            for i in range(num_splits)
        )
        if all_splits_exist:
            print(f"Skip already processed file: {filename}")
            continue

        try:
            data = TimeSeries.read(file_full_path, channel=f"{detector_name}:TEST_INJ").value
            total_length = len(data)
            split_length = total_length // num_splits

            for i in range(num_splits):
                start = i * split_length
                end = (i + 1) * split_length
                chunk = data[start:end]
                sample = {
                    "data": chunk,
                    "label": label,
                }
                split_filename = f"{base_name}_p{i}.npy"
                np.save(os.path.join(save_path, split_filename), sample)

            count += 1
            if count % 20 == 0:
                print(f"Processed {count} superimposed files from detector {detector_name}.")

        except Exception as exc:
            print(f"Failed processing {filename}: {exc}")

    print(f"{detector_name} done. Split files saved to: {save_path}")


if __name__ == "__main__":
    detectors = ["H1", "L1"]
#    detectors = ["L1"]
    num_splits = NUM_SPLITS

    scripts_dir = Path(__file__).resolve().parent
    input_root = scripts_dir / "training_set_superimposed" / version
    save_root = scripts_dir / "processed_data_superimposed" / version

    base_label_path = input_root / "train_idx.csv"
    labels_by_pop = get_labels_by_pop(base_label_path)

    for detector_name in detectors:
        base_path = input_root / detector_name
        save_path = save_root / f"{detector_name}_splits"
        get_superimposed_data_by_pop(
            detector_name,
            str(base_path),
            str(save_path),
            labels_by_pop,
            num_splits=num_splits,
        )
