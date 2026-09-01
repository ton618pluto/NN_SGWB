import os
from pathlib import Path

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


#version_number = "v71"
#version_number = "v61"   # this will be changed
#version_number = "v100"   # this will be changed
version_number = "v100"   # this will be changed
#scale = 500   # this will be changed
scale = 100   # this will be changed

#input_version = f"fixed_one_test_{version_number}_random_two_noise_{scale}"   # this will be changed
input_version = f"fixed_one_test_{version_number}_random_two_noise_{scale}_shifted"  # this will be changed

# label_version = f"fixed_one_test_{version_number}"   # this will be changed
label_version = f"fixed_one_test_{version_number}_random_two_noise_100"   # this will be changed

#output_version = f"fixed_one_test_{version_number}_random_two_noise_{scale}"   # this will be changed
output_version = f"fixed_one_test_{version_number}_random_two_noise_{scale}_shifted"   # this will be changed
NUM_SPLITS = 1




def get_label(file_path):
    df = pd.read_csv(file_path)
    return {row[0]: list(row[2:]) for row in df.itertuples(index=False)}


def get_sgwb_data(d_name, base_path, save_path, labels_dict, num_splits=NUM_SPLITS):
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(base_path):
        print(f"Input directory not found: {base_path}")
        return

    files = sorted(f for f in os.listdir(base_path) if f.endswith(".gwf"))
    print(f"Processing detector {d_name}, {len(files)} files, each split into {num_splits} segments")

    count = 0
    for filename in files:
        temp_filename = filename.replace(f"{d_name}-SUPERIMPOSED-", "").replace(".gwf", "")
        if temp_filename not in labels_dict:
            print(f"Skip {filename}: no matching label {temp_filename}")
            continue

        label = labels_dict[temp_filename]
        file_full_path = os.path.join(base_path, filename)
        base_name = os.path.splitext(filename)[0]

        all_splits_exist = all(
            os.path.exists(os.path.join(save_path, f"{base_name}_p{i}.npy"))
            for i in range(num_splits)
        )
        if all_splits_exist:
            print(f"Skip processed file: {filename}")
            continue

        try:
            data = TimeSeries.read(file_full_path, channel=f"{d_name}:TEST_INJ").value
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
            if count % 100 == 0:
                print(f"Processed {count} superimposed CBC-SGWB files for detector {d_name}")

        except Exception as exc:
            print(f"Failed processing {filename}: {exc}")

    print(f"{d_name} processing complete, saved to: {save_path}")


if __name__ == "__main__":
    detectors = ["H1", "L1"]
    num_splits = NUM_SPLITS
    folder_name = "training_set0"

    scripts_dir = Path(__file__).resolve().parent
    input_root = scripts_dir / "training_set_superimposed_test" / input_version
    label_root = scripts_dir / "training_set_test" / label_version / folder_name
    save_root = scripts_dir / "processed_data_test_superimposed" / output_version

    base_label_path = label_root / "train_idx.csv"
    labels_dict = get_label(base_label_path)

    for d_name in detectors:
        base_path = input_root / d_name
        save_path = save_root / f"{d_name}_splits"
        get_sgwb_data(d_name, str(base_path), str(save_path), labels_dict, num_splits=num_splits)
