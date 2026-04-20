import os
from pathlib import Path

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


def get_label(file_path):
    df = pd.read_csv(file_path)
    return {row[0]: list(row[2:]) for row in df.itertuples(index=False)}


def get_superimposed_data(d_name, base_path, save_path, labels_dict, num_splits=8):
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(base_path):
        print(f"输入目录不存在: {base_path}")
        return

    files = sorted(f for f in os.listdir(base_path) if f.endswith(".gwf"))
    print(f"开始处理探测器 {d_name}，共 {len(files)} 个文件，每个文件切分为 {num_splits} 段。")

    count = 0
    for filename in files:
        temp_filename = filename.replace(f"{d_name}-SUPERIMPOSED-", "").replace(".gwf", "")
        if temp_filename not in labels_dict:
            print(f"跳过 {filename}: 没找到对应标签 {temp_filename}")
            continue

        label = labels_dict[temp_filename]
        file_full_path = os.path.join(base_path, filename)
        base_name = os.path.splitext(filename)[0]

        all_splits_exist = all(
            os.path.exists(os.path.join(save_path, f"{base_name}_p{i}.npy"))
            for i in range(num_splits)
        )
        if all_splits_exist:
            print(f"跳过已处理文件: {filename}")
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
            if count % 20 == 0:
                print(f"已处理 {count} 个来自探测器 {d_name} 的叠加文件。")

        except Exception as exc:
            print(f"处理 {filename} 失败: {exc}")

    print(f"{d_name} 处理完成，切分文件保存在: {save_path}")


if __name__ == "__main__":
    detectors = ["H1", "L1"]
    version = "v0"

    scripts_dir = Path(__file__).resolve().parent
    input_root = scripts_dir / "training_set_superimposed" / version
    save_root = scripts_dir / "processed_data_superimposed" / version

    base_label_path = input_root / "train_idx.csv"
    labels_dict = get_label(base_label_path)

    for d_name in detectors:
        base_path = input_root / d_name
        save_path = save_root / f"{d_name}_splits"
        get_superimposed_data(d_name, str(base_path), str(save_path), labels_dict)
