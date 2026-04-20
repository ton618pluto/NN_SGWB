import os

import numpy as np
from tqdm import tqdm


def calculate_stats(data_dir, show_progress=True, desc=None):
    all_labels = []
    file_list = sorted(f for f in os.listdir(data_dir) if f.endswith('.npy'))
    progress_desc = desc or f"Calculating stats: {os.path.basename(data_dir)}"

    print("正在统计标签分布...")
    iterator = tqdm(file_list, desc=progress_desc, unit="file") if show_progress else file_list

    for file_name in iterator:
        path = os.path.join(data_dir, file_name)
        sample = np.load(path, allow_pickle=True).item()
        all_labels.append(sample['label'])

    all_labels = np.array(all_labels)
    mean = np.mean(all_labels, axis=0)
    std = np.std(all_labels, axis=0)

    std[std == 0] = 1.0

    return mean, std


if __name__ == "__main__":
    train_mean, train_std = calculate_stats('./processed_data_train/H1_splits')
    print(f"Mean: {train_mean}")
    print(f"Std: {train_std}")
