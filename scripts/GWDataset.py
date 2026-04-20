import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from getMean_Std import calculate_stats


class GWDataset(Dataset):
    def __init__(self, data_dir, mean=None, std=None, show_progress=True, dataset_name=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name or os.path.basename(os.path.normpath(data_dir))
        self.file_list = sorted(f for f in os.listdir(data_dir) if f.endswith('.npy'))
        self.printRange = True

        print(f"[{self.dataset_name}] Found {len(self.file_list)} files in {data_dir}")

        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            mean_array, std_array = calculate_stats(
                data_dir,
                show_progress=show_progress,
                desc=f"[{self.dataset_name}] Loading labels"
            )
            print(f'均值: {mean_array}，标准差: {std_array}')

            mean_filtered = np.delete(mean_array, 2)
            std_filtered = np.delete(std_array, 2)

            self.mean = torch.tensor(mean_filtered).float()
            self.std = torch.tensor(std_filtered).float()

        self.scale_factor = 1e23

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample_dict = np.load(file_path, allow_pickle=True).item()

        data = sample_dict['data'] * self.scale_factor
        label = sample_dict['label']
        label_filtered = np.concatenate([label[:2], label[3:]])

        data_tensor = torch.from_numpy(data).float().unsqueeze(0)
        label_tensor = torch.tensor(label_filtered).float()

        if self.mean is not None and self.std is not None:
            label_tensor = (label_tensor - self.mean) / self.std

        return data_tensor, label_tensor


if __name__ == "__main__":
    h1_train_path = './processed_data_train/H1_splits'
    train_dataset = GWDataset(h1_train_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    for images, labels in train_loader:
        print(f"输入张量形状 (Batch, Channel, Length): {images.shape}")
        print(f"标签张量形状: {labels.shape}")
        break
