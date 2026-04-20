from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _sample_key(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("-", 1)[-1]


class GWDataset(Dataset):
    def __init__(
        self,
        h1_dir,
        l1_dir=None,
        mean=None,
        std=None,
        show_progress=True,
        dataset_name=None,
    ):
        self.h1_dir = Path(h1_dir)
        self.l1_dir = Path(l1_dir) if l1_dir is not None else None
        self.dataset_name = dataset_name or self.h1_dir.name
        self.show_progress = show_progress
        self.scale_factor = 1e23

        if not self.h1_dir.exists():
            raise FileNotFoundError(f"H1 directory not found: {self.h1_dir}")

        self.records = self._build_records()
        if not self.records:
            raise RuntimeError(f"[{self.dataset_name}] No matched samples found.")

        if mean is not None and std is not None:
            self.mean = mean.float()
            self.std = std.float()
        else:
            mean_array, std_array = self._calculate_stats()
            self.mean = torch.tensor(mean_array).float()
            self.std = torch.tensor(std_array).float()

        self.num_channels = 2 if self.records[0]["l1"] is not None else 1

        print(
            f"[{self.dataset_name}] Found {len(self.records)} samples "
            f"({self.num_channels} channel{'s' if self.num_channels > 1 else ''})"
        )

    def _build_records(self):
        h1_files = sorted(path for path in self.h1_dir.iterdir() if path.suffix == ".npy")

        if self.l1_dir is None:
            return [{"h1": path, "l1": None} for path in h1_files]

        if not self.l1_dir.exists():
            raise FileNotFoundError(f"L1 directory not found: {self.l1_dir}")

        l1_map = {_sample_key(path.name): path for path in self.l1_dir.iterdir() if path.suffix == ".npy"}
        records = []
        progress = h1_files
        if self.show_progress:
            progress = tqdm(h1_files, desc=f"[{self.dataset_name}] Matching H1/L1", unit="file")

        for h1_path in progress:
            l1_path = l1_map.get(_sample_key(h1_path.name))
            if l1_path is not None:
                records.append({"h1": h1_path, "l1": l1_path})

        return records

    def _calculate_stats(self):
        labels = []
        progress = self.records
        if self.show_progress:
            progress = tqdm(self.records, desc=f"[{self.dataset_name}] Loading labels", unit="file")

        for record in progress:
            sample_dict = np.load(record["h1"], allow_pickle=True).item()
            label = np.asarray(sample_dict["label"], dtype=np.float32)
            label_filtered = np.concatenate([label[:2], label[3:]])
            labels.append(label_filtered)

        all_labels = np.stack(labels, axis=0)
        mean_array = all_labels.mean(axis=0)
        std_array = all_labels.std(axis=0)
        std_array[std_array == 0] = 1.0
        return mean_array, std_array

    def __len__(self):
        return len(self.records)

    def _load_channel(self, sample_dict):
        data = np.asarray(sample_dict["data"], dtype=np.float32) * self.scale_factor
        return data - data.mean(dtype=np.float32)

    def __getitem__(self, idx):
        record = self.records[idx]
        h1_sample = np.load(record["h1"], allow_pickle=True).item()

        channels = [self._load_channel(h1_sample)]
        if record["l1"] is not None:
            l1_sample = np.load(record["l1"], allow_pickle=True).item()
            channels.append(self._load_channel(l1_sample))

        label = np.asarray(h1_sample["label"], dtype=np.float32)
        label_filtered = np.concatenate([label[:2], label[3:]])

        data_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()
        label_tensor = torch.tensor(label_filtered).float()

        if self.mean is not None and self.std is not None:
            label_tensor = (label_tensor - self.mean) / self.std

        return data_tensor, label_tensor
