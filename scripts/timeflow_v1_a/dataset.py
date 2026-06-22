from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


PARAMETER_SPECS = {
    "zp": {"index": 0, "range": (0.85, 2.85)},
    "alpha_z": {"index": 1, "range": (1.0, 5.0)},
    "beta_z": {"index": 2, "range": (1.0, 5.0)},
    "alpha_m": {"index": 3, "range": (-3.0, 7.0)},
    "m_max": {"index": 4, "range": (100.0, 100.0)},
    "delta_m": {"index": 5, "range": (0.0, 10.0)},
    "m_min": {"index": 6, "range": (2.0, 2.0)},
#    "lambda_peak": {"index": 7, "range": (0.02, 0.06)},
    "lambda_peak": {"index": 7, "range": (0.04, 0.04)},
#    "mu_m": {"index": 8, "range": (25.0, 100.0)},
    "mu_m": {"index": 8, "range": (25.0, 50.0)},
    "sigma_m": {"index": 9, "range": (0.01, 5.0)},
    "beta_q": {"index": 10, "range": (-5.0, 5.0)},
}

# Only edit this list when you want to change the training labels.
#SELECTED_PARAMETERS = ["alpha_m","delta_m","mu_m","sigma_m"]
#SELECTED_PARAMETERS = ["mu_m","sigma_m"]
#SELECTED_PARAMETERS = ["zp", "delta_m","beta_q"]
SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m","beta_q"]
#SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m","sigma_m","beta_q"]
#SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m","mu_m","beta_q"]
#SELECTED_PARAMETERS = ["zp","alpha_z","beta_z", "alpha_m","delta_m","beta_q"]
#SELECTED_PARAMETERS = ["zp", "alpha_m","delta_m","lambda_peak","mu_m","sigma_m","beta_q"]

PARAMETER_NAMES = list(SELECTED_PARAMETERS)
LABEL_INDICES = [PARAMETER_SPECS[name]["index"] for name in SELECTED_PARAMETERS]
PARAMETER_RANGES = {name: PARAMETER_SPECS[name]["range"] for name in SELECTED_PARAMETERS}


def _sample_key(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("-", 1)[-1]


def _filter_label(label: np.ndarray) -> np.ndarray:
    # Original label order:
    # [0] zp
    # [1] alpha_z
    # [2] beta_z
    # [3] alpha_m
    # [4] m_max
    # [5] delta_m
    # [6] m_min
    # [7] lambda_peak
    # [8] mu_m
    # [9] sigma_m
    # [10] beta_q
    #
    # Current training labels are controlled by SELECTED_PARAMETERS above.
    return np.asarray(label, dtype=np.float32)[LABEL_INDICES]


class GWDatasetV1A(Dataset):
    def __init__(
        self,
        h1_dir: str | Path | None = None,
        l1_dir: str | Path | None = None,
        label_mean: torch.Tensor | None = None,
        label_std: torch.Tensor | None = None,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
        records: list[dict] | None = None,
    ) -> None:
        self.h1_dir = Path(h1_dir) if h1_dir is not None else None
        self.l1_dir = Path(l1_dir) if l1_dir is not None else None
        self.dataset_name = dataset_name
        self.input_scale = 1e23
        self.sample_step = max(1, int(sample_step))

        if records is not None:
            self.records = list(records)
        else:
            if self.h1_dir is None or not self.h1_dir.exists():
                raise FileNotFoundError(f"H1 directory not found: {self.h1_dir}")
            self.records = self.build_records(
                h1_dir=self.h1_dir,
                l1_dir=self.l1_dir,
                sample_step=self.sample_step,
                dataset_name=self.dataset_name,
            )

        if not self.records:
            raise RuntimeError(f"No samples found for {self.dataset_name}")

        if label_mean is not None and label_std is not None:
            self.label_mean = label_mean.float()
            self.label_std = label_std.float()
        else:
            mean_array, std_array = self._calculate_label_stats()
            self.label_mean = torch.tensor(mean_array, dtype=torch.float32)
            self.label_std = torch.tensor(std_array, dtype=torch.float32)

        if feature_mean is not None and feature_std is not None:
            self.feature_mean = feature_mean.float()
            self.feature_std = feature_std.float()
        else:
            mean_array, std_array = self._calculate_feature_stats()
            self.feature_mean = torch.tensor(mean_array, dtype=torch.float32)
            self.feature_std = torch.tensor(std_array, dtype=torch.float32)

        print(f"[{self.dataset_name}] Loaded {len(self.records)} samples (2 channels)")

    @staticmethod
    def build_records(
        h1_dir: str | Path,
        l1_dir: str | Path | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
    ) -> list[dict]:
        h1_dir = Path(h1_dir)
        l1_dir = Path(l1_dir) if l1_dir else None
        h1_files = sorted(path for path in h1_dir.iterdir() if path.suffix == ".npy")[:: max(1, int(sample_step))]
        if l1_dir is None or not l1_dir.exists():
            raise FileNotFoundError(f"L1 directory not found: {l1_dir}")

        l1_map = {_sample_key(path.name): path for path in l1_dir.iterdir() if path.suffix == ".npy"}
        records: list[dict] = []
        for h1_path in tqdm(h1_files, desc=f"[{dataset_name}] Matching H1/L1", unit="file"):
            l1_path = l1_map.get(_sample_key(h1_path.name))
            if l1_path is not None:
                records.append({"h1": h1_path, "l1": l1_path})
        return records

    @staticmethod
    def split_records(
        records: list[dict],
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.0")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(records), generator=generator).tolist()
        test_size = max(1, int(round(len(records) * test_fraction)))
        val_size = max(1, int(round(len(records) * val_fraction)))
        if test_size + val_size >= len(records):
            val_size = max(1, len(records) - test_size - 1)

        test_indices = set(permutation[:test_size])
        val_indices = set(permutation[test_size:test_size + val_size])
        train_records = [record for idx, record in enumerate(records) if idx not in test_indices and idx not in val_indices]
        val_records = [record for idx, record in enumerate(records) if idx in val_indices]
        test_records = [record for idx, record in enumerate(records) if idx in test_indices]
        return train_records, val_records, test_records

    def _load_channel(self, path: Path) -> np.ndarray:
        sample = np.load(path, allow_pickle=True).item()
        data = np.asarray(sample["data"], dtype=np.float32) * self.input_scale
        return data - data.mean(dtype=np.float32)

    def _calculate_label_stats(self) -> tuple[np.ndarray, np.ndarray]:
        labels = []
        for record in tqdm(self.records, desc=f"[{self.dataset_name}] Loading labels", unit="file"):
            sample = np.load(record["h1"], allow_pickle=True).item()
            labels.append(_filter_label(sample["label"]))
        all_labels = np.stack(labels, axis=0)
        mean = all_labels.mean(axis=0)
        std = all_labels.std(axis=0)
        std[std == 0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def _calculate_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        channel_sum = np.zeros(2, dtype=np.float64)
        channel_sq_sum = np.zeros(2, dtype=np.float64)
        total_count = 0
        for record in tqdm(self.records, desc=f"[{self.dataset_name}] Loading feature stats", unit="file"):
            h1 = self._load_channel(record["h1"])
            l1 = self._load_channel(record["l1"])
            features = np.stack([h1, l1], axis=0).astype(np.float64)
            channel_sum += features.sum(axis=1)
            channel_sq_sum += np.square(features).sum(axis=1)
            total_count += features.shape[1]
        mean = channel_sum / total_count
        var = channel_sq_sum / total_count - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        h1_sample = np.load(record["h1"], allow_pickle=True).item()
        h1 = np.asarray(h1_sample["data"], dtype=np.float32) * self.input_scale
        h1 = h1 - h1.mean(dtype=np.float32)
        l1 = self._load_channel(record["l1"])
        features = np.stack([h1, l1], axis=0)
        features = (features - self.feature_mean[:, None].numpy()) / self.feature_std[:, None].numpy()

        label = _filter_label(h1_sample["label"])
        label_tensor = torch.from_numpy(label)
        label_tensor = (label_tensor - self.label_mean) / self.label_std
        return torch.from_numpy(features.astype(np.float32)), label_tensor
