from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


PARAMETER_NAMES = [
    "zp",
    "alpha_m",
    "delta_m",
    "lambda_peak",
    "sigma_m",
    "mu_m",
    "beta_q",
]


def _sample_key(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("-", 1)[-1]


def _filter_label(label: np.ndarray) -> np.ndarray:
    # Original order:
    # [zp, alpha_z, beta_z, alpha_m, m_max, delta_m, m_min, lambda_peak, mu_m, sigma_m, beta_q]
    selected = [0, 3, 5, 7, 9, 8, 10]
    return np.asarray(label, dtype=np.float32)[selected]


class GWDatasetV9(Dataset):
    def __init__(
        self,
        h1_dir: str | Path | None = None,
        l1_dir: str | Path | None = None,
        label_mean: torch.Tensor | None = None,
        label_std: torch.Tensor | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
        records: list[dict] | None = None,
    ) -> None:
        self.h1_dir = Path(h1_dir) if h1_dir is not None else None
        self.l1_dir = Path(l1_dir) if l1_dir else None
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
            raise RuntimeError(f"No .npy files found for {self.dataset_name}")

        if label_mean is not None and label_std is not None:
            self.label_mean = label_mean.float()
            self.label_std = label_std.float()
        else:
            mean_array, std_array = self._calculate_label_stats()
            self.label_mean = torch.tensor(mean_array, dtype=torch.float32)
            self.label_std = torch.tensor(std_array, dtype=torch.float32)

        self.num_channels = 2 if self.records[0]["l1"] is not None else 1

        print(
            f"[{self.dataset_name}] Loaded {len(self.records)} samples "
            f"({self.num_channels} channel{'s' if self.num_channels > 1 else ''})"
        )

    @staticmethod
    def build_records(
        h1_dir: str | Path,
        l1_dir: str | Path | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
    ) -> list[dict]:
        h1_dir = Path(h1_dir)
        l1_dir = Path(l1_dir) if l1_dir else None
        sample_step = max(1, int(sample_step))

        h1_files = sorted(path for path in h1_dir.iterdir() if path.suffix == ".npy")
        h1_files = h1_files[::sample_step]

        if l1_dir is None or not l1_dir.exists():
            return [{"h1": path, "l1": None} for path in h1_files]

        l1_map = {_sample_key(path.name): path for path in l1_dir.iterdir() if path.suffix == ".npy"}
        records = []

        progress = tqdm(h1_files, desc=f"[{dataset_name}] Matching H1/L1", unit="file")
        for h1_path in progress:
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
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.0")
        if len(records) < 3:
            raise ValueError("At least 3 samples are required to split train/val/test")

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(len(records), generator=generator).tolist()

        test_size = max(1, int(round(len(records) * test_fraction)))
        val_size = max(1, int(round(len(records) * val_fraction)))
        if test_size + val_size >= len(records):
            overflow = test_size + val_size - (len(records) - 1)
            if overflow > 0:
                if val_size >= test_size:
                    val_size -= overflow
                else:
                    test_size -= overflow

        test_indices = set(permutation[:test_size])
        val_indices = set(permutation[test_size:test_size + val_size])

        train_records = [record for idx, record in enumerate(records) if idx not in test_indices and idx not in val_indices]
        val_records = [record for idx, record in enumerate(records) if idx in val_indices]
        test_records = [record for idx, record in enumerate(records) if idx in test_indices]
        return train_records, val_records, test_records

    def _calculate_label_stats(self) -> tuple[np.ndarray, np.ndarray]:
        labels = []
        progress = tqdm(self.records, desc=f"[{self.dataset_name}] Loading labels", unit="file")
        for record in progress:
            sample = np.load(record["h1"], allow_pickle=True).item()
            labels.append(_filter_label(sample["label"]))

        all_labels = np.stack(labels, axis=0)
        mean = all_labels.mean(axis=0)
        std = all_labels.std(axis=0)
        std[std == 0] = 1.0
        return mean, std

    def __len__(self) -> int:
        return len(self.records)

    def _load_channel(self, path: Path) -> np.ndarray:
        sample = np.load(path, allow_pickle=True).item()
        data = np.asarray(sample["data"], dtype=np.float32) * self.input_scale
        return data - data.mean(dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        h1_sample = np.load(record["h1"], allow_pickle=True).item()

        label = _filter_label(h1_sample["label"])
        label_tensor = torch.from_numpy(label)
        label_tensor = (label_tensor - self.label_mean) / self.label_std

        channels = [np.asarray(h1_sample["data"], dtype=np.float32) * self.input_scale]
        channels[0] = channels[0] - channels[0].mean(dtype=np.float32)

        if record["l1"] is not None:
            channels.append(self._load_channel(record["l1"]))

        data_tensor = torch.from_numpy(np.stack(channels, axis=0))
        return data_tensor, label_tensor

