from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


PARAMETER_SPECS = {
    "zp": {"index": 0, "range": (1.9, 1.9)},
    "alpha_z": {"index": 1, "range": (2.7, 2.7)},
    "beta_z": {"index": 2, "range": (2.9, 2.9)},
    "alpha_m": {"index": 3, "range": (-3.0, 7.0)},
    "m_max": {"index": 4, "range": (100.0, 100.0)},
    "delta_m": {"index": 5, "range": (0.0, 10.0)},
    "m_min": {"index": 6, "range": (2.0, 2.0)},
    "lambda_peak": {"index": 7, "range": (0.04, 0.04)},
    "mu_m": {"index": 8, "range": (25.0, 100.0)},
    "sigma_m": {"index": 9, "range": (0.01, 5.0)},
    "beta_q": {"index": 10, "range": (1.08, 1.08)},
}

# Only edit this list when you want to change the training labels.
# SELECTED_PARAMETERS = ["zp", "alpha_m"]
SELECTED_PARAMETERS = ["alpha_m", "delta_m", "mu_m", "sigma_m"]

def resolve_parameter_names(selected_parameters: list[str] | tuple[str, ...] | None = None) -> list[str]:
    names = list(selected_parameters) if selected_parameters is not None else list(SELECTED_PARAMETERS)
    unknown = [name for name in names if name not in PARAMETER_SPECS]
    if unknown:
        raise KeyError(f"Unknown parameter names: {unknown}")
    return names


def resolve_label_indices(selected_parameters: list[str] | tuple[str, ...] | None = None) -> list[int]:
    return [PARAMETER_SPECS[name]["index"] for name in resolve_parameter_names(selected_parameters)]


def resolve_parameter_ranges(selected_parameters: list[str] | tuple[str, ...] | None = None) -> dict[str, tuple[float, float] | None]:
    names = resolve_parameter_names(selected_parameters)
    return {name: PARAMETER_SPECS[name]["range"] for name in names}


PARAMETER_NAMES = resolve_parameter_names()
LABEL_INDICES = resolve_label_indices()
PARAMETER_RANGES = resolve_parameter_ranges()

DATA_VERSION_RANGE_OVERRIDES = {}


def get_parameter_ranges(
    data_version: str | None = None,
    parameter_names: list[str] | None = None,
) -> dict[str, tuple[float, float] | None]:
    ranges = {name: spec["range"] for name, spec in PARAMETER_SPECS.items()}
    if data_version in DATA_VERSION_RANGE_OVERRIDES:
        ranges.update(DATA_VERSION_RANGE_OVERRIDES[data_version])
    if parameter_names is None:
        return ranges
    return {name: ranges[name] for name in parameter_names}


def _sample_key(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem.split("-", 1)[-1]


def _filter_label(label: np.ndarray, label_indices: list[int] | None = None) -> np.ndarray:
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
    indices = LABEL_INDICES if label_indices is None else label_indices
    return np.asarray(label, dtype=np.float32)[indices]


class GWDatasetV1B(Dataset):
    def __init__(
        self,
        feature_dir: str | Path | None = None,
        label_mean: torch.Tensor | None = None,
        label_std: torch.Tensor | None = None,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
        records: list[dict] | None = None,
        selected_parameters: list[str] | tuple[str, ...] | None = None,
        sampling_frequency: int = 64,
        nperseg: int = 64,
        feature_stats_cache_path: str | Path | None = None,
    ) -> None:
        self.feature_dir = Path(feature_dir) if feature_dir is not None else None
        self.dataset_name = dataset_name
        self.sample_step = max(1, int(sample_step))
        self.fs = int(sampling_frequency)
        self.nperseg = int(nperseg)
        self.eps = 1e-12
        self.feature_stats_cache_path = Path(feature_stats_cache_path) if feature_stats_cache_path is not None else None
        self.parameter_names = resolve_parameter_names(selected_parameters)
        self.label_indices = resolve_label_indices(self.parameter_names)

        if records is not None:
            self.records = list(records)
        else:
            if self.feature_dir is None or not self.feature_dir.exists():
                raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")
            self.records = self.build_records(
                feature_dir=self.feature_dir,
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
            mean_array, std_array = self._load_or_calculate_feature_stats()
            self.feature_mean = torch.tensor(mean_array, dtype=torch.float32)
            self.feature_std = torch.tensor(std_array, dtype=torch.float32)

        print(f"[{self.dataset_name}] Loaded {len(self.records)} samples (5-channel frequency features)")

    @staticmethod
    def build_records(
        feature_dir: str | Path,
        sample_step: int = 1,
        dataset_name: str = "dataset",
    ) -> list[dict]:
        feature_dir = Path(feature_dir)
        feature_files = sorted(
            (path for path in feature_dir.iterdir() if path.suffix == ".npy"),
            key=lambda path: path.name,
        )[:: max(1, int(sample_step))]
        records: list[dict] = []
        for feature_path in tqdm(feature_files, desc=f"[{dataset_name}] Loading feature file list", unit="file"):
            records.append({"feat": feature_path})
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

    def _load_feature_sample(self, path: Path) -> dict:
        return np.load(path, allow_pickle=True).item()

    def _calculate_label_stats(self) -> tuple[np.ndarray, np.ndarray]:
        labels = []
        for record in tqdm(self.records, desc=f"[{self.dataset_name}] Loading labels", unit="file"):
            sample = self._load_feature_sample(record["feat"])
            labels.append(_filter_label(sample["label"], self.label_indices))
        all_labels = np.stack(labels, axis=0)
        mean = all_labels.mean(axis=0)
        std = all_labels.std(axis=0)
        std[std == 0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def _calculate_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        channel_sum = np.zeros(5, dtype=np.float64)
        channel_sq_sum = np.zeros(5, dtype=np.float64)
        total_bins = 0
        for record in tqdm(self.records, desc=f"[{self.dataset_name}] Loading feature stats", unit="file"):
            sample = self._load_feature_sample(record["feat"])
            features = np.asarray(sample["data"], dtype=np.float32).astype(np.float64)
            channel_sum += features.sum(axis=1)
            channel_sq_sum += np.square(features).sum(axis=1)
            total_bins += features.shape[1]
        mean = channel_sum / total_bins
        var = channel_sq_sum / total_bins - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)

    def _load_or_calculate_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_stats_cache_path is not None and self.feature_stats_cache_path.exists():
            cache = np.load(self.feature_stats_cache_path)
            cached_fs = int(np.asarray(cache["sampling_frequency"]).item())
            cached_nperseg = int(np.asarray(cache["nperseg"]).item())
            if cached_fs != self.fs or cached_nperseg != self.nperseg:
                raise ValueError(
                    "Cached feature stats were computed with a different "
                    f"(fs, nperseg)=({cached_fs}, {cached_nperseg}) than the current "
                    f"configuration ({self.fs}, {self.nperseg})."
                )
            mean = np.asarray(cache["feature_mean"], dtype=np.float32)
            std = np.asarray(cache["feature_std"], dtype=np.float32)
            print(f"[{self.dataset_name}] Loaded cached feature stats: {self.feature_stats_cache_path}")
            return mean, std

        mean, std = self._calculate_feature_stats()

        if self.feature_stats_cache_path is not None:
            self.feature_stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.feature_stats_cache_path,
                feature_mean=mean.astype(np.float32),
                feature_std=std.astype(np.float32),
                sampling_frequency=np.int64(self.fs),
                nperseg=np.int64(self.nperseg),
            )
            print(f"[{self.dataset_name}] Saved cached feature stats: {self.feature_stats_cache_path}")

        return mean, std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        sample = self._load_feature_sample(record["feat"])
        features = np.asarray(sample["data"], dtype=np.float32)
        features = (features - self.feature_mean[:, None].numpy()) / self.feature_std[:, None].numpy()

        label = _filter_label(sample["label"], self.label_indices)
        label_tensor = torch.from_numpy(label)
        label_tensor = (label_tensor - self.label_mean) / self.label_std
        return torch.from_numpy(features.astype(np.float32)), label_tensor
