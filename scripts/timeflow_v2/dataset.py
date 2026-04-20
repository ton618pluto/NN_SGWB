from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _sample_key(file_name: str) -> str:
    # H1/L1 文件名前缀不同，这里只保留公共部分用于配对。
    stem = Path(file_name).stem
    return stem.split("-", 1)[-1]


def _filter_label(label: np.ndarray) -> np.ndarray:
    # 按你当前的数据定义，丢掉原始标签中的第 3 个分量，保留 10 维目标。
    return np.concatenate([label[:2], label[3:]]).astype(np.float32, copy=False)


class GWDatasetV2(Dataset):
    def __init__(
        self,
        h1_dir: str | Path,
        l1_dir: str | Path | None = None,
        label_mean: torch.Tensor | None = None,
        label_std: torch.Tensor | None = None,
        sample_step: int = 1,
        dataset_name: str = "dataset",
    ) -> None:
        # h1_dir / l1_dir：切分后的 .npy 数据目录。
        self.h1_dir = Path(h1_dir)
        self.l1_dir = Path(l1_dir) if l1_dir else None
        self.dataset_name = dataset_name
        # 原始 strain 数值很小，先放大到更适合神经网络训练的尺度。
        self.input_scale = 1e23
        # sample_step=100 表示每 100 个文件取 1 个，方便快速检查流程。
        self.sample_step = max(1, int(sample_step))

        if not self.h1_dir.exists():
            raise FileNotFoundError(f"H1 directory not found: {self.h1_dir}")

        # records 保存每个样本对应的 H1/L1 文件路径。
        self.records = self._build_records()
        if not self.records:
            raise RuntimeError(f"No .npy files found for {self.dataset_name}")

        if label_mean is not None and label_std is not None:
            # 验证集直接复用训练集统计量，避免数据泄漏。
            self.label_mean = label_mean.float()
            self.label_std = label_std.float()
        else:
            # 训练集首次构造时统计标签均值和标准差，供标准化使用。
            mean_array, std_array = self._calculate_label_stats()
            self.label_mean = torch.tensor(mean_array, dtype=torch.float32)
            self.label_std = torch.tensor(std_array, dtype=torch.float32)

        # 如果成功匹配到 L1，就按双通道输入；否则退化成单通道。
        self.num_channels = 2 if self.records[0]["l1"] is not None else 1

        print(
            f"[{self.dataset_name}] Loaded {len(self.records)} samples "
            f"({self.num_channels} channel{'s' if self.num_channels > 1 else ''})"
        )

    def _build_records(self) -> list[dict]:
        # 先按文件名排序，保证抽样和训练过程可复现。
        h1_files = sorted(path for path in self.h1_dir.iterdir() if path.suffix == ".npy")
        h1_files = h1_files[::self.sample_step]

        if self.l1_dir is None or not self.l1_dir.exists():
            # 只有 H1 时，直接返回单通道记录。
            return [{"h1": path, "l1": None} for path in h1_files]

        # 把 L1 做成字典，后面按 key 快速匹配到对应样本。
        l1_map = {_sample_key(path.name): path for path in self.l1_dir.iterdir() if path.suffix == ".npy"}
        records = []

        progress = tqdm(h1_files, desc=f"[{self.dataset_name}] Matching H1/L1", unit="file")
        for h1_path in progress:
            key = _sample_key(h1_path.name)
            l1_path = l1_map.get(key)
            if l1_path is not None:
                records.append({"h1": h1_path, "l1": l1_path})

        return records

    def _calculate_label_stats(self) -> tuple[np.ndarray, np.ndarray]:
        # 扫一遍训练集标签，计算标准化所需的均值和标准差。
        labels = []
        progress = tqdm(self.records, desc=f"[{self.dataset_name}] Loading labels", unit="file")
        for record in progress:
            sample = np.load(record["h1"], allow_pickle=True).item()
            label = np.asarray(sample["label"], dtype=np.float32)
            labels.append(_filter_label(label))

        all_labels = np.stack(labels, axis=0)
        mean = all_labels.mean(axis=0)
        std = all_labels.std(axis=0)
        std[std == 0] = 1.0
        return mean, std

    def __len__(self) -> int:
        return len(self.records)

    def _load_channel(self, path: Path) -> np.ndarray:
        # 每个通道都做相同预处理：缩放 + 去均值。
        sample = np.load(path, allow_pickle=True).item()
        data = np.asarray(sample["data"], dtype=np.float32) * self.input_scale
        return data - data.mean(dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 读取当前样本的 H1 数据与标签。
        record = self.records[index]
        h1_sample = np.load(record["h1"], allow_pickle=True).item()
        label = np.asarray(h1_sample["label"], dtype=np.float32)
        label = _filter_label(label)
        label_tensor = torch.from_numpy(label)
        # 标签统一做 z-score 标准化，便于 flow 建模。
        label_tensor = (label_tensor - self.label_mean) / self.label_std

        # 第一个通道始终是 H1。
        channels = [np.asarray(h1_sample["data"], dtype=np.float32) * self.input_scale]
        channels[0] = channels[0] - channels[0].mean(dtype=np.float32)

        if record["l1"] is not None:
            # 如果有 L1，则拼成双通道输入：[2, 524288]。
            l1_data = self._load_channel(record["l1"])
            channels.append(l1_data)

        data_tensor = torch.from_numpy(np.stack(channels, axis=0))
        return data_tensor, label_tensor
