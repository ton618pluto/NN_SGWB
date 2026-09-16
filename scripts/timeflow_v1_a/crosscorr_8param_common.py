from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import csd
from torch import nn


PARAMETER_NAMES = [
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "delta_m",
    "mu_m",
    "sigma_m",
    "beta_q",
]

PARAMETER_INDICES = np.asarray([0, 1, 2, 3, 5, 8, 9, 10], dtype=np.int64)

PARAMETER_RANGES = {
    "zp": (0.85, 2.85),
    "alpha_z": (1.0, 3.0),
    "beta_z": (1.0, 5.0),
    "alpha_m": (-3.0, 7.0),
    "delta_m": (0.0, 10.0),
    "mu_m": (25.0, 50.0),
    "sigma_m": (0.01, 5.0),
    "beta_q": (-5.0, 5.0),
}

SAMPLE_RATE = 512.0
FFT_LENGTH = 32.0
OVERLAP = 16.0
WINDOW = "hann"
FREQ_MIN = 20.0
FREQ_MAX = 172.0
NUM_COARSE_BINS = 128

POP_PATTERN = re.compile(r"(pop\d+)")


def sample_key(path: Path) -> str:
    name = path.stem
    for prefix in ("H1-", "L1-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def pop_key(path: Path) -> str:
    match = POP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract population id from {path.name}")
    return match.group(1)


def match_npy_pairs(h1_dir: Path, l1_dir: Path) -> list[tuple[Path, Path]]:
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 directory not found: {l1_dir}")

    l1_map = {sample_key(path): path for path in l1_dir.glob("*.npy")}
    pairs = []
    for h1_path in sorted(h1_dir.glob("*.npy")):
        l1_path = l1_map.get(sample_key(h1_path))
        if l1_path is not None:
            pairs.append((h1_path, l1_path))
    if not pairs:
        raise RuntimeError(f"No matched H1/L1 NPY files under {h1_dir.parent}")
    return pairs


def split_pairs_by_pop(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for pair in pairs:
        grouped.setdefault(pop_key(pair[0]), []).append(pair)

    pop_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(pop_ids))
    test_size = max(1, int(round(len(pop_ids) * test_fraction)))
    val_size = max(1, int(round(len(pop_ids) * val_fraction)))
    test_ids = {pop_ids[index] for index in permutation[:test_size]}
    val_ids = {pop_ids[index] for index in permutation[test_size:test_size + val_size]}

    train_pairs = [pair for pop_id in pop_ids if pop_id not in test_ids | val_ids for pair in grouped[pop_id]]
    val_pairs = [pair for pop_id in pop_ids if pop_id in val_ids for pair in grouped[pop_id]]
    test_pairs = [pair for pop_id in pop_ids if pop_id in test_ids for pair in grouped[pop_id]]
    return train_pairs, val_pairs, test_pairs


def load_npy_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sample = np.load(path, allow_pickle=True).item()
    data = np.asarray(sample["data"], dtype=np.float64)
    full_label = np.asarray(sample["label"], dtype=np.float64)
    if full_label.size <= int(PARAMETER_INDICES.max()):
        raise ValueError(f"Label in {path} has only {full_label.size} entries")
    return data, full_label[PARAMETER_INDICES]


def coarse_cross_spectrum(
    h1: np.ndarray,
    l1: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    if h1.shape != l1.shape:
        raise ValueError(f"H1/L1 shape mismatch: {h1.shape} != {l1.shape}")
    nperseg = int(round(FFT_LENGTH * sample_rate))
    noverlap = int(round(OVERLAP * sample_rate))
    if h1.size < nperseg:
        raise ValueError(f"Need at least {nperseg} samples, got {h1.size}")

    freqs, cross = csd(
        h1,
        l1,
        fs=sample_rate,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    mask = (freqs >= FREQ_MIN) & (freqs <= min(FREQ_MAX, 0.49 * sample_rate))
    freqs = freqs[mask]
    cross_real = np.real(cross[mask])
    groups = np.array_split(np.arange(freqs.size), NUM_COARSE_BINS)
    coarse_freqs = np.asarray([freqs[group].mean() for group in groups], dtype=np.float64)
    coarse_cross = np.asarray([cross_real[group].mean() for group in groups], dtype=np.float64)
    return coarse_freqs, coarse_cross


class CrossCorrEmulator(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = NUM_COARSE_BINS, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        return self.network(parameters)

