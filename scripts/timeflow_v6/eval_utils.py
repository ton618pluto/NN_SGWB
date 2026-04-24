from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import GWDatasetV6
    from model import GWFlowModelV6
else:
    from .dataset import GWDatasetV6
    from .model import GWFlowModelV6


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_flow_v6.pt"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config_resume.json"
LEGACY_RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "data_split.json"

PARAMETER_NAMES = [
    "zp",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _infer_l1_path(h1_path: str | Path) -> Path | None:
    h1_path = Path(h1_path)
    if "H1_splits" not in h1_path.parts:
        return None
    parts = list(h1_path.parts)
    parts[parts.index("H1_splits")] = "L1_splits"
    l1_name = h1_path.name
    if l1_name.startswith("H1-"):
        l1_name = "L1-" + l1_name[3:]
    parts[-1] = l1_name
    return Path(*parts)


def _deserialize_record(item: str | dict) -> dict:
    if isinstance(item, dict):
        h1_path = Path(item["h1"])
        l1_value = item.get("l1")
        l1_path = Path(l1_value) if l1_value is not None else _infer_l1_path(h1_path)
        return {"h1": h1_path, "l1": l1_path}
    h1_path = Path(item)
    return {"h1": h1_path, "l1": _infer_l1_path(h1_path)}


def load_records_from_manifest(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = load_json(path)
    return (
        [_deserialize_record(item) for item in manifest["train_records"]],
        [_deserialize_record(item) for item in manifest["val_records"]],
        [_deserialize_record(item) for item in manifest["test_records"]],
    )


def _get_required_config_value(key: str) -> int:
    config_path = RUN_CONFIG_PATH if RUN_CONFIG_PATH.exists() else LEGACY_RUN_CONFIG_PATH
    run_config = load_json(config_path)
    if key not in run_config:
        raise KeyError(f"Missing '{key}' in {config_path}")
    return int(run_config[key])


def build_model(checkpoint: dict, device: torch.device) -> GWFlowModelV6:
    model = GWFlowModelV6(
        param_dim=int(checkpoint["param_dim"]),
        in_channels=int(checkpoint["num_channels"]),
        context_dim=_get_required_config_value("context_dim"),
        flow_layers=_get_required_config_value("flow_layers"),
        flow_hidden_features=_get_required_config_value("flow_hidden"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_split_dataset(checkpoint: dict, split_name: str) -> GWDatasetV6:
    train_records, val_records, test_records = load_records_from_manifest(SPLIT_MANIFEST_PATH)
    records_map = {"train": train_records, "val": val_records, "test": test_records}
    records = records_map[split_name]
    use_l1 = int(checkpoint["num_channels"]) == 2
    if not use_l1:
        records = [{"h1": record["h1"], "l1": None} for record in records]
    return GWDatasetV6(
        records=records,
        label_mean=checkpoint["label_mean"].cpu(),
        label_std=checkpoint["label_std"].cpu(),
        dataset_name=f"{split_name}_eval",
    )


def inverse_label_normalization(values: torch.Tensor | np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    mean_array = mean.detach().cpu().numpy()
    std_array = std.detach().cpu().numpy()
    return values * std_array + mean_array


@torch.inference_mode()
def sample_posterior(model: GWFlowModelV6, waveform: torch.Tensor, num_samples: int, device: torch.device) -> torch.Tensor:
    waveform = waveform.unsqueeze(0).to(device)
    samples = model.sample(waveform, num_samples=num_samples)
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    return samples
