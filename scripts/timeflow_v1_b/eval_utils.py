from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import GWDatasetV1B
from model import GWFlowModelV1B
from model2 import GWFlowModelV1B_Model2


def load_run_config(output_dir: Path) -> dict:
    config_path = output_dir / "run_config_resume.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_model_and_dataset(split: str, checkpoint_path: Path, device: torch.device):
    output_dir = checkpoint_path.parent
    run_config = load_run_config(output_dir)
    split_manifest = json.loads(Path(run_config["split_manifest"]).read_text(encoding="utf-8"))
    records = split_manifest[f"{split}_records"]
    dataset = GWDatasetV1B(
        dataset_name=split,
        records=records,
        selected_parameters=run_config["parameter_names"],
        label_mean=torch.tensor(run_config["label_mean"], dtype=torch.float32),
        label_std=torch.tensor(run_config["label_std"], dtype=torch.float32),
        feature_mean=torch.tensor(run_config["feature_mean"], dtype=torch.float32),
        feature_std=torch.tensor(run_config["feature_std"], dtype=torch.float32),
        sampling_frequency=run_config.get("sampling_frequency", 64),
        nperseg=run_config.get("nperseg", 64),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cls = GWFlowModelV1B_Model2 if run_config.get("model_variant") == "model2" else GWFlowModelV1B
    model = model_cls(
        param_dim=len(run_config["parameter_names"]),
        in_channels=5,
        context_dim=run_config["context_dim"],
        flow_layers=run_config["flow_layers"],
        flow_hidden_features=run_config["flow_hidden"],
        flow_bins=run_config["flow_bins"],
        flow_tail_bound=run_config["flow_tail_bound"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataset, run_config
