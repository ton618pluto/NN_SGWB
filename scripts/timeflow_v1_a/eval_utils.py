from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from dataset import GWDatasetV1A
from model import GWFlowModelV1A
from model2 import GWFlowModelV1A_Model2


def load_run_config(output_dir: Path) -> dict:
    config_path = output_dir / "run_config_resume.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_split_manifest_path(run_config: dict, output_dir: Path) -> Path:
    configured_path = Path(run_config["split_manifest"])
    if configured_path.exists():
        return configured_path
    fallback_path = output_dir / "data_split_v1_a.json"
    if fallback_path.exists():
        return fallback_path
    return configured_path


def resolve_record_path(path_str: str) -> str:
    configured_path = Path(path_str)
    if configured_path.exists():
        return str(configured_path)

    parts = configured_path.parts
    if "processed_data_train" in parts:
        anchor = parts.index("processed_data_train")
        repo_scripts_dir = Path(__file__).resolve().parent.parent
        fallback_path = repo_scripts_dir.joinpath(*parts[anchor:])
        if fallback_path.exists():
            return str(fallback_path)
    return path_str


def resolve_records(records: list[dict]) -> list[dict]:
    resolved = []
    for record in records:
        updated = dict(record)
        if "h1" in updated:
            updated["h1"] = resolve_record_path(updated["h1"])
        if "l1" in updated:
            updated["l1"] = resolve_record_path(updated["l1"])
        resolved.append(updated)
    return resolved


def load_model_and_dataset(split: str, checkpoint_path: Path, device: torch.device):
    output_dir = checkpoint_path.parent
    run_config = load_run_config(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split_manifest_path = resolve_split_manifest_path(run_config, output_dir)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    records = resolve_records(split_manifest[f"{split}_records"])
    dataset_kwargs = dict(
        dataset_name=split,
        records=records,
        label_mean=torch.tensor(run_config["label_mean"], dtype=torch.float32),
        label_std=torch.tensor(run_config["label_std"], dtype=torch.float32),
        feature_mean=torch.tensor(run_config["feature_mean"], dtype=torch.float32),
        feature_std=torch.tensor(run_config["feature_std"], dtype=torch.float32),
    )
    if "selected_parameters" in inspect.signature(GWDatasetV1A.__init__).parameters:
        dataset_kwargs["selected_parameters"] = list(run_config["parameter_names"])
    dataset = GWDatasetV1A(**dataset_kwargs)
    model_variant = run_config.get("model_variant", "model1")
    model_cls = GWFlowModelV1A_Model2 if model_variant == "model2" else GWFlowModelV1A
    model = model_cls(
        param_dim=len(run_config["parameter_names"]),
        in_channels=2,
        context_dim=run_config["context_dim"],
        flow_layers=run_config["flow_layers"],
        flow_hidden_features=run_config["flow_hidden"],
        flow_bins=run_config["flow_bins"],
        flow_tail_bound=run_config["flow_tail_bound"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataset, run_config
