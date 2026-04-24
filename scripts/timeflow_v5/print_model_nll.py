from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import GWDatasetV5
    from model import GWFlowModelV5
else:
    from .dataset import GWDatasetV5
    from .model import GWFlowModelV5


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best_flow_v5.pt"
LATEST_CHECKPOINT_PATH = OUTPUT_DIR / "latest_flow_v5.pt"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config_resume.json"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "data_split.json"
BATCH_SIZE = 64
NUM_WORKERS = 0


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_checkpoint_path() -> Path:
    if BEST_CHECKPOINT_PATH.exists():
        return BEST_CHECKPOINT_PATH
    if LATEST_CHECKPOINT_PATH.exists():
        return LATEST_CHECKPOINT_PATH
    raise FileNotFoundError(
        f"No checkpoint found in {OUTPUT_DIR}. "
        f"Expected {BEST_CHECKPOINT_PATH.name} or {LATEST_CHECKPOINT_PATH.name}."
    )


def choose_run_config_path() -> Path:
    if RUN_CONFIG_PATH.exists():
        return RUN_CONFIG_PATH
    raise FileNotFoundError(
        f"No run config found in {OUTPUT_DIR}. "
        f"Expected {RUN_CONFIG_PATH.name}."
    )


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


def load_split_records() -> dict[str, list[dict]]:
    manifest = load_json(SPLIT_MANIFEST_PATH)
    return {
        "train": [_deserialize_record(item) for item in manifest["train_records"]],
        "val": [_deserialize_record(item) for item in manifest["val_records"]],
        "test": [_deserialize_record(item) for item in manifest["test_records"]],
    }


def build_model(checkpoint: dict, run_config: dict, device: torch.device) -> GWFlowModelV5:
    model = GWFlowModelV5(
        param_dim=int(checkpoint["param_dim"]),
        in_channels=int(checkpoint["num_channels"]),
        context_dim=int(run_config["context_dim"]),
        flow_layers=int(run_config["flow_layers"]),
        flow_hidden_features=int(run_config["flow_hidden"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_dataset(records: list[dict], checkpoint: dict, split_name: str) -> GWDatasetV5:
    use_l1 = int(checkpoint["num_channels"]) == 2
    if not use_l1:
        records = [{"h1": record["h1"], "l1": None} for record in records]

    return GWDatasetV5(
        records=records,
        label_mean=checkpoint["label_mean"].cpu(),
        label_std=checkpoint["label_std"].cpu(),
        dataset_name=f"{split_name}_nll",
    )


@torch.inference_mode()
def evaluate_nll(model: GWFlowModelV5, dataset: GWDatasetV5, device: torch.device) -> float:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss = 0.0
    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = model(labels, waveforms)
        total_loss += loss.item() * waveforms.size(0)

    return total_loss / len(dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = choose_checkpoint_path()
    run_config_path = choose_run_config_path()

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run config: {run_config_path}")
    print(f"Split manifest: {SPLIT_MANIFEST_PATH}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_config = load_json(run_config_path)
    split_records = load_split_records()
    model = build_model(checkpoint, run_config, device)

    for split_name in ("train", "val", "test"):
        dataset = build_dataset(split_records[split_name], checkpoint, split_name)
        nll = evaluate_nll(model, dataset, device)
        print(f"{split_name}_nll={nll:.6f}")


if __name__ == "__main__":
    main()
