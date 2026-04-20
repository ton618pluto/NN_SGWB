from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import GWDatasetV5
    from model import GWFlowModelV5
else:
    from .dataset import GWDatasetV5
    from .model import GWFlowModelV5


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TRAIN_H1_DIR = SCRIPTS_ROOT / "processed_data_superimposed" / "v0" / "H1_splits"
TRAIN_L1_DIR = SCRIPTS_ROOT / "processed_data_superimposed" / "v0" / "L1_splits"
USE_L1 = True
TRAIN_SAMPLE_STEP = 1
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
CONTEXT_DIM = 256
FLOW_LAYERS = 6
FLOW_HIDDEN = 256
SEED = 42
DRY_RUN = False
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_NAME = "best_flow_v5.pt"
SPLIT_MANIFEST_NAME = "data_split.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(dataset: GWDatasetV5, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        generator=generator,
    )


def save_split_manifest(train_records: list[dict], val_records: list[dict], test_records: list[dict]) -> None:
    def serialize_record(record: dict) -> dict:
        return {
            "h1": str(record["h1"]),
            "l1": str(record["l1"]) if record["l1"] is not None else None,
        }

    payload = {
        "train_records": [serialize_record(record) for record in train_records],
        "val_records": [serialize_record(record) for record in val_records],
        "test_records": [serialize_record(record) for record in test_records],
    }
    with (OUTPUT_DIR / SPLIT_MANIFEST_NAME).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_run_config(train_dataset: GWDatasetV5, train_size: int, val_size: int, test_size: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "train_h1_dir": str(TRAIN_H1_DIR),
        "train_l1_dir": str(TRAIN_L1_DIR),
        "use_l1": USE_L1,
        "train_sample_step": TRAIN_SAMPLE_STEP,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "context_dim": CONTEXT_DIM,
        "flow_layers": FLOW_LAYERS,
        "flow_hidden": FLOW_HIDDEN,
        "seed": SEED,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "label_mean": train_dataset.label_mean.tolist(),
        "label_std": train_dataset.label_std.tolist(),
        "num_channels": train_dataset.num_channels,
        "split_manifest": str(OUTPUT_DIR / SPLIT_MANIFEST_NAME),
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train_epoch(
    model: GWFlowModelV5,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="Train", unit="batch")

    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = model(labels, waveforms)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * waveforms.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: GWFlowModelV5, loader: DataLoader, device: torch.device, desc: str) -> float:
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc=desc, unit="batch", leave=False)

    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = model(labels, waveforms)

        total_loss += loss.item() * waveforms.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_l1_dir = TRAIN_L1_DIR if USE_L1 else None

    print(f"Device: {device}")
    print(f"Train H1: {TRAIN_H1_DIR}")
    print(f"Train L1: {train_l1_dir}")
    print(f"Validation split: {VAL_FRACTION:.0%}")
    print(f"Test split: {TEST_FRACTION:.0%}")

    all_records = GWDatasetV5.build_records(
        h1_dir=TRAIN_H1_DIR,
        l1_dir=train_l1_dir,
        sample_step=TRAIN_SAMPLE_STEP,
        dataset_name="full_superimposed",
    )
    train_records, val_records, test_records = GWDatasetV5.split_records(
        records=all_records,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=SEED,
    )

    train_dataset = GWDatasetV5(records=train_records, dataset_name="train_superimposed")
    val_dataset = GWDatasetV5(
        records=val_records,
        label_mean=train_dataset.label_mean,
        label_std=train_dataset.label_std,
        dataset_name="val_superimposed",
    )
    test_dataset = GWDatasetV5(
        records=test_records,
        label_mean=train_dataset.label_mean,
        label_std=train_dataset.label_std,
        dataset_name="test_superimposed",
    )

    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=SEED)
    val_loader = make_loader(val_dataset, BATCH_SIZE, shuffle=False, seed=SEED + 1)
    test_loader = make_loader(test_dataset, BATCH_SIZE, shuffle=False, seed=SEED + 2)

    model = GWFlowModelV5(
        param_dim=train_dataset.label_mean.numel(),
        in_channels=train_dataset.num_channels,
        context_dim=CONTEXT_DIM,
        flow_layers=FLOW_LAYERS,
        flow_hidden_features=FLOW_HIDDEN,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split_manifest(train_records, val_records, test_records)
    save_run_config(
        train_dataset,
        train_size=len(train_records),
        val_size=len(val_records),
        test_size=len(test_records),
    )
    checkpoint_path = OUTPUT_DIR / CHECKPOINT_NAME

    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")
    print(f"Test samples: {len(test_records)}")
    print(f"Model input channels: {train_dataset.num_channels}")
    print(f"Parameter dimension: {train_dataset.label_mean.numel()}")
    print(f"Checkpoint: {checkpoint_path}")

    start_time = time.time()
    best_val_loss = float("inf")

    if DRY_RUN:
        waveforms, labels = next(iter(train_loader))
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        loss = model(labels, waveforms)
        loss.backward()
        print(f"Dry run OK. Batch shape: {tuple(waveforms.shape)}, loss: {loss.item():.4f}")
        return

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate(model, val_loader, device, desc="Val")
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"train_nll={train_loss:.6f}  "
            f"val_nll={val_loss:.6f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            print(f"gpu_reserved={reserved:.2f} GB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label_mean": train_dataset.label_mean,
                    "label_std": train_dataset.label_std,
                    "num_channels": train_dataset.num_channels,
                    "param_dim": train_dataset.label_mean.numel(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "seed": SEED,
                    "val_fraction": VAL_FRACTION,
                    "test_fraction": TEST_FRACTION,
                },
                checkpoint_path,
            )
            print("Saved new best checkpoint.")

    test_loss = evaluate(model, test_loader, device, desc="Test")
    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nFinished in {total_minutes:.2f} min. Best val NLL: {best_val_loss:.6f}")
    print(f"Final test NLL with last-epoch weights: {test_loss:.6f}")


if __name__ == "__main__":
    main()
