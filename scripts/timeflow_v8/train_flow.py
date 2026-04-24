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
    from dataset import GWDatasetV8, PARAMETER_NAMES
    from model import GWFlowModelV8
else:
    from .dataset import GWDatasetV8, PARAMETER_NAMES
    from .model import GWFlowModelV8


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
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
CONTEXT_DIM = 256
FLOW_LAYERS = 6
FLOW_HIDDEN = 256
SEED = 42
DRY_RUN = False
AUTO_RESUME = True
GPU_INDEX = 0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BEST_CHECKPOINT_NAME = "best_flow_v8.pt"
LATEST_CHECKPOINT_NAME = "latest_flow_v8.pt"
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


def make_loader(dataset: GWDatasetV8, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
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
        return {"h1": str(record["h1"]), "l1": str(record["l1"]) if record["l1"] is not None else None}

    payload = {
        "train_records": [serialize_record(record) for record in train_records],
        "val_records": [serialize_record(record) for record in val_records],
        "test_records": [serialize_record(record) for record in test_records],
    }
    with (OUTPUT_DIR / SPLIT_MANIFEST_NAME).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_run_config(train_dataset: GWDatasetV8, train_size: int, val_size: int, test_size: int) -> None:
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
        "parameter_names": PARAMETER_NAMES,
        "split_manifest": str(OUTPUT_DIR / SPLIT_MANIFEST_NAME),
        "best_checkpoint": str(OUTPUT_DIR / BEST_CHECKPOINT_NAME),
        "latest_checkpoint": str(OUTPUT_DIR / LATEST_CHECKPOINT_NAME),
    }
    with (OUTPUT_DIR / "run_config_resume.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train_epoch(model: GWFlowModelV8, loader: DataLoader, optimizer: optim.Optimizer, scaler: GradScaler, device: torch.device) -> float:
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
def evaluate(model: GWFlowModelV8, loader: DataLoader, device: torch.device, desc: str) -> float:
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


def build_checkpoint_payload(
    model: GWFlowModelV8,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    train_dataset: GWDatasetV8,
    best_val_loss: float,
    epoch: int,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "label_mean": train_dataset.label_mean,
        "label_std": train_dataset.label_std,
        "num_channels": train_dataset.num_channels,
        "param_dim": train_dataset.label_mean.numel(),
        "best_val_loss": best_val_loss,
        "epoch": epoch,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
    }


def try_resume_training(
    checkpoint_path: Path,
    model: GWFlowModelV8,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    if not AUTO_RESUME or not checkpoint_path.exists():
        return 1, float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    last_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    start_epoch = last_epoch + 1

    print(f"Resume detected: loading {checkpoint_path}")
    print(f"Resume epoch: {last_epoch} -> continue from epoch {start_epoch}")
    print(f"Best val NLL so far: {best_val_loss:.6f}")

    return start_epoch, best_val_loss


def main() -> None:
    set_seed(SEED)
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    train_l1_dir = TRAIN_L1_DIR if USE_L1 else None
    print(f"Device: {device}")
    print(f"Train H1: {TRAIN_H1_DIR}")
    print(f"Train L1: {train_l1_dir}")
    print(f"Training parameters: {PARAMETER_NAMES}")

    all_records = GWDatasetV8.build_records(TRAIN_H1_DIR, train_l1_dir, sample_step=TRAIN_SAMPLE_STEP, dataset_name="full_superimposed")
    train_records, val_records, test_records = GWDatasetV8.split_records(all_records, VAL_FRACTION, TEST_FRACTION, SEED)

    train_dataset = GWDatasetV8(records=train_records, dataset_name="train_superimposed")
    val_dataset = GWDatasetV8(records=val_records, label_mean=train_dataset.label_mean, label_std=train_dataset.label_std, dataset_name="val_superimposed")
    test_dataset = GWDatasetV8(records=test_records, label_mean=train_dataset.label_mean, label_std=train_dataset.label_std, dataset_name="test_superimposed")

    train_loader = make_loader(train_dataset, BATCH_SIZE, True, SEED)
    val_loader = make_loader(val_dataset, BATCH_SIZE, False, SEED + 1)
    test_loader = make_loader(test_dataset, BATCH_SIZE, False, SEED + 2)

    model = GWFlowModelV8(
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
    save_run_config(train_dataset, len(train_records), len(val_records), len(test_records))
    best_checkpoint_path = OUTPUT_DIR / BEST_CHECKPOINT_NAME
    latest_checkpoint_path = OUTPUT_DIR / LATEST_CHECKPOINT_NAME

    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")
    print(f"Test samples: {len(test_records)}")
    print(f"Model input channels: {train_dataset.num_channels}")
    print(f"Parameter dimension: {train_dataset.label_mean.numel()}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")

    start_time = time.time()
    start_epoch, best_val_loss = try_resume_training(
        latest_checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
    )

    if start_epoch > EPOCHS:
        print(f"Training already reached epoch {start_epoch - 1}, no further epochs to run.")
        test_loss = evaluate(model, test_loader, device, desc="Test")
        print(f"Final test NLL with resumed weights: {test_loss:.6f}")
        return

    if DRY_RUN:
        waveforms, labels = next(iter(train_loader))
        loss = model(labels.to(device), waveforms.to(device))
        loss.backward()
        print(f"Dry run OK. Batch shape: {tuple(waveforms.shape)}, loss: {loss.item():.4f}")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate(model, val_loader, device, desc="Val")
        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start
        print(f"train_nll={train_loss:.6f}  val_nll={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_dataset=train_dataset,
                best_val_loss=best_val_loss,
                epoch=epoch,
            )
            torch.save(best_payload, best_checkpoint_path)
            print("Saved new best checkpoint.")

        latest_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_dataset=train_dataset,
            best_val_loss=best_val_loss,
            epoch=epoch,
        )
        torch.save(latest_payload, latest_checkpoint_path)
        print("Saved latest checkpoint.")

    test_loss = evaluate(model, test_loader, device, desc="Test")
    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nFinished in {total_minutes:.2f} min. Best val NLL: {best_val_loss:.6f}")
    print(f"Final test NLL with last-epoch weights: {test_loss:.6f}")


if __name__ == "__main__":
    main()
