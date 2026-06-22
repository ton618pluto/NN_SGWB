from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GWDatasetV1A, PARAMETER_NAMES
from model import GWFlowModelV1A


DATA_VERSION = "v8"
USE_QUIET_LOGGING = False
TRAIN_SAMPLE_STEP = 1
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
#WEIGHT_DECAY = 1e-6
NUM_WORKERS = 0
CONTEXT_DIM = 256
FLOW_LAYERS = 8
FLOW_HIDDEN = 256
FLOW_BINS = 8
FLOW_TAIL_BOUND = 3.0
SEED = 42
AUTO_RESUME = True
GPU_INDEX = 1
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(__file__).resolve().parent


def build_paths() -> tuple[Path, Path, Path]:
    base = SCRIPTS_DIR / "processed_data_superimposed" / DATA_VERSION
    return base / "H1_splits", base / "L1_splits", VERSION_DIR / f"outputs_dataset_{DATA_VERSION}"


def serialize_records(records: list[dict]) -> list[dict]:
    return [{"h1": str(record["h1"]), "l1": str(record["l1"])} for record in records]


def ensure_history_header(path: Path) -> None:
    if not path.exists():
        path.write_text("epoch,train_nll,val_nll\n", encoding="utf-8")


def append_history(path: Path, epoch: int, train_nll: float, val_nll: float) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{epoch},{train_nll:.8f},{val_nll:.8f}\n")


def evaluate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        iterator = loader if USE_QUIET_LOGGING else tqdm(loader, desc="Val", leave=False)
        for data, label in iterator:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            loss = -model.log_prob(label, data).mean()
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def try_resume_training(model, optimizer, scheduler, scaler, latest_path: Path, device: torch.device):
    if not AUTO_RESUME or not latest_path.exists():
        return 1, float("inf")
    checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"Resumed from {latest_path} at epoch {checkpoint['epoch']}.")
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_nll"])


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")

    h1_dir, l1_dir, output_dir = build_paths()
    if not h1_dir.exists():
        raise FileNotFoundError(f"H1 directory not found: {h1_dir}")
    if not l1_dir.exists():
        raise FileNotFoundError(f"L1 directory not found: {l1_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset version: {DATA_VERSION}")
    print(f"H1 dir: {h1_dir}")
    print(f"L1 dir: {l1_dir}")
    print(f"Output dir: {output_dir}")
    print("Training parameters:")
    for index, name in enumerate(PARAMETER_NAMES, start=1):
        print(f"  {index}. {name}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")

    all_records = GWDatasetV1A.build_records(h1_dir=h1_dir, l1_dir=l1_dir, sample_step=TRAIN_SAMPLE_STEP, dataset_name="all")
    train_records, val_records, test_records = GWDatasetV1A.split_records(all_records, VAL_FRACTION, TEST_FRACTION, SEED)

    split_manifest_path = output_dir / "data_split_v1_a.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "data_version": DATA_VERSION,
                "train_records": serialize_records(train_records),
                "val_records": serialize_records(val_records),
                "test_records": serialize_records(test_records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = GWDatasetV1A(records=train_records, dataset_name="train")
    val_dataset = GWDatasetV1A(
        records=val_records,
        dataset_name="val",
        label_mean=train_dataset.label_mean,
        label_std=train_dataset.label_std,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = GWFlowModelV1A(
        param_dim=len(PARAMETER_NAMES),
        in_channels=2,
        context_dim=CONTEXT_DIM,
        flow_layers=FLOW_LAYERS,
        flow_hidden_features=FLOW_HIDDEN,
        flow_bins=FLOW_BINS,
        flow_tail_bound=FLOW_TAIL_BOUND,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_path = output_dir / "best_flow_v1_a.pt"
    latest_path = output_dir / "latest_flow_v1_a.pt"
    history_path = output_dir / "epoch_nll_history_v1_a.txt"
    ensure_history_header(history_path)

    start_epoch, best_val_nll = try_resume_training(model, optimizer, scheduler, scaler, latest_path, device)
    run_config = {
        "data_version": DATA_VERSION,
        "parameter_names": PARAMETER_NAMES,
        "label_mean": train_dataset.label_mean.tolist(),
        "label_std": train_dataset.label_std.tolist(),
        "feature_mean": train_dataset.feature_mean.tolist(),
        "feature_std": train_dataset.feature_std.tolist(),
        "context_dim": CONTEXT_DIM,
        "flow_layers": FLOW_LAYERS,
        "flow_hidden": FLOW_HIDDEN,
        "flow_bins": FLOW_BINS,
        "flow_tail_bound": FLOW_TAIL_BOUND,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "split_manifest": str(split_manifest_path),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    (output_dir / "run_config_resume.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        epoch_start = time.time()
        iterator = train_loader if USE_QUIET_LOGGING else tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for data, label in iterator:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = -model.log_prob(label, data).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not USE_QUIET_LOGGING and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(train_nll=f"{loss.item():.4f}")
            batch_size = data.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_nll = total_train_loss / max(total_train_samples, 1)
        val_nll = evaluate_epoch(model, val_loader, device)
        scheduler.step(val_nll)
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_nll={train_nll:.6f} val_nll={val_nll:.6f} lr={lr:.2e} time={epoch_time:.1f}s")
        append_history(history_path, epoch, train_nll, val_nll)

        checkpoint = {
            "epoch": epoch,
            "best_val_nll": min(best_val_nll, val_nll),
            "num_channels": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "label_mean": train_dataset.label_mean,
            "label_std": train_dataset.label_std,
            "feature_mean": train_dataset.feature_mean,
            "feature_std": train_dataset.feature_std,
        }
        torch.save(checkpoint, latest_path)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(checkpoint, best_path)
            print("Saved new best checkpoint.")


if __name__ == "__main__":
    main()
