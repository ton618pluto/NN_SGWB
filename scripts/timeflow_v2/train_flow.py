import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import GWDatasetV2
    from model import GWFlowModelV2
else:
    from .dataset import GWDatasetV2
    from .model import GWFlowModelV2


# =========================
# 直接在这里改训练配置：
# 1. 数据路径
# 2. 是否使用 L1
# 3. 抽样比例
# 4. 训练超参数
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
# 仓库根目录。
SCRIPTS_ROOT = REPO_ROOT / "scripts"
# `scripts` 目录根路径，下面默认去找处理好的训练/验证数据。

TRAIN_H1_DIR = SCRIPTS_ROOT / "processed_data_train" / "v2" / "training_set0" / "H1_splits"
# 训练集 H1 切片目录，里面应是 `.npy` 文件。
TRAIN_L1_DIR = SCRIPTS_ROOT / "processed_data_train" / "v2" / "training_set0" / "L1_splits"
# 训练集 L1 切片目录，文件名应能和 H1 一一对应。
VAL_H1_DIR = SCRIPTS_ROOT / "processed_data_val" / "v1" / "H1_splits"
# 验证集 H1 切片目录。
VAL_L1_DIR = SCRIPTS_ROOT / "processed_data_val" / "v1" / "L1_splits"
# 验证集 L1 切片目录。

USE_L1 = True
# 是否把 L1 一起作为第二个输入通道；True 为 H1+L1，False 为仅 H1。
TRAIN_SAMPLE_STEP = 1
# 训练集抽样步长；100 表示每隔 100 个 .npy 文件取 1 个样本。
VAL_SAMPLE_STEP = 1
# 验证集抽样步长；1 表示验证集全量使用。

BATCH_SIZE = 32
# 每个 batch 的样本数；显存不够可以改小到 4 或 8。
EPOCHS = 100
# 总训练轮数；若只想检查流程，建议先改成 1~3。
LEARNING_RATE = 5e-5
# 优化器学习率；flow 模型通常用较小学习率更稳定。
WEIGHT_DECAY = 1e-4
# AdamW 的权重衰减系数，用于轻度正则化。
NUM_WORKERS = 0
# DataLoader 的并行进程数；Windows 环境先设 0 最稳。
CONTEXT_DIM = 256
# 时域编码器输出给 flow 的条件特征维度。
FLOW_LAYERS = 6
# flow 中可逆变换层数；更大表达力更强，但训练更慢。
FLOW_HIDDEN = 256
# 每层自回归网络的隐藏层宽度。
SEED = 42
# 随机种子，保证数据抽样和训练过程尽量可复现。

DRY_RUN = False
# 若设为 True，只跑一个 batch 的前向/反向，用来检查流程是否打通。

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
# 输出目录：保存 run_config 和最优模型 checkpoint。
CHECKPOINT_NAME = "best_flow_v2.pt"
# 最优模型保存时使用的文件名。


def set_seed(seed: int) -> None:
    # 固定随机种子，方便复现实验。
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: GWDatasetV2, batch_size: int, shuffle: bool) -> DataLoader:
    # 这里不做复杂封装，只保留当前训练需要的 DataLoader 配置。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def save_run_config(train_dataset: GWDatasetV2) -> None:
    # 保存本次训练配置和标签标准化统计量，便于后续复现实验。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "train_h1_dir": str(TRAIN_H1_DIR),
        "train_l1_dir": str(TRAIN_L1_DIR),
        "val_h1_dir": str(VAL_H1_DIR),
        "val_l1_dir": str(VAL_L1_DIR),
        "use_l1": USE_L1,
        "train_sample_step": TRAIN_SAMPLE_STEP,
        "val_sample_step": VAL_SAMPLE_STEP,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "context_dim": CONTEXT_DIM,
        "flow_layers": FLOW_LAYERS,
        "flow_hidden": FLOW_HIDDEN,
        "seed": SEED,
        "label_mean": train_dataset.label_mean.tolist(),
        "label_std": train_dataset.label_std.tolist(),
        "num_channels": train_dataset.num_channels,
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train_epoch(
    model: GWFlowModelV2,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    # 单个 epoch 的训练逻辑。
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
def evaluate(model: GWFlowModelV2, loader: DataLoader, device: torch.device) -> float:
    # 验证阶段只前向计算，不更新参数。
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc="Val", unit="batch", leave=False)

    for waveforms, labels in progress:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = model(labels, waveforms)

        total_loss += loss.item() * waveforms.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    set_seed(SEED)

    # 自动选择 GPU / CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_l1_dir = TRAIN_L1_DIR if USE_L1 else None
    val_l1_dir = VAL_L1_DIR if USE_L1 else None

    print(f"Device: {device}")
    print(f"Train H1: {TRAIN_H1_DIR}")
    print(f"Train L1: {train_l1_dir}")
    print(f"Val H1: {VAL_H1_DIR}")
    print(f"Val L1: {val_l1_dir}")

    # 构造训练集和验证集；验证集复用训练集的标签标准化统计量。
    train_dataset = GWDatasetV2(
        h1_dir=TRAIN_H1_DIR,
        l1_dir=train_l1_dir,
        sample_step=TRAIN_SAMPLE_STEP,
        dataset_name="train",
    )
    val_dataset = GWDatasetV2(
        h1_dir=VAL_H1_DIR,
        l1_dir=val_l1_dir,
        label_mean=train_dataset.label_mean,
        label_std=train_dataset.label_std,
        sample_step=VAL_SAMPLE_STEP,
        dataset_name="val",
    )

    # DataLoader 负责按 batch 提供双通道时域数据。
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_dataset, BATCH_SIZE, shuffle=False)

    # 模型的输入通道数和参数维度，都从数据集自动推断。
    model = GWFlowModelV2(
        param_dim=train_dataset.label_mean.numel(),
        in_channels=train_dataset.num_channels,
        context_dim=CONTEXT_DIM,
        flow_layers=FLOW_LAYERS,
        flow_hidden_features=FLOW_HIDDEN,
    ).to(device)

    # 使用 AdamW + ReduceLROnPlateau 训练 flow 模型。
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config(train_dataset)
    checkpoint_path = OUTPUT_DIR / CHECKPOINT_NAME

    print(f"Model input channels: {train_dataset.num_channels}")
    print(f"Parameter dimension: {train_dataset.label_mean.numel()}")
    print(f"Checkpoint: {checkpoint_path}")

    start_time = time.time()
    best_val_loss = float("inf")

    if DRY_RUN:
        # dry run 只跑一个 batch，用来快速检查前向和反向是否能打通。
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
        val_loss = evaluate(model, val_loader, device)
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
            # 仅保存验证集表现最好的 checkpoint。
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
                },
                checkpoint_path,
            )
            print("Saved new best checkpoint.")

    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nFinished in {total_minutes:.2f} min. Best val NLL: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
