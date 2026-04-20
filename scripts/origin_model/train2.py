import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from GWDataset import GWDataset
    from model.GWFlowModel import GWFlowModel
else:
    from .GWDataset import GWDataset
    from .model.GWFlowModel import GWFlowModel


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
OUTPUT_PATH = Path(__file__).resolve().parent / "best_flow_model.pth"


def resolve_data_dir(*relative_paths: tuple[str, ...]) -> Path:
    candidates = []
    for relative_path in relative_paths:
        candidates.append(ROOT_DIR.joinpath(*relative_path))
        candidates.append(SCRIPTS_DIR.joinpath(*relative_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Data directory not found. Checked:\n{searched}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    lr = 1e-6
    epochs = 100

    train_h1_path = resolve_data_dir(
        ("processed_data_train", "v2", "training_set0", "H1_splits"),
    )
    train_l1_path = resolve_data_dir(
        ("processed_data_train", "v2", "training_set0", "L1_splits"),
    )
    val_h1_path = resolve_data_dir(
        ("processed_data_val", "v1", "H1_splits"),
    )
    val_l1_path = resolve_data_dir(
        ("processed_data_val", "v1", "L1_splits"),
    )

    print(f"\n{'=' * 30}\n[初始化阶段] 开始准备归一化流训练环境\n{'=' * 30}")
    print(f"使用设备: {device}")
    print(f"Train H1: {train_h1_path}")
    print(f"Train L1: {train_l1_path}")
    print(f"Val H1: {val_h1_path}")
    print(f"Val L1: {val_l1_path}")

    train_dataset = GWDataset(
        h1_dir=train_h1_path,
        l1_dir=train_l1_path,
        show_progress=True,
        dataset_name="train"
    )
    val_dataset = GWDataset(
        h1_dir=val_h1_path,
        l1_dir=val_l1_path,
        mean=train_dataset.mean,
        std=train_dataset.std,
        show_progress=True,
        dataset_name="val"
    )

    if val_dataset.num_channels != train_dataset.num_channels:
        raise RuntimeError(
            f"Train/val input channels mismatch: {train_dataset.num_channels} vs {val_dataset.num_channels}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GWFlowModel(
        param_dim=train_dataset.mean.numel(),
        context_dim=256,
        in_channels=train_dataset.num_channels,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    print(f"\n{'=' * 30}\n[开始训练任务: Normalizing Flow]\n{'=' * 30}")
    print(f"模型输入通道数: {train_dataset.num_channels}")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        train_nll = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 训练", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = model(labels, images)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_nll += loss.item() * images.size(0)
            train_bar.set_postfix(NLL=f"{loss.item():.4f}")

        avg_train_nll = train_nll / len(train_dataset)

        model.eval()
        val_nll = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 验证", leave=False)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                loss = model(labels, images)
                val_nll += loss.item() * images.size(0)

        avg_val_nll = val_nll / len(val_dataset)
        scheduler.step(avg_val_nll)

        epoch_duration = time.time() - epoch_start_time
        print(f"\n> Epoch {epoch + 1} 完成! 耗时: {epoch_duration:.1f}s")
        print(f"  平均训练 NLL (Loss): {avg_train_nll:.6f}")
        print(f"  平均验证 NLL (Loss): {avg_val_nll:.6f}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved(0) / 1024 ** 3
            print(f"  GPU 显存占用: {mem:.2f} GB")

        if avg_val_nll < best_val_loss:
            best_val_loss = avg_val_nll
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  *** [保存] 发现更低 NLL，模型已更新: {OUTPUT_PATH} ***")

        print("-" * 50)

    total_time = (time.time() - start_time) / 60
    print(f"\n{'=' * 30}\n[任务完成] 总耗时: {total_time:.2f} 分钟\n最佳验证 NLL: {best_val_loss:.8f}\n{'=' * 30}")


if __name__ == "__main__":
    train()
