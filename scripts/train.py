import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

from GWDataset import GWDataset
from model.GWFlowModel import GWFlowModel  # 确保你已经创建了这个流模型类


def train():
    # --- 1. 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128  # 流模型显存占用较高，建议先从 16 开始
    # lr = 5e-5  # 流模型通常需要更小的学习率以保证稳定
    lr = 1e-6  # 流模型通常需要更小的学习率以保证稳定
    epochs = 100

    train_path = './processed_data_train/training_set5/H1_splits'
    val_path = './processed_data_val/H1_splits'

    print(f"\n{'=' * 30}\n[初始化阶段] 开始准备归一化流训练环境\n{'=' * 30}")
    print(f"使用设备: {device}")

    # --- 2. 数据准备 ---
    train_dataset = GWDataset(train_path)
    val_dataset = GWDataset(val_path, mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- 3. 模型定义 ---
    # param_dim=10 (你的物理量个数), context_dim=256 (CNN提取的特征维度)
    model = GWFlowModel(param_dim=10, context_dim=256).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器（可选，但在训练流模型时非常有用）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    print(f"\n{'=' * 30}\n[开始训练任务: Normalizing Flow]\n{'=' * 30}")

    # --- 4. 训练与验证循环 ---
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- 训练阶段 ---
        model.train()
        train_nll = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 训练", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 重要：在流模型中，forward 通常直接返回负对数似然 (NLL)
            # 传参顺序依照你 GWFlowModel 的定义：(theta, x)
            loss = model(labels, images)

            loss.backward()

            # 梯度裁剪（流模型训练容易梯度爆炸，建议加上）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_nll += loss.item() * images.size(0)
            train_bar.set_postfix(NLL=f"{loss.item():.4f}")

        avg_train_nll = train_nll / len(train_dataset)

        # --- 验证阶段 ---
        model.eval()
        val_nll = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] 验证", leave=False)

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_bar):
                images, labels = images.to(device), labels.to(device)

                # 计算验证集的 NLL
                loss = model(labels, images)

                # 打印统计信息
                # if i == 0:  # 只看第一个 Batch 即可
                #     print(f"\n[数据统计] Batch {i}:")
                #     print(f"  Images Mean: {images.mean().item():.2e}") # 使用 .2e 显示科学计数法
                #     print(f"  Images Std:  {images.std().item():.6f}")  # 关键看这个！
                #     print(f"  Labels Mean: {labels.mean().item():.6f}")
                #
                #     # 记录异常但尝试跳过，或者直接 break 进行调试
                #     found_nan = True
                #     # 如果你想定位到具体的样本文件名，可以在 Dataset 里返回文件名，或者在这里打印 i
                #     # 建议初次排查直接跳过，看后续 Batch 是否正常
                #     continue

                val_nll += loss.item() * images.size(0)

        avg_val_nll = val_nll / len(val_dataset)

        # 更新学习率
        scheduler.step(avg_val_nll)

        # --- 结果展示 ---
        epoch_duration = time.time() - epoch_start_time
        print(f"\n> Epoch {epoch + 1} 完成! 耗时: {epoch_duration:.1f}s")
        print(f"  平均的训练 NLL (Loss): {avg_train_nll:.6f}")
        print(f"  平均的验证 NLL (Loss): {avg_val_nll:.6f}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved(0) / 1024 ** 3
            print(f"  GPU 显存占用: {mem:.2f} GB")

        # --- 5. 保存最优模型 ---
        if avg_val_nll < best_val_loss:
            best_val_loss = avg_val_nll
            torch.save(model.state_dict(), 'best_flow_model.pth')
            print(f"  *** [保存] 发现更低 NLL，模型已更新 ***")

        print("-" * 50)

    total_time = (time.time() - start_time) / 60
    print(f"\n{'=' * 30}\n[任务完成] 总耗时: {total_time:.2f} 分钟\n最佳验证 NLL: {best_val_loss:.8f}\n{'=' * 30}")


if __name__ == "__main__":
    train()