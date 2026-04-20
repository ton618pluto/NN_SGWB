# `timeflow_v2`

这套代码保留“原始时域输入”的前提，但训练入口已经改成更直接的版本：

- 用更深的 `1D` 残差编码器替代 `flatten + 大全连接`
- 支持 `H1` 单通道或 `H1+L1` 双通道
- 继续使用 conditional normalizing flow 做联合参数后验
- 训练路径和超参数直接写在 `train_flow.py` 顶部
- 依赖 `nflows`

## 直接运行

在仓库根目录执行：

```powershell
python .\scripts\timeflow_v2\train_flow.py
```

## 需要改哪里

直接改 `scripts/timeflow_v2/train_flow.py` 顶部这些常量：

- `TRAIN_H1_DIR`
- `TRAIN_L1_DIR`
- `VAL_H1_DIR`
- `VAL_L1_DIR`
- `USE_L1`
- `BATCH_SIZE`
- `EPOCHS`

如果你只有 `H1`，把 `USE_L1 = False`。
