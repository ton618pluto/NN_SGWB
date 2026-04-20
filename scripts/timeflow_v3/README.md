# `timeflow_v3`

`timeflow_v3` 基于 `timeflow_v2` 的结构，但训练流程改成了只使用原训练集，并从中按固定随机种子切分出 `20%` 作为验证集。

主要变化：
- 默认数据路径改为 `scripts/processed_data_train/v3/training_set0`
- 不再依赖单独的 `processed_data_val`
- 训练/验证划分由固定 `SEED` 控制，可复现
- 训练时会把本次切分结果保存到 `outputs/data_split.json`

运行方式：

```powershell
python .\scripts\timeflow_v3\train_flow.py
```

训练完成后，下面这些脚本会直接读取：
- `outputs/best_flow_v3.pt`
- `outputs/run_config.json`
- `outputs/data_split.json`

常用评估脚本：

```powershell
python .\scripts\timeflow_v3\predict_one_val_sample.py
python .\scripts\timeflow_v3\predict_one_val_sample_safe_kde.py
python .\scripts\timeflow_v3\evaluate_new_model_param_mae_multi_sample.py
```

常用配置都写在 `train_flow.py` 顶部：
- `TRAIN_H1_DIR`
- `TRAIN_L1_DIR`
- `USE_L1`
- `TRAIN_SAMPLE_STEP`
- `VAL_FRACTION`
- `BATCH_SIZE`
- `EPOCHS`
- `SEED`
