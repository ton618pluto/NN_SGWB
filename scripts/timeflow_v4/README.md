# `timeflow_v4`

`timeflow_v4` 基于 `timeflow_v3`，但把原训练集按固定随机种子切成：

- `70%` 训练集
- `20%` 验证集
- `10%` 测试集

普通训练会输出：

- `outputs/best_flow_v4.pt`
- `outputs/run_config.json`
- `outputs/data_split.json`

断点续训版本还会额外输出：

- `outputs/latest_flow_v4.pt`
- `outputs/run_config_resume.json`

运行普通训练：

```powershell
python .\scripts\timeflow_v4\train_flow.py
```

运行断点续训：

```powershell
python .\scripts\timeflow_v4\train_flow_resume.py
```

`train_flow_resume.py` 会自动检查 `outputs/latest_flow_v4.pt`：

- 如果存在，则从上一次的下一个 epoch 继续训练
- 如果不存在，则从头开始训练

运行测试集评估：

```powershell
python .\scripts\timeflow_v4\evaluate_test_set.py
```

常用配置写在 `train_flow.py` 或 `train_flow_resume.py` 顶部：

- `TRAIN_H1_DIR`
- `TRAIN_L1_DIR`
- `USE_L1`
- `VAL_FRACTION`
- `TEST_FRACTION`
- `BATCH_SIZE`
- `EPOCHS`
- `SEED`
