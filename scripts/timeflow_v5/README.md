# `timeflow_v5`

`timeflow_v5` 使用叠加噪声后的切分数据进行训练，输入目录为：

- `scripts/processed_data_superimposed/v0/H1_splits`
- `scripts/processed_data_superimposed/v0/L1_splits`

普通训练会输出：

- `scripts/timeflow_v5/outputs/best_flow_v5.pt`
- `scripts/timeflow_v5/outputs/run_config.json`
- `scripts/timeflow_v5/outputs/data_split.json`

断点续训版本还会额外输出：

- `scripts/timeflow_v5/outputs/latest_flow_v5.pt`
- `scripts/timeflow_v5/outputs/run_config_resume.json`

运行普通训练：

```powershell
python .\scripts\timeflow_v5\train_flow.py
```

运行断点续训：

```powershell
python .\scripts\timeflow_v5\train_flow_resume.py
```

`train_flow_resume.py` 会自动检查 `outputs/latest_flow_v5.pt`：

- 如果存在，则从上一次的下一个 epoch 继续训练
- 如果不存在，则从头开始训练

运行测试集评估：

```powershell
python .\scripts\timeflow_v5\evaluate_test_set.py
```
