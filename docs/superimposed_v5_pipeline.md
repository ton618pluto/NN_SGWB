# Superimposed V5 Data Pipeline

这份文档说明当前仓库里，从超参数采样到生成带噪声训练数据、再到 `timeflow_v5` 训练的完整数据流。

## 总流程

```text
draw_hyperparameters_v2.py
  -> joint_hyperparams_train/v3/joint_hyperparams_train_v3.npz

draw_CBC_params_diff.py
  -> parameter_sampling_train/v3/CBC_params_examplexxxxx.npz

generate_pure_cbc_frames_sampling5.py
  -> training_set/v3/training_set0/H1/*.gwf
  -> training_set/v3/training_set0/L1/*.gwf
  -> training_set/v3/training_set0/train_idx.csv

generate_pure_noise_frames.py
  -> noise_waveform_H1/*.gwf
  -> noise_waveform_L1/*.gwf

superimposed_friendly.py
  -> training_set_superimposed/v0/H1/*.gwf
  -> training_set_superimposed/v0/L1/*.gwf

processData_superimposed.py
  -> processed_data_superimposed/v0/H1_splits/*.npy
  -> processed_data_superimposed/v0/L1_splits/*.npy

timeflow_v5/train_flow.py
  -> 读取 processed_data_superimposed/v0/H1_splits 和 L1_splits
  -> 输出 scripts/timeflow_v5/outputs/*
```

## Step 1: 采样超参数

脚本：
- `scripts/draw_hyperparameters_v2.py`

作用：
- 采样一组联合超参数。
- 目前默认生成 `100` 组。

输出：
- `scripts/joint_hyperparams_train/v3/joint_hyperparams_train_v3.npz`

文件内容：
- `alpha_z`
- `beta_z`
- `zp`
- `alpha_m`
- `m_max`
- `delta_m`
- `m_min`
- `lambda_peak`
- `mu_m`
- `sigma_m`
- `beta_q`

这一份 `.npz` 是后面 `draw_CBC_params_diff.py` 的输入超参数表。

## Step 2: 按超参数组生成 CBC 事件参数文件

脚本：
- `scripts/draw_CBC_params_diff.py`

输入：
- `scripts/joint_hyperparams_train/v3/joint_hyperparams_train_v3.npz`

作用：
- 对每个超参数组 `pop_idx`，生成一份 CBC 事件参数文件。
- 每个文件里包含该组超参数下采样出来的一批事件参数，例如 `tc`、`m1`、`m2`、`z`、`dL`、`ra`、`dec` 等。
- 同时把该组对应的真实超参数也写进去，后面可作为标签使用。

输出目录：
- `scripts/parameter_sampling_train/v3`

典型输出文件：
- `scripts/parameter_sampling_train/v3/CBC_params_example00000.npz`
- `scripts/parameter_sampling_train/v3/CBC_params_example00001.npz`

数据含义：
- 一个 `CBC_params_examplexxxxx.npz` 对应一个超参数组，也就是后续的一个 `popxxxxx`。

## Step 3: 生成纯信号 frame

脚本：
- `scripts/generate_pure_cbc_frames_sampling5.py`

输入：
- `scripts/parameter_sampling_train/v3/CBC_params_examplexxxxx.npz`

作用：
- 逐个读取 `CBC_params_examplexxxxx.npz`。
- 对每个超参数组生成多段纯 CBC 信号 frame。
- 当前代码里 `all_jobs=24`，实际使用 `range(1, all_jobs)`，因此每组会对应 `23` 个样本。
- 样本编号会写成：
  - `pop00000_sample00000`
  - `pop00000_sample00001`
  - ...
  - `pop00000_sample00022`

输出目录：
- `scripts/training_set/v3/training_set0/H1`
- `scripts/training_set/v3/training_set0/L1`

典型输出文件：
- `scripts/training_set/v3/training_set0/H1/H1-pop00000_sample00000.gwf`
- `scripts/training_set/v3/training_set0/L1/L1-pop00000_sample00000.gwf`

附加输出：
- `scripts/training_set/v3/training_set0/train_idx.csv`

`train_idx.csv` 的作用：
- 记录每个 `sample_id` 对应的标签，也就是那组超参数值。
- 后面做 `processData_superimposed.py` 时仍然需要这份标签表。

## Step 4: 生成纯噪声 frame

脚本：
- `scripts/generate_pure_noise_frames.py`

作用：
- 为 `H1` 和 `L1` 生成纯噪声 `.gwf` 文件。
- 当前配对逻辑下，只需要准备 `23` 个噪声编号就够用，因为不同 `pop` 会复用同一组 `frame_num` 对应噪声。

输出目录：
- `scripts/noise_waveform_H1`
- `scripts/noise_waveform_L1`

典型输出文件：
- `scripts/noise_waveform_H1/H1-STRAIN-00001-duration.gwf`
- `scripts/noise_waveform_L1/L1-STRAIN-00001-duration.gwf`

编号关系：
- 纯信号的 `sample00000` 会配噪声 `STRAIN-00001`
- 纯信号的 `sample00001` 会配噪声 `STRAIN-00002`
- ...
- 纯信号的 `sample00022` 会配噪声 `STRAIN-00023`

说明：
- 如果你希望噪声可复现，可以改用 `scripts/generate_pure_noise_frames_seed.py`。
- 但就当前文档主流程而言，这一步按 `generate_pure_noise_frames.py` 记。

## Step 5: 纯信号 + 纯噪声叠加

脚本：
- `scripts/superimposed_friendly.py`

信号输入目录：
- `scripts/training_set/v3/training_set0/H1`
- `scripts/training_set/v3/training_set0/L1`

噪声输入目录：
- `scripts/noise_waveform_H1`
- `scripts/noise_waveform_L1`

输出目录：
- `scripts/training_set_superimposed/v0/H1`
- `scripts/training_set_superimposed/v0/L1`

典型输出文件：
- `scripts/training_set_superimposed/v0/H1/H1-SUPERIMPOSED-pop00000_sample00000.gwf`
- `scripts/training_set_superimposed/v0/L1/L1-SUPERIMPOSED-pop00000_sample00000.gwf`

叠加配对规则：
- 固定一个超参数组 `pop_num`
- 遍历该组的 `frame_num=0..22`
- 用
  - `H1-popxxxxx_sample00000.gwf` 配 `H1-STRAIN-00001-duration.gwf`
  - `H1-popxxxxx_sample00001.gwf` 配 `H1-STRAIN-00002-duration.gwf`
  - ...
  - `H1-popxxxxx_sample00022.gwf` 配 `H1-STRAIN-00023-duration.gwf`
- `L1` 同理

这一步的产物是“带噪声的完整 `.gwf` 数据”，还不是训练脚本直接读取的 `.npy`。

## Step 6: 把叠加后的 `.gwf` 切分成训练用 `.npy`

脚本：
- `scripts/processData_superimposed.py`

输入目录：
- `scripts/training_set_superimposed/v0/H1`
- `scripts/training_set_superimposed/v0/L1`

标签输入：
- `scripts/training_set_superimposed/v0/train_idx.csv`

注意：
- `superimposed_friendly.py` 只生成叠加后的 `.gwf`，不会自动生成或复制 `train_idx.csv`。
- 因此在运行 `processData_superimposed.py` 前，需要保证：

```text
scripts/training_set_superimposed/v0/train_idx.csv
```

已经存在。

最直接的做法是把：

```text
scripts/training_set/v3/training_set0/train_idx.csv
```

复制到：

```text
scripts/training_set_superimposed/v0/train_idx.csv
```

作用：
- 读取叠加后的 `.gwf`
- 每个文件切成 `8` 段
- 每段保存为一个 `.npy`
- 每个 `.npy` 内包含：
  - `data`
  - `label`

输出目录：
- `scripts/processed_data_superimposed/v0/H1_splits`
- `scripts/processed_data_superimposed/v0/L1_splits`

典型输出文件：
- `scripts/processed_data_superimposed/v0/H1_splits/H1-SUPERIMPOSED-pop00000_sample00000_p0.npy`
- `scripts/processed_data_superimposed/v0/L1_splits/L1-SUPERIMPOSED-pop00000_sample00000_p0.npy`

## Step 7: 用叠加后的切分数据训练 TimeFlow V5

脚本：
- `scripts/timeflow_v5/train_flow.py`

训练输入目录：
- `scripts/processed_data_superimposed/v0/H1_splits`
- `scripts/processed_data_superimposed/v0/L1_splits`

这一步会：
- 建立 H1/L1 配对样本
- 按固定随机种子切分成 train / val / test
- 训练 `timeflow_v5` 模型

输出目录：
- `scripts/timeflow_v5/outputs`

典型输出文件：
- `scripts/timeflow_v5/outputs/best_flow_v5.pt`
- `scripts/timeflow_v5/outputs/run_config.json`
- `scripts/timeflow_v5/outputs/data_split.json`

## 最终数据流总结

### 超参数流

```text
draw_hyperparameters_v2.py
  -> joint_hyperparams_train/v3/joint_hyperparams_train_v3.npz
  -> draw_CBC_params_diff.py
  -> parameter_sampling_train/v3/CBC_params_examplexxxxx.npz
```

### 纯信号流

```text
parameter_sampling_train/v3/CBC_params_examplexxxxx.npz
  -> generate_pure_cbc_frames_sampling5.py
  -> training_set/v3/training_set0/H1/*.gwf
  -> training_set/v3/training_set0/L1/*.gwf
  -> training_set/v3/training_set0/train_idx.csv
```

### 噪声流

```text
generate_pure_noise_frames.py
  -> noise_waveform_H1/*.gwf
  -> noise_waveform_L1/*.gwf
```

### 叠加与训练流

```text
training_set/v3/training_set0/H1|L1/*.gwf
+ noise_waveform_H1|L1/*.gwf
  -> superimposed_friendly.py
  -> training_set_superimposed/v0/H1|L1/*.gwf

training_set_superimposed/v0/H1|L1/*.gwf
+ training_set_superimposed/v0/train_idx.csv
  -> processData_superimposed.py
  -> processed_data_superimposed/v0/H1_splits|L1_splits/*.npy

processed_data_superimposed/v0/H1_splits|L1_splits/*.npy
  -> timeflow_v5/train_flow.py
  -> timeflow_v5/outputs/*
```

## 运行顺序建议

```powershell
python .\scripts\draw_hyperparameters_v2.py
python .\scripts\draw_CBC_params_diff.py
python .\scripts\generate_pure_cbc_frames_sampling5.py
python .\scripts\generate_pure_noise_frames.py
python .\scripts\superimposed_friendly.py
python .\scripts\processData_superimposed.py
python .\scripts\timeflow_v5\train_flow.py
```

如果噪声需要可复现，可以把第 4 步改为：

```powershell
python .\scripts\generate_pure_noise_frames_seed.py
```
