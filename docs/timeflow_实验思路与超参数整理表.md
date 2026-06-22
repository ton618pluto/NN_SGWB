# TimeFlow 实验思路与超参数整理表

## 当前主线

当前 `timeflow_v1_a` 主线对应的是：

- 数据版本：`v21`
- 数据目录：`processed_data_train/v21/H1_splits` 和 `processed_data_train/v21/L1_splits`
- 模型开关：`USE_MODEL2 = True`
- 输入形式：`H1 + L1` 双通道时域序列
- 训练目标：条件后验 `p(theta | x)`
- 当前训练标签：`["alpha_m", "lambda_peak"]`

对应代码：

- 训练配置：`scripts/timeflow_v1_a/train_flow.py`
- 数据集：`scripts/timeflow_v1_a/dataset.py`
- 主模型：`scripts/timeflow_v1_a/model2.py`
- 超参数采样：`scripts/draw_hyperparameters_LHS.py`

## 实验链路

当前这条线的核心流程是：

1. 先生成一批群体超参数样本，保存为 `.npz`。
2. 基于这些群体超参数生成对应的 CBC/SGWB 时域数据。
3. 将 H1、L1 的样本各自保存为 `.npy`，每个样本里包含：
   - `data`
   - `label`
4. 在训练阶段把 H1/L1 按 sample key 配对，形成双通道输入。
5. 用时域编码器提取 `context`，再交给 conditional normalizing flow 学习后验。
6. 用测试集上的 `NLL`、覆盖率和置信区间宽度评估模型。

## 当前超参数定义

原始标签顺序在 `timeflow_v1_a/dataset.py` 里固定为：

```text
[zp, alpha_z, beta_z, alpha_m, m_max, delta_m, m_min, lambda_peak, mu_m, sigma_m, beta_q]
```

对应索引如下：

| 索引 | 参数名 |
|---|---|
| 0 | `zp` |
| 1 | `alpha_z` |
| 2 | `beta_z` |
| 3 | `alpha_m` |
| 4 | `m_max` |
| 5 | `delta_m` |
| 6 | `m_min` |
| 7 | `lambda_peak` |
| 8 | `mu_m` |
| 9 | `sigma_m` |
| 10 | `beta_q` |

## `v21` 数据里哪些参数在变化

当前 `draw_hyperparameters_LHS.py` 里，`v21` 对应的采样逻辑可以概括为：

### 真正在变化的参数

| 参数 | 当前范围 | 采样方式 |
|---|---|---|
| `alpha_m` | `[-3.0, 7.0]` | LHS 独立采样 |
| `delta_m` | `[0.0, 10.0]` | LHS 独立采样 |
| `mu_m` | `[25.0, 100.0]` | LHS 独立采样 |
| `sigma_m` | `[0.01, 5.0]` | LHS 独立采样 |

### 当前固定不变的参数

| 参数 | 固定值 |
|---|---:|
| `zp` | `1.9` |
| `alpha_z` | `2.7` |
| `beta_z` | `2.9` |
| `m_max` | `100.0` |
| `m_min` | `2.0` |
| `lambda_peak` | `0.04` |
| `beta_q` | `1.08` |

这和 `timeflow_v1_a/dataset.py` 里的 `DATA_VERSION_RANGE_OVERRIDES["v21"]` 是一致的。

## 当前训练标签

当前 `timeflow_v1_a` 真正送进 flow 学的标签不是全部 11 个参数，而是由：

```python
SELECTED_PARAMETERS = ["alpha_m", "lambda_peak"]
```

控制的。

因此当前训练任务实际上是 2 维后验：

```text
theta = [alpha_m, lambda_peak]
```

这里需要特别注意：

- `alpha_m` 在 `v21` 里是变化参数。
- `lambda_peak` 在 `v21` 里是固定值 `0.04`。

所以当前代码虽然在训练 2 维标签，但其中一维实际上没有变化。这会导致：

- `lambda_peak` 这一维缺少可学习的真实变化；
- 数据集标准化时该维度的原始方差会接近 0；
- 该任务更接近于 “学 `alpha_m`，同时带一个常数标签维度”。

如果目标是做真正的二维后验，这里的标签选择应当和 `v21` 的可变参数保持一致。

## 数据集构造与划分

### 输入数据

训练时读取：

- `processed_data_train/v21/H1_splits`
- `processed_data_train/v21/L1_splits`

每个样本文件是一个 `.npy` 字典，至少包含：

- `data`
- `label`

### H1/L1 配对

`GWDatasetV1A.build_records()` 会：

1. 先遍历 H1 目录里的 `.npy`
2. 从文件名中提取 sample key
3. 到 L1 目录里寻找同 key 的样本
4. 形成一条 record：

```python
{"h1": h1_path, "l1": l1_path}
```

### 训练/验证/测试划分

当前划分参数：

- `VAL_FRACTION = 0.1`
- `TEST_FRACTION = 0.1`
- `SEED = 42`

因此是：

- 训练集：80%
- 验证集：10%
- 测试集：10%

切分方式是：

- 对配对后的全部 `records` 做一次固定种子的随机置乱；
- 前 10% 给 `test`；
- 接下来的 10% 给 `val`；
- 剩余给 `train`。

这意味着当前不是：

- 按文件顺序切分
- 按物理参数分层切分
- 按某些参数组合单独留出 OOD 测试集

而是普通随机切分，但可复现。

切分结果会保存到：

- `timeflow_v1_a/outputs_dataset_v21_model2/data_split_v1_a.json`

后续评估脚本会直接复用这份 manifest。

## 当前输入预处理

`GWDatasetV1A` 里的预处理逻辑是：

1. 读取 H1/L1 的 `data`
2. 乘以 `input_scale = 1e23`
3. 对每个通道分别减去本样本均值
4. 再使用训练集统计量做标准化

即：

```text
features -> scale -> per-sample mean subtraction -> train-set normalization
```

标签侧的处理是：

1. 从原始 11 维 `label` 中按 `SELECTED_PARAMETERS` 抽取子集
2. 使用训练集上的 `label_mean / label_std` 做标准化

因此验证集和测试集的特征统计量、标签统计量都来自训练集，避免了信息泄漏。

## 当前模型结构

### 编码器

当前主线使用的是 `GWFlowModelV1A_Model2`。

编码器 `TemporalEncoder1DModel2` 的特点：

- 输入通道数：`2`
- stem：`Conv1d(2 -> 32, kernel_size=15, stride=1, padding=7)`
- 后接 7 个残差卷积块
- 前两层不降采样，后面逐步降采样
- 使用 `GroupNorm + GELU`
- 最后做三种全局池化特征汇聚：
  - `avg`
  - `max`
  - `std`
- 拼接后投影到 `context_dim = 256`

这比早期版本更强调保留前段时域细节。

### Flow

当前 flow 配置为：

- `FLOW_LAYERS = 8`
- `FLOW_HIDDEN = 256`
- `FLOW_BINS = 8`
- `FLOW_TAIL_BOUND = 3.0`
- 每层由：
  - `RandomPermutation`
  - `MaskedPiecewiseRationalQuadraticAutoregressiveTransform`

组成。

基础分布为：

- `StandardNormal((param_dim,))`

训练目标为：

```text
NLL = -log p(theta | x)
```

## 当前训练超参数

`timeflow_v1_a/train_flow.py` 当前配置如下：

| 参数 | 当前值 |
|---|---:|
| `DATA_VERSION` | `v21` |
| `USE_MODEL2` | `True` |
| `TRAIN_SAMPLE_STEP` | `1` |
| `VAL_FRACTION` | `0.1` |
| `TEST_FRACTION` | `0.1` |
| `BATCH_SIZE` | `32` |
| `EPOCHS` | `100` |
| `LEARNING_RATE` | `1e-4` |
| `WEIGHT_DECAY` | `1e-4` |
| `NUM_WORKERS` | `0` |
| `CONTEXT_DIM` | `256` |
| `FLOW_LAYERS` | `8` |
| `FLOW_HIDDEN` | `256` |
| `FLOW_BINS` | `8` |
| `FLOW_TAIL_BOUND` | `3.0` |
| `SEED` | `42` |
| `AUTO_RESUME` | `True` |
| `GPU_INDEX` | `0` |

训练实现上还包括：

- 优化器：`AdamW`
- 学习率调度：`ReduceLROnPlateau`
- 混合精度：`torch.amp.autocast`
- 梯度缩放：`torch.amp.GradScaler`

## 当前评估方式

测试集评估脚本是：

- `scripts/timeflow_v1_a/evaluate_test_set.py`
- `scripts/timeflow_v1_a/predict_one_test_sample_safe_kde.py`

当前会关注：

- 测试集平均 `NLL`
- 每个参数的 `MAE`
- `68% / 95%` 覆盖率
- `68% / 95%` 置信区间宽度
- 单样本 posterior corner/KDE 图

单样本后验采样数当前默认是：

- `NUM_POSTERIOR_SAMPLES = 800` 用于整套测试集评估
- `NUM_POSTERIOR_SAMPLES = 1000` 用于单样本 KDE 图

## 当前版本和旧文档的主要差异

这份整理表对应的是现在代码的真实状态，和旧版本相比，主要变化有：

- 主线已经切到 `v21`
- 输入是 `H1/L1` 双通道时域数据
- 当前模型主线是 `model2`
- 数据集划分是 `80/10/10`
- 当前真正变化的超参数是 `alpha_m / delta_m / mu_m / sigma_m`
- 当前训练标签改成了 `["alpha_m", "lambda_peak"]`
- 其中 `lambda_peak` 在 `v21` 下实际上是固定值

## 建议单独核对的点

当前代码里最需要单独留意的是这件事：

- 数据生成层面：`lambda_peak` 在 `v21` 中固定为 `0.04`
- 训练层面：`lambda_peak` 被放进了 `SELECTED_PARAMETERS`

如果你的目标是做真正有信息量的二维后验，建议重新确认：

1. 是不是应该改训练标签；
2. 还是应该改 `v21` 的超参数采样，让 `lambda_peak` 真正变化起来。
