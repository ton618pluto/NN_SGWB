# TimeFlow V2 对话整理

## 1. 任务背景

本次讨论围绕 `scripts` 目录下的 SGWB 参数估计任务展开，核心目标是：

- 保持**原始时域数据**作为模型输入；
- 在现有 `CNNModel` 和 `GWFlowModel` 的基础上，重新设计一套更适合长时域输入的模型；
- 新代码放在一个独立目录中，避免覆盖原有实现；
- 代码尽量简洁，便于直接修改路径和参数后运行。

---

## 2. 原始模型阅读结论

用户要求先阅读现有模型：

- `scripts/model/CNNModel.py`
- `scripts/model/GWFlowModel.py`

阅读后的主要判断：

1. 原有 `SimpleCNN1D` 采用：
   - 3 层卷积 + pooling
   - 最后 `flatten`
   - 再接大规模全连接层

2. 原有 `GWFlowModel` 采用：
   - `SimpleCNN1D` 作为 `embedding_net`
   - `nflows` 做 conditional normalizing flow
   - 学习联合后验 `p(theta | x)`

3. 存在的主要问题：
   - 原始时域长度为 `524288`，旧编码器较依赖 `flatten + FC`；
   - 对长时域统计信息建模不够自然；
   - 原始代码主要按单通道思路组织；
   - 训练入口参数较零散，不利于快速修改和直接运行。

---

## 3. 数据输入讨论结论

期间讨论过是否使用频域 summary，但用户明确说明：

> 课题组要求必须使用原始时域数据，这不在讨论范围外。

因此最终方案严格限定在：

- 输入为原始时域数据；
- 支持 `H1` 和 `L1` 双探测器；
- 不引入频域特征替代方案。

最终实现中：

- 若 `USE_L1 = True`，则输入形状为 `[2, 524288]`；
- 若 `USE_L1 = False`，则输入形状为 `[1, 524288]`。

---

## 4. 新增目录与新实现

为避免改坏原始代码，新实现全部放在：

`scripts/timeflow_v2`

新增文件包括：

- `scripts/timeflow_v2/__init__.py`
- `scripts/timeflow_v2/dataset.py`
- `scripts/timeflow_v2/model.py`
- `scripts/timeflow_v2/train_flow.py`
- `scripts/timeflow_v2/README.md`
- `scripts/timeflow_v2/model_fig/timeflow_v2_model_structure.html`

原始文件未被覆盖：

- `scripts/model/CNNModel.py`
- `scripts/model/GWFlowModel.py`
- `scripts/GWDataset.py`

---

## 5. 新版模型设计思路

### 5.1 总体结构

新版模型 `TimeFlow V2` 的整体流程为：

1. 从 `.npy` 文件读取原始时域样本；
2. 将 `H1` / `L1` 拼成单通道或双通道输入；
3. 用新的 `TemporalEncoder1D` 提取时域特征；
4. 将编码器输出作为 conditional flow 的 `context`；
5. 使用归一化流学习 10 维物理超参数的联合后验。

---

### 5.2 编码器改进

相较旧版 `SimpleCNN1D`，新版 `TemporalEncoder1D` 的改动：

- 使用更深的残差 1D 卷积结构；
- 通过 dilation 扩大感受野；
- 不再使用超大的 `flatten + 大全连接`；
- 改用 `AdaptiveAvgPool1d` + `AdaptiveMaxPool1d`；
- 最后通过 projector 输出固定维度 `context`。

默认配置下：

- `context_dim = 256`

---

### 5.3 条件归一化流

新版 `GWFlowModelV2` 保留了原来的总体思路：

- 编码器输出 `context`
- flow 建模 `p(theta | x)`

默认 flow 配置：

- `param_dim = 10`
- `flow_layers = 6`
- `flow_hidden = 256`
- `num_bins = 8`
- `tail_bound = 3.0`

训练目标仍然是：

`loss = -log p(theta | x)`

---

## 5.4 相比原始简单 CNN 的改进与优势

用户原始模型中的 `SimpleCNN1D` 主要结构是：

- 3 层卷积；
- 逐层 pooling；
- 最后直接 `flatten`；
- 用大规模全连接层输出结果或提供 flow 的 context。

修改后的 `TimeFlow V2` 相比这个简单 CNN，主要优势如下。

### （1）更适合超长时域输入

原始输入长度是：

`524288`

对于这么长的时域序列，旧版简单 CNN 的问题在于：

- 卷积层数较浅；
- 高层特征主要依赖最后的 `flatten + FC` 来整合全局信息；
- 对长时间尺度的统计结构利用不够自然。

而新版模型：

- 使用更深的残差卷积块；
- 通过 stride 和 dilation 逐步扩大感受野；
- 更适合从长时间原始时域中提取全局信息。

也就是说，新版模型不是把大量信息“硬塞给全连接层”，而是先在时域卷积层里尽可能完成层级式特征抽取。

### （2）参数利用方式更合理

旧版简单 CNN 最后的 `flatten` 会生成较大的特征向量，再接全连接层。这种方式会带来两个问题：

- 全连接层参数量较大；
- 更容易过拟合；
- 对长输入来说，训练代价偏高。

新版模型改用：

- `AdaptiveAvgPool1d`
- `AdaptiveMaxPool1d`

这样做的好处是：

- 不需要超大的展平向量；
- 参数量更可控；
- 保留了全局平均信息和显著峰值信息；
- 更适合作为后续 flow 的条件特征。

### （3）残差结构让训练更稳定

原始简单 CNN 结构比较直接，优点是实现简单，但在网络继续加深时，训练会更困难。

新版模型引入了残差块，优点包括：

- 更容易堆叠更深的时域网络；
- 梯度传播更稳定；
- 在不明显增加训练难度的情况下提高表达能力。

这对于当前这种长时域、弱信号、统计规律复杂的任务更重要。

### （4）支持双探测器联合输入

原始训练流程更多是围绕单通道 `H1` 来组织。

新版 `TimeFlow V2` 明确支持：

- `H1` 单通道；
- `H1 + L1` 双通道。

当 `USE_L1 = True` 时，输入可以直接组织为：

`[2, 524288]`

这样模型在编码阶段就可以同时看到两个探测器的原始时域数据，比单通道方案更符合你当前的任务目标。

### （5）和归一化流的衔接更自然

旧版 `SimpleCNN1D` 虽然也可以给 flow 提供 context，但由于最后依赖大规模全连接层，context 的形成方式偏“压缩式回归头”。

新版模型里：

- 编码器先提取全局时域表示；
- 再投影到固定维度 `context_dim`；
- 然后交给 conditional flow 建模联合后验。

这种结构更接近“特征提取器 + 后验估计器”的清晰分工：

- 编码器负责“看懂波形”；
- flow 负责“刻画参数后验”。

### （6）更适合后续扩展

原始简单 CNN 更像一个 baseline，适合快速试验；
但如果后续想继续增强模型，扩展空间有限。

新版模型则更容易继续往下扩展，例如：

- 增减残差块数量；
- 调整 dilation 策略；
- 调整 `context_dim`；
- 替换或增强 flow 层数；
- 后续加入更复杂的双分支编码结构。

因此从研究路线来看，新版更适合作为后续实验的主干版本。

### （7）需要客观看待的一点

虽然新版模型在结构上比简单 CNN 更合理，但这并不自动保证效果一定更好。

真正能否优于原模型，还取决于：

- 数据集规模；
- 标签是否足够可辨识；
- 双通道数据质量；
- 训练是否稳定；
- 超参数是否合适。

因此更准确的说法应该是：

> 新版模型在结构设计上更适合当前“长时域 + 双探测器 + 联合后验估计”的任务，但最终效果仍需要通过实验验证。

---

## 6. 数据集实现调整

### 6.1 数据集文件

数据集新版实现位于：

`scripts/timeflow_v2/dataset.py`

### 6.2 数据处理规则

当前数据集逻辑固定为：

1. 从 `H1` 和 `L1` 的 `.npy` 文件读取数据；
2. 按文件名去掉前缀后进行 `H1/L1` 配对；
3. 原始数据乘以 `1e23`；
4. 每个通道单独去均值；
5. 标签删除原始第 `3` 个分量，保留 `10` 维；
6. 标签做 z-score 标准化。

### 6.3 抽样机制

为了解决数据量太大、只想先跑通流程的问题，增加了抽样机制：

- `TRAIN_SAMPLE_STEP`
- `VAL_SAMPLE_STEP`

含义：

- `100` 表示每隔 100 个文件取 1 个样本；
- `1` 表示全部使用。

例如当前默认设置：

- 训练集：`1/100`
- 验证集：全量

---

## 7. 数据加载进度条

用户要求在加载数据时显示进度。

因此在 `dataset.py` 中加入了 `tqdm`：

1. 匹配 `H1/L1` 文件时显示：
   - `[train] Matching H1/L1`
   - `[val] Matching H1/L1`

2. 统计与加载标签时显示：
   - `[train] Loading labels`
   - `[val] Loading labels`

这使得初始化阶段能明显看到数据准备进度。

---

## 8. 训练脚本简化

### 8.1 原则

用户明确指出：

> 输入目录不用弄那么复杂，尽量写在代码里，有需要就改代码就行。

因此 `train_flow.py` 最终被简化为：

- 不再依赖复杂命令行参数；
- 所有路径和训练超参数直接写在文件顶部；
- 用户需要修改时，直接改常量即可。

### 8.2 当前配置入口

配置集中在：

`scripts/timeflow_v2/train_flow.py`

用户主要需要改的变量：

- `TRAIN_H1_DIR`
- `TRAIN_L1_DIR`
- `VAL_H1_DIR`
- `VAL_L1_DIR`
- `USE_L1`
- `TRAIN_SAMPLE_STEP`
- `VAL_SAMPLE_STEP`
- `BATCH_SIZE`
- `EPOCHS`

### 8.3 运行方式

当前运行命令为：

```powershell
python .\scripts\timeflow_v2\train_flow.py
```

---

## 9. 双通道输入说明

用户特别追问：

> 你帮我写的代码输入是双通道的吗？

最终确认：

- 是的，支持双通道；
- 对应逻辑在：
  - `scripts/timeflow_v2/train_flow.py`
  - `scripts/timeflow_v2/dataset.py`
  - `scripts/timeflow_v2/model.py`

具体含义：

1. `train_flow.py` 中：
   - `USE_L1 = True` 表示启用双通道

2. `dataset.py` 中：
   - 成功配对 `H1/L1` 后，把二者作为同一个样本；
   - `H1` 作为第一个通道；
   - `L1` 作为第二个通道；
   - 最终输入形状为 `[2, 524288]`

3. `model.py` 中：
   - `TemporalEncoder1D(in_channels=2)` 接收双通道输入；
   - 若只使用 H1，则 `in_channels=1`

---

## 10. 注释完善

用户要求：

- 给新增代码加上清晰注释；
- 特别希望配置区和路径区更容易理解。

因此在以下文件中都补充了中文注释：

- `scripts/timeflow_v2/__init__.py`
- `scripts/timeflow_v2/dataset.py`
- `scripts/timeflow_v2/model.py`
- `scripts/timeflow_v2/train_flow.py`

重点补充的内容包括：

- 路径配置说明；
- 抽样比例含义；
- 双通道拼接逻辑；
- 编码器结构说明；
- flow 的作用；
- 训练循环和 checkpoint 保存逻辑。

---

## 11. 模型结构 HTML 展示页

用户要求：

> 将模型结构写成 html 文件的形式展示，并保存在 `timeflow_v2` 下的 `model_fig` 目录中。

因此新增：

`scripts/timeflow_v2/model_fig/timeflow_v2_model_structure.html`

该 HTML 页面包含：

- 整体流程图；
- 编码器结构；
- 条件 flow 结构；
- 训练与推理方式；
- 当前默认配置展示。

可直接在浏览器中打开查看。

---

## 12. 当前代码状态

目前已经完成：

- 独立新目录实现；
- 支持 H1/L1 双通道；
- 支持按 `1/100` 抽样快速测试；
- 数据加载进度条；
- 简化版训练入口；
- 中文注释；
- 模型结构 HTML 展示页。

当前环境中曾发现一个限制：

- 本地环境缺少 `nflows` 时，完整训练无法真正启动；
- 但新版代码已经加入了缺失依赖时的提示信息。

---

## 13. 当前推荐使用方式

1. 打开：
   - `scripts/timeflow_v2/train_flow.py`

2. 检查并修改：
   - 数据目录
   - `USE_L1`
   - 抽样步长
   - batch size 和 epoch

3. 运行：

```powershell
python .\scripts\timeflow_v2\train_flow.py
```

4. 打开模型结构展示页：

`scripts/timeflow_v2/model_fig/timeflow_v2_model_structure.html`

---

## 14. 一句话总结

本次对话最终得到了一套**独立于旧代码的、支持原始时域双通道输入的 TimeFlow V2 实现**，其特点是：

- 更适合长时域信号建模；
- 支持 H1/L1 双通道；
- 可用小比例抽样快速跑通流程；
- 训练入口和配置更直观；
- 注释和结构展示更适合后续维护与汇报。
