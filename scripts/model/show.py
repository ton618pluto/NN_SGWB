import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from matplotlib import rcParams

# 设置SCI论文格式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',  # SCI论文常用字体
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 创建图形
fig, (ax_main, ax_legend) = plt.subplots(2, 1, figsize=(15, 20),
                                         gridspec_kw={'height_ratios': [5, 0.5]})

# 隐藏图例坐标轴
ax_legend.axis('off')

# 设置主图
ax_main.axis('off')
ax_main.set_xlim(0, 100)
ax_main.set_ylim(0, 100)
ax_main.set_aspect('equal')

# 颜色定义
colors = {
    'input': '#4A90E2',  # 蓝色
    'conv': '#50E3C2',  # 青色
    'pool': '#B8E986',  # 浅绿
    'fc': '#9013FE',  # 紫色
    'norm': '#F8E71C',  # 黄色
    'flow': '#FF6B6B',  # 红色
    'output': '#FF8E00',  # 橙色
    'arrow': '#333333'  # 深灰色
}

# 字体设置
title_font = {'fontsize': 20, 'fontweight': 'bold', 'ha': 'center'}
section_font = {'fontsize': 16, 'fontweight': 'bold'}
text_font = {'fontsize': 10, 'ha': 'center', 'va': 'center'}

# 1. 添加标题
ax_main.text(50, 95, 'Normalizing Flow for Gravitational Wave Parameter Inference',
             fontdict=title_font)


# 2. 绘制数据流路径
def draw_arrow(x1, y1, x2, y2, color=colors['arrow'], width=2, head_width=5, style='-'):
    """绘制箭头"""
    ax_main.arrow(x1, y1, x2 - x1, y2 - y1,
                  head_width=head_width, head_length=8,
                  fc=color, ec=color, linewidth=width,
                  length_includes_head=True, alpha=0.8,
                  linestyle=style)


# 3. 绘制模块
def draw_module(x, y, width, height, label, color, text, radius=0.1):
    """绘制模块"""
    # 绘制圆角矩形
    rect = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax_main.add_patch(rect)

    # 添加标签
    ax_main.text(x, y + height / 2 + 2, label, ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    # 添加内部文本
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1) / 2 - i
        ax_main.text(x, y + y_offset * 3, line, ha='center', va='center',
                     fontsize=9, style='italic')


# 4. 绘制CNN部分 (左半边)
cnn_y = 80
cnn_blocks = [
    (20, cnn_y, 15, 8, "Input\nWaveform", colors['input'],
     "Shape: (1, 524288)\nTime-series data"),
    (40, cnn_y, 15, 8, "CNN Encoder", colors['conv'],
     "Conv1D + BatchNorm\nReLU + MaxPool\nMulti-scale features"),
    (60, cnn_y, 15, 8, "Context\nVector", colors['fc'],
     "Shape: (512,)\nLatent representation\nfor conditional flow"),
]

# 绘制CNN模块
for x, y, w, h, label, color, text in cnn_blocks:
    draw_module(x, y, w, h, label, color, text)

# 绘制CNN内部详细结构
cnn_detail_y = 65
cnn_layers = [
    (20, cnn_detail_y, 12, 6, "Conv1D", colors['conv'],
     "k=64, s=4, p=32\nChannels: 1→16"),
    (30, cnn_detail_y, 12, 6, "BatchNorm", colors['norm'],
     "Normalization\n16 channels"),
    (40, cnn_detail_y, 12, 6, "MaxPool", colors['pool'],
     "k=4, s=4\nDownsample 4×"),
    (50, cnn_detail_y, 12, 6, "Conv1D", colors['conv'],
     "k=16, s=2, p=8\nChannels: 16→32"),
    (60, cnn_detail_y, 12, 6, "Conv1D", colors['conv'],
     "k=8, s=2, p=4\nChannels: 32→64"),
    (70, cnn_detail_y, 12, 6, "Flatten", colors['fc'],
     "8192 → 256\n→ 512"),
]

for x, y, w, h, label, color, text in cnn_layers:
    draw_module(x, y, w, h, label, color, text)

# 5. 绘制Flow部分 (右半边)
flow_y = 40
flow_blocks = [
    (20, flow_y, 15, 8, "Parameter\nSpace", colors['input'],
     "Shape: (10,)\nGW parameters\nθ ∈ ℝ¹⁰"),
    (40, flow_y, 15, 8, "Normalizing\nFlow", colors['flow'],
     "8 layers\nMAF transforms\nLearn p(θ|x)"),
    (60, flow_y, 15, 8, "Base\nDistribution", colors['output'],
     "𝒩(0, I)\nStandard normal\nz ∈ ℝ¹⁰"),
]

for x, y, w, h, label, color, text in flow_blocks:
    draw_module(x, y, w, h, label, color, text)

# 6. 绘制Flow内部详细结构
flow_detail_y = 25
flow_layers = []
for i in range(6):
    x = 20 + i * 10
    flow_layers.append((x, flow_detail_y, 8, 6, f"Layer {i + 1}", colors['flow'],
                        f"Permute + MAF\nHidden: 256\nBins: 8"))

for x, y, w, h, label, color, text in flow_layers:
    draw_module(x, y, w, h, label, color, text)

# 7. 绘制数据流箭头
# CNN数据流
draw_arrow(27.5, cnn_y, 32.5, cnn_y)  # Input → CNN
draw_arrow(47.5, cnn_y, 52.5, cnn_y)  # CNN → Context

# CNN内部
for i in range(len(cnn_layers) - 1):
    draw_arrow(cnn_layers[i][0] + 6, cnn_detail_y, cnn_layers[i + 1][0] - 6, cnn_detail_y)

# Flow数据流
draw_arrow(27.5, flow_y, 32.5, flow_y)  # Parameters → Flow
draw_arrow(47.5, flow_y, 52.5, flow_y)  # Flow → Base

# Flow内部
for i in range(len(flow_layers) - 1):
    draw_arrow(flow_layers[i][0] + 4, flow_detail_y, flow_layers[i + 1][0] - 4, flow_detail_y)

# 上下文条件连接
ax_main.arrow(60, 76, 0, -20, head_width=2, head_length=3,
              fc=colors['arrow'], ec=colors['arrow'], linewidth=2,
              length_includes_head=True, alpha=0.6, linestyle='--')
ax_main.text(63, 66, "Context\nconditioning", ha='left', va='center',
             fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="white", alpha=0.8))

# 8. 绘制训练/推理路径
train_path_y = 15
# 训练路径
ax_main.text(20, train_path_y, "Training:", fontsize=12, fontweight='bold', ha='left')
ax_main.plot([25, 45], [train_path_y, train_path_y], color='k', linewidth=3, linestyle='-')
ax_main.scatter([25, 45], [train_path_y, train_path_y], c='k', s=50, zorder=5)
ax_main.text(25, train_path_y - 2, "θ, x", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
ax_main.text(35, train_path_y + 1, "Forward: NLL loss", ha='center', fontsize=10,
             style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
ax_main.text(45, train_path_y - 2, "Loss", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5))

# 推理路径
infer_path_y = 10
ax_main.text(20, infer_path_y, "Inference:", fontsize=12, fontweight='bold', ha='left')
ax_main.plot([25, 45], [infer_path_y, infer_path_y], color='g', linewidth=3, linestyle='--')
ax_main.scatter([25, 45], [infer_path_y, infer_path_y], c='g', s=50, zorder=5)
ax_main.text(25, infer_path_y - 2, "x", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
ax_main.text(35, infer_path_y + 1, "Sample: θ ~ p(θ|x)", ha='center', fontsize=10,
             style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
ax_main.text(45, infer_path_y - 2, "Posterior\nsamples", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

# 9. 添加数学公式
formula_y = 5
formulas = [
    (20, formula_y, r"$x \in \mathbb{R}^{1 \times 524288}$"),
    (40, formula_y, r"$\text{CNN}(x) = h \in \mathbb{R}^{512}$"),
    (60, formula_y, r"$f_{\phi}: \mathbb{R}^{10} \rightarrow \mathbb{R}^{10}$"),
    (80, formula_y, r"$z = f_{\phi}(\theta; h) \sim \mathcal{N}(0, I)$"),
]

for x, y, formula in formulas:
    ax_main.text(x, y, formula, ha='center', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 10. 添加图例
legend_elements = [
    patches.Patch(facecolor=colors['input'], edgecolor='black', label='Input/Output Layer'),
    patches.Patch(facecolor=colors['conv'], edgecolor='black', label='Convolutional Layer'),
    patches.Patch(facecolor=colors['pool'], edgecolor='black', label='Pooling Layer'),
    patches.Patch(facecolor=colors['norm'], edgecolor='black', label='Normalization'),
    patches.Patch(facecolor=colors['fc'], edgecolor='black', label='Fully Connected'),
    patches.Patch(facecolor=colors['flow'], edgecolor='black', label='Flow Layer'),
    Line2D([0], [0], color='k', lw=2, label='Training Path'),
    Line2D([0], [0], color='g', lw=2, linestyle='--', label='Inference Path'),
]

ax_legend.legend(handles=legend_elements, loc='center', ncol=4,
                 fontsize=10, framealpha=0.9, fancybox=True)

# 11. 添加模型描述
desc_text = """Model Description:
• Architecture: Conditional Normalizing Flow for Bayesian inference
• Encoder: 1D CNN extracts features from gravitational wave signals
• Flow: 8-layer Masked Autoregressive Flow (MAF) with rational-quadratic splines
• Training: Minimize negative log-likelihood: ℒ = -𝔼[log p(θ|x)]
• Inference: Sample from posterior: θ ∼ p(θ|x) via flow inversion
• Applications: Gravitational wave parameter estimation, Bayesian inference
"""

ax_main.text(80, 85, desc_text, ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

# 12. 添加分节标题
ax_main.text(50, 90, "Feature Extraction Network", fontdict=section_font,
             ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
ax_main.text(50, 50, "Normalizing Flow Transformation", fontdict=section_font,
             ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

# 调整布局
plt.tight_layout()

# 保存图片（多种格式）
save_formats = ['png', 'pdf', 'svg', 'eps']
for fmt in save_formats:
    filename = f'gwflow_model_structure.{fmt}'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"Saved: {filename}")

plt.show()

# 单独创建一个更简洁的版本用于论文
fig2, ax2 = plt.subplots(figsize=(12, 8))

# 简化版本 - 只显示主要模块
ax2.axis('off')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)

# 绘制主要模块
main_modules = [
    (20, 60, 20, 12, "Input\nGravitational Wave", colors['input'],
     "x ∈ ℝ^{1×524288}\nTime-domain strain data"),
    (50, 60, 20, 12, "CNN Encoder", colors['conv'],
     "Feature extraction\nMultiple conv layers\nOutput: h ∈ ℝ^{512}"),
    (80, 60, 20, 12, "Context Vector", colors['fc'],
     "Conditioning vector\nh = CNN(x)\nDeterministic encoding"),
    (20, 30, 20, 12, "Parameters\nθ ∈ ℝ^{10}", colors['input'],
     "GW source parameters\n(mass, spin, distance, ...)"),
    (50, 30, 20, 12, "Normalizing Flow", colors['flow'],
     "f_φ: ℝ^{10} → ℝ^{10}\n8 MAF layers\nLearn p(θ|x)"),
    (80, 30, 20, 12, "Base\nDistribution", colors['output'],
     "z ∼ 𝒩(0, I)\nLatent variables\nSimple prior"),
]

for x, y, w, h, label, color, text in main_modules:
    # 绘制圆角矩形
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.add_patch(rect)

    # 添加标签
    ax2.text(x, y + h / 2 + 2, label, ha='center', va='bottom',
             fontsize=12, fontweight='bold')

    # 添加内部文本
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1) / 2 - i
        ax2.text(x, y + y_offset * 4, line, ha='center', va='center',
                 fontsize=9, style='italic')

# 绘制主要箭头
# 水平箭头
for x1, x2, y in [(25, 40, 60), (70, 80, 60), (25, 40, 30), (70, 80, 30)]:
    ax2.arrow(x1, y, x2 - x1, 0, head_width=3, head_length=5,
              fc=colors['arrow'], ec=colors['arrow'], linewidth=2)

# 垂直箭头（上下文连接）
ax2.arrow(80, 54, 0, -18, head_width=3, head_length=5,
          fc=colors['arrow'], ec=colors['arrow'], linewidth=2, linestyle='--')
ax2.text(82, 45, "Context\nconditioning", ha='left', va='center',
         fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="white", alpha=0.8))

# 添加数学描述
ax2.text(50, 15, r"Training: $\mathcal{L} = -\mathbb{E}_{(\theta,x)}[\log p_\phi(\theta|x)]$",
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))
ax2.text(50, 10, r"Inference: $\theta = f_\phi^{-1}(z; h)$ where $z \sim \mathcal{N}(0, I)$",
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))

ax2.text(50, 95, "Conditional Normalizing Flow for Gravitational Wave Analysis",
         fontdict=title_font, ha='center')

plt.tight_layout()
plt.savefig('gwflow_simplified.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.savefig('gwflow_simplified.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

print("\n图片已保存为多种格式:")
print("1. gwflow_model_structure.png/pdf/svg/eps - 详细版本")
print("2. gwflow_simplified.pdf/png - 简化版本（适合论文）")