import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib

matplotlib.use('Agg')  # 绘图后端

# ---------- 显示设置 ----------
# 依然保留中文显示配置，否则 print 语句输出中文时，虽然终端没问题，但如果涉及日志记录可能会报错
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 参数名称映射：Key 保持原样，Value 换成英文用于图片显示
parameter_names = {
    'alpha_z': 'Redshift Ascent Index (alpha_z)',
    'beta_z': 'Redshift Descent Index (beta_z)',
    'zp': 'Peak Redshift (zp)',
    'alpha_m': 'Mass Power-law Index (alpha_m)',
    'm_max': 'Max BH Mass Cutoff',
    'delta_m': 'Low Mass Smoothing (delta_m)',
    'm_min': 'Min BH Mass Cutoff',
    'lambda_peak': 'Gaussian Peak Fraction (lambda_peak)',
    'mu_m': 'Gaussian Peak Mean (mu_m)',
    'sigma_m': 'Gaussian Peak Std Dev (sigma_m)',
    'beta_q': 'Mass Ratio Index (beta_q)',
}


# 打印参数范围和统计信息（print 语句保留中文）
def print_param_ranges(path):
    try:
        data = np.load(path)
        # 表头保留中文说明，但内部参数名用英文描述
        print(f"{'参数代码':<12} | {'均值':<8} | {'标准差':<8} | {'最小值':<8} | {'最大值':<8} | {'英文描述'}")
        print("-" * 100)

        for key in data.files:
            values = data[key]
            v_mean, v_std = np.mean(values), np.std(values)
            v_min, v_max = np.min(values), np.max(values)
            name = parameter_names.get(key, "Unknown")
            print(f"{key:<12} | {v_mean:<8.2f} | {v_std:<8.2f} | {v_min:<8.2f} | {v_max:<8.2f} | {name}")

        data.close()

    except FileNotFoundError:
        print(f"错误：找不到文件 {path}")
    except Exception as e:
        print(f"发生错误：{e}")


# 绘制每个参数的独立分布图（图片内为英文）
def save_individual_distributions(file_path, output_dir='hyperparam_distributions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"成功创建目录: {output_dir}")

    try:
        data = np.load(file_path)
        keys = data.files

        for key in keys:
            values = data[key]
            plt.figure(figsize=(8, 5))
            sns.histplot(values, kde=True, color='royalblue', edgecolor='white')

            # 图片标题和标签设为英文
            plt.title(f"Distribution of {parameter_names.get(key, key)}", fontsize=14, fontweight='bold')
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.grid(axis='y', alpha=0.3)

            save_path = os.path.join(output_dir, f"{key}_distribution.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"{key:<12} | 已保存到 {save_path}")

        data.close()
        print(f"\n[完成] 所有 {len(keys)} 个参数分布图已保存至 '{output_dir}/' 文件夹。")

    except Exception as e:
        print(f"生成图片时发生错误：{e}")


# 绘制所有参数在一张总图（图片内为英文）
def plot_param_distributions(path, save_name='./hyperparam_distributions/joint_hyperparam_distributions.png'):
    try:
        data = np.load(path)
        keys = sorted(list(data.files))
        num_params = len(keys)

        cols = 3
        rows = math.ceil(num_params / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        print(f"\n{'开始绘制总表...':<12}")
        print(f"{'参数':<12} | {'均值':<8} | {'标准差':<8} | {'最小值':<8} | {'最大值':<8}")
        print("-" * 65)

        for i, key in enumerate(keys):
            values = data[key]
            v_mean, v_std = np.mean(values), np.std(values)
            v_min, v_max = np.min(values), np.max(values)
            print(f"{key:<12} | {v_mean:<8.2f} | {v_std:<8.2f} | {v_min:<8.2f} | {v_max:<8.2f}")

            sns.histplot(values, kde=True, ax=axes[i], color='royalblue', edgecolor='white')
            # 图表内使用英文
            axes[i].set_title(parameter_names.get(key, key), fontsize=10, fontweight='bold')
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"\n[成功] 总分布图已保存为: {save_name}")
        data.close()

    except Exception as e:
        print(f"执行失败，错误信息：{e}")


# ----------------- 执行部分 -----------------
file_path = './joint_hyperparams_train/v2/joint_hyperparams.npz'

# 1. 打印统计信息（控制台输出中文）
print_param_ranges(file_path)

# 2. 保存独立分布图（控制台输出中文提示，图片为英文）
save_individual_distributions(file_path)

# 3. 绘制总图（控制台输出中文提示，图片为英文）
plot_param_distributions(file_path)