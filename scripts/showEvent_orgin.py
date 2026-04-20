import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


# 将探测器质量转换成普通质量
def get_real_masses(path):
    """
    加载数据并将探测器质量转换为源质量
    """
    data = np.load(path)
    # 将数据转为字典，方便修改
    params = {key: data[key] for key in data.files}

    # 检查是否存在红移 z 和质量参数
    if 'z' in params:
        z = params['z']

        # 转换 m1 和 m2 (从探测器质量变为源质量)
        if 'm1' in params:
            params['m1'] = params['m1'] / (1 + z)
        if 'm2' in params:
            params['m2'] = params['m2'] / (1 + z)

        # 如果 m1 是啁啾质量 (Chirp Mass)，同样需要转换
        # M_src = M_det / (1 + z)

    return params


# 打印各个参数的范围
def print_param_ranges(path):
    try:
        # 加载 npz 文件
        data = get_real_masses(path)

        print('主质量：',data['m1'])
        print('次质量：',data['m2'])

        print(f"{'参数代码':<10} | {'最小 (Min)':<12} | {'最大 (Max)':<12} | {'中文名称'}")
        print("-" * 70)

        # 遍历文件中的所有 key
        for key in data:
            values = data[key]
            v_min = np.min(values)
            v_max = np.max(values)

            # 获取中文描述，如果 key 不在字典里则显示未知
            name = parameter_names.get(key, "未知参数")

            print(f"{key:<12} | {v_min:<13.2f} | {v_max:<13.2f} | {name}")

        # 记得关闭文件流
        # data.close()
        return data

    except FileNotFoundError:
        print(f"错误：找不到文件 {path}")
    except Exception as e:
        print(f"发生错误：{e}")


# 各个参数的分布
def save_individual_distributions(data, output_dir='events_distribution18'):
    """
    加载npz文件，为每个参数生成独立的分布图，并保存在指定目录下。
    """
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        # 2. 加载数据

        print(f"{'Parameter':<10} | {'Status'}")
        print("-" * 30)

        # 3. 遍历参数并绘图
        for key in data:
            values = data[key]

            # 创建独立的画布
            plt.figure(figsize=(8, 5))

            # 绘制分布图
            sns.histplot(values, kde=True, color='royalblue', edgecolor='white')

            # 设置英文标签和标题
            plt.title(f"Distribution of {key}", fontsize=14, fontweight='bold')
            plt.xlabel(key)
            plt.ylabel("Count")
            plt.grid(axis='y', alpha=0.3)

            # 保存图片，文件名为 参数名.png
            save_path = os.path.join(output_dir, f"{key}_distribution.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

            # 关闭当前画布，释放内存
            plt.close()

            print(f"{key:<10} | Saved to {save_path}")

        # data.close()
        print(f"\n[Success] All {len(data)} plots are saved in '{output_dir}/' folder.")



    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data


def plot_param_distributions(data, save_name='cbc_distributions.png'):
    """
    Load npz file, print parameter statistics, and generate distribution plots in English.
    """
    try:
        # Load the npz file
        num_params = len(data)

        # Calculate grid size (3 columns)
        cols = 3
        rows = math.ceil(num_params / cols)

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        print(f"{'Parameter':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 65)

        i = 0
        for key in data:
            values = data[key]
            # print(key)

            # Print statistics to console
            v_mean, v_std = np.mean(values), np.std(values)
            v_min, v_max = np.min(values), np.max(values)
            print(f"{key:<10} | {v_mean:<10.2f} | {v_std:<10.2f} | {v_min:<10.2f} | {v_max:<10.2f}")

            # Plot distribution
            sns.histplot(values, kde=True, ax=axes[i], color='royalblue', edgecolor='white')
            axes[i].set_title(f"Distribution of {key}", fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")
            i += 1

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.show()

        print(f"\n[Success] Distribution plot saved as: {save_name}")
        # data.close()

    except Exception as e:
        print(f"Error during execution: {e}")


# 对应你图片中的参数映射，方便打印更友好的中文标签
parameter_names = {
    'tc': '并合时间 (Coalescence Time)',
    'm1': '主质量 (Primary Mass)',
    'm2': '次质量 (Secondary Mass)',
    'dL': '光度距离 (Luminosity Distance)',
    'z': '红移 (Redshift)',
    'ra': '赤经 (Right Ascension)',
    'dec': '赤纬 (Declination)',
    'psi': '偏振角 (Polarization Angle)',
    'phi_c': '并合相位 (Coalescence Phase)',
    'iota': '轨道倾角 (Inclination Angle)',
    'true_alpha_z': 'alpha_z',
    'true_beta_z': 'beta_z',
    'true_zp': 'zp',

    'true_alpha_m': 'alpha_m',
    'true_m_max': 'm_max',

    'true_delta_m': 'delta_m',
    'true_m_min': 'm_min',

    'true_lambda_peak': 'lambda_peak',
    'true_mu_m': 'mu_m',
    'true_sigma_m': 'sigma_m',
    'true_beta_q' : 'beta_q'
}

# 文件路径
# file_path = './parameter_sampling_train/CBC_params_example00000.npz'
file_path = './CBC_params_example18.npz'
save_path = 'events_distribution18/cbc_distributions18.png'

data = print_param_ranges(file_path)
# 将每个参数图分开放
save_individual_distributions(data)

# 所有参数图放在一张
# plot_param_distributions(data,save_path)

# 执行函数
