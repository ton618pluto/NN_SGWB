import os
import matplotlib.pyplot as plt
import gwpy.timeseries as ts
from gwpy.timeseries import TimeSeries
import warnings

# 压制 gwpy 警告信息，保持输出界面整洁
warnings.filterwarnings('ignore', category=UserWarning, module='gwpy')

# 配置 matplotlib，使用通用字体，确保英文字符正常显示，避免中文乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置参数 ---
JOB_NUMBER = 2  # 要可视化的 Job 编号
DETECTORS = ['H1', 'L1']  # 要绘制的探测器列表
BASE_OUTPUT_DIR = 'plots_noise'  # 输出目录
DURATION_TO_PLOT = 1.0  # 仅绘制前 1.0 秒的数据

# 如果输出目录不存在，则创建它
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)
    print(f"警告: 正在创建输出目录: {BASE_OUTPUT_DIR}")


def visualize_h1_l1_time_series_comparison(job_number, output_dir):
    """
    读取 H1 和 L1 噪声时域数据，并将它们绘制在同一张图上。
    """

    all_strain_data = {}

    # 循环读取两个探测器的 GWF 文件
    for detector_name in DETECTORS:
        # 构建文件路径
        detector_base_dir = f'./noise_waveform_{detector_name}'
        file_name = f"{detector_name}-STRAIN-{job_number:05d}-duration.gwf"
        full_path = os.path.join(detector_base_dir, file_name)
        channel_name = f"{detector_name}:TEST_INJ"

        print(f"尝试读取 {detector_name} 文件: {full_path}")

        try:
            # 读取整个时间序列
            full_strain = TimeSeries.read(full_path, channel=channel_name)

            # 截取绘图所需的时长数据
            num_samples = int(DURATION_TO_PLOT * full_strain.sample_rate.value)
            all_strain_data[detector_name] = full_strain[:num_samples]
            print(f"成功: {detector_name} 数据读取成功。")

        except Exception as e:
            print(f"错误: 读取 {detector_name} GWF 文件失败: {e}")
            print(f"跳过 {detector_name} 的可视化。")
            return

    # --- 绘制对比图 ---

    plt.figure(figsize=(12, 6))

    # 绘制 H1 数据
    h1_data = all_strain_data['H1']
    plt.plot(h1_data.times.value, h1_data.value, label='H1 Strain', alpha=0.7)  # 标签使用英文

    # 绘制 L1 数据
    l1_data = all_strain_data['L1']
    plt.plot(l1_data.times.value, l1_data.value, label='L1 Strain', alpha=0.7)  # 标签使用英文

    # 使用英文标签和标题
    plt.xlabel("Time (s)")
    plt.ylabel(r"Strain (m/$m$)")
    plt.title(f"H1 and L1 Noise Time Series Comparison (Job {job_number}, First {DURATION_TO_PLOT}s)")
    plt.legend()
    plt.grid(True)

    # 保存图片，文件名使用英文
    comparison_fig_path = os.path.join(output_dir, f'H1_L1_Noise_TimeSeries_Comparison_{job_number}.png')
    plt.savefig(comparison_fig_path)
    plt.close()

    print(f"\n🎉 H1 和 L1 时域对比图已保存到: {comparison_fig_path}")


# --- 主执行部分 ---
visualize_h1_l1_time_series_comparison(JOB_NUMBER, BASE_OUTPUT_DIR)

print("\n=============== ✅ 任务完成 ===============")