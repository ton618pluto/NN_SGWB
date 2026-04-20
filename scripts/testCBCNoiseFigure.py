import os
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import numpy as np



def showSuperimposed(detector_name):
    # --- 配置参数 ---
    pop_nums = 1
    frame_nums = 2
    OUTPUT_DIR = f'./superimposed_{detector_name}/'  # 叠加信号的路径
    SIGNAL_DIR = f'./training_set/{detector_name}/'  # 纯净信号的路径
    PLOT_DIR = f'./plots_SGWB_noise/'  # 用于保存图片的目录
    STRAIN_CHANNEL = f'{detector_name}:TEST_INJ'  # 确保使用您写入时使用的正确通道名
    BANDPASS_LOW = 30  # 滤波低频截止
    BANDPASS_HIGH = 400  # 滤波高频截止
    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            # 构造文件路径
            signal_path = os.path.join(SIGNAL_DIR, f'{detector_name}-pop{pop_num:05d}_sample{frame_num:02d}.gwf')
            superimposed_path = os.path.join(OUTPUT_DIR, f'{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:02d}.gwf')

            plot_filename = f'{detector_name}-WAVEFORMS-pop{pop_num:05d}_sample{frame_num:02d}-comparison.png'
            plot_path = os.path.join(PLOT_DIR, plot_filename)

            # 创建图片输出目录
            if not os.path.exists(PLOT_DIR):
                os.makedirs(PLOT_DIR)

            print(f"正在读取叠加文件：{superimposed_path}")
            print(f"正在读取纯信号文件：{signal_path}")

            try:
                # 1. 读取叠加后的 TimeSeries 数据
                data_superimposed = TimeSeries.read(superimposed_path, STRAIN_CHANNEL)
                print(f'叠加后的结果',data_superimposed)

                # 2. 读取纯信号的 TimeSeries 数据
                data_signal = TimeSeries.read(signal_path, STRAIN_CHANNEL)
                print(f'信号强度',data_signal)

                # 3. 对数据进行带通滤波
                data_superimposed_filtered = data_superimposed.bandpass(BANDPASS_LOW, BANDPASS_HIGH).crop(
                    data_superimposed.times.min(), data_superimposed.times.max()
                )
                data_signal_filtered = data_signal.bandpass(BANDPASS_LOW, BANDPASS_HIGH).crop(
                    data_signal.times.min(), data_signal.times.max()
                )

                # 4. 绘制时域波形图 (使用标准的 Matplotlib)

                # 创建 Figure 和 Axes 对象
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # **核心修正：使用 Matplotlib ax1.plot() 直接绘制数据**

                # 绘制叠加信号 (Signal + Noise)
                # x 轴使用 TimeSeries 的时间戳 (.times.value)
                # y 轴使用 TimeSeries 的数值 (.value)
                ax1.plot(
                    data_superimposed_filtered.times.value,
                    data_superimposed_filtered.value,
                    label='Superimposed (Signal + Noise)',
                    color='C0',
                    linewidth=0.5  # 降低线宽，因为叠加数据通常较密
                )

                # 在同一个 Axes 上绘制纯信号
                ax1.plot(
                    data_signal_filtered.times.value,
                    data_signal_filtered.value,
                    label='Injected Signal (Pure)',
                    color='C1',
                    linestyle='--',
                    linewidth=1.5
                )

                # 设置 Axes 属性
                ax1.set_ylabel('Strain (Filtered)', fontsize=14)
                ax1.set_xlabel('Time (s)', fontsize=14)
                ax1.set_title(f'Event {detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:02d}: Filtered Waveform Comparison ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)', fontsize=16)

                # 标注信号可能发生的区域
                duration = data_superimposed.duration.value
                mid_time = data_superimposed.times.min().value + duration / 2

                # 将时间轴刻度转换为相对于信号中心的相对时间（可选，但更常见）
                start_time = data_superimposed_filtered.times.value[0]
                ax1.set_xlim(start_time, start_time + duration)  # 确保 X 轴范围正确

                # 添加垂直线 (使用绝对时间戳)
                ax1.axvline(mid_time, color='r', linestyle=':', linewidth=1.5, label='Approx. Injection Time')

                # 显示图例 (Legend)
                ax1.legend()
                plt.tight_layout()

                # 5. 保存图表到文件，不显示
                plt.savefig(plot_path)
                plt.close(fig)

                print(f"✅ 波形对比图已成功保存到：{plot_path}")
                print(f"提示：纯信号 (虚线) 现在应该清晰地显示在叠加信号 (实线) 的上方。")

            except FileNotFoundError:
                print(f"错误：未找到文件，请检查路径：{superimposed_path} 或 {signal_path}。")
            except Exception as e:
                # 如果仍然报错，打印出更详细的错误信息
                print(f"发生致命错误: {e}")
                print("请检查您是否已安装所有依赖项 (gwpy, matplotlib, numpy) 并且文件路径正确。")

if __name__ == "__main__":
    if __name__ == "__main__":
        detectors = ['H1', 'L1']
        for detector_name in detectors:
            showSuperimposed((detector_name))
        print("\n--- 所有事件处理完毕 ---")