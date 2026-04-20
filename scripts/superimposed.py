import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.io import gwf

# --- 配置参数 ---
def superimpose(detector_name):
    # 假设文件编号从 1 到 N_EVENTS
    pop_nums = 200
    frame_nums = 23

    # 路径设置
    SIGNAL_DIR = f'./training_set/{detector_name}/'
    NOISE_DIR = f'./noise_waveform_{detector_name}/'
    OUTPUT_DIR = f'./superimposed_{detector_name}/'  # 用于保存叠加结果的目录

    # GWF 文件中 Strain 数据的通道名称 (这个名称可能需要根据您的文件实际情况修改)
    STRAIN_CHANNEL = f'{detector_name}:TEST_INJ'  # 这是一个常见的占位符，请替换为实际通道名

    # 创建输出目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"开始处理 {pop_nums * frame_nums} 个事件...")

    # 循环处理每个文件编号
    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            # 构造文件路径
            signal_filename = f'{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf'
            noise_filename = f'{detector_name}-STRAIN-{(frame_num + 1):05d}-duration.gwf'

            signal_path = os.path.join(SIGNAL_DIR, signal_filename)
            noise_path = os.path.join(NOISE_DIR, noise_filename)
            output_path = os.path.join(OUTPUT_DIR, f'{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf')

            # 检查文件是否存在
            if not os.path.exists(signal_path):
                print(f"信号文件不存在：{signal_path}")
                continue
            if not os.path.exists(noise_path):
                print(f"噪声文件不存在：{noise_path}")
                continue

            print(f"\n--- 正在处理事件 pop{pop_num:05d}_sample{frame_num:02d} ---")

            try:
                # 1. 读取信号 TimeSeries
                # TimeSeries.read() 会自动根据文件后缀识别格式
                # 注意：这里我们假设信号和噪声文件中的时间戳和采样率是匹配的
                signal_ts = TimeSeries.read(signal_path, STRAIN_CHANNEL)
                print(f"读取信号文件：{signal_path} (长度: {len(signal_ts)})")

                # 2. 读取噪声 TimeSeries
                noise_ts = TimeSeries.read(noise_path, STRAIN_CHANNEL)
                print(f"读取噪声文件：{noise_path} (长度: {len(noise_ts)})")

                # 3. 检查 TimeSeries 匹配性 (确保数据可以安全相加)
                if len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt:
                    print(f"🚨 警告：事件 pop{pop_num:05d}_sample{frame_num:02d} 的信号和噪声 TimeSeries 长度或采样率不匹配！跳过叠加。")
                    continue

                # 4. 叠加信号和噪声
                # 在 gwpy 中，TimeSeries 对象可以直接相加
                superimposed_ts = signal_ts + noise_ts
                print("✅ 信号和噪声叠加完成。")
                print(f"信号:{signal_ts}")
                print(f"噪声:{noise_ts}")
                print(f"叠加后的结果：{superimposed_ts}")

                # 5. 可选：将叠加后的数据保存到新的 GWF 文件
                # 我们使用 TimeSeries.write() 方法。
                # 注意：gwpy.io.gwf.write() 可以用来写入 GWF 文件，
                # 但 TimeSeries.write() 提供了更高级的封装。
                # 默认情况下，它将使用原 TimeSeries 的属性（如起始时间、采样率）。
                superimposed_ts.write(
                    output_path,
                )
                print(f"💾 叠加结果已保存到：{output_path}")
                print()

            except Exception as e:
                print(f"❌ 处理事件 pop{pop_num:05d}_sample{frame_num:05d} 时发生错误: {e}")


if __name__ == "__main__":
    dectors=['H1','L1']
    for dector_name in dectors:
        superimpose(dector_name)
    print("\n--- 所有事件处理完毕 ---")