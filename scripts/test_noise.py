from gwpy.timeseries import TimeSeries
import os

# 1. 确定文件路径
job_number = 2563
typeList=['H1','L1']
for item in typeList:
    file_name = f"{item}-STRAIN-{job_number:05d}-duration.gwf"
    file_path = os.path.join(f'./noise_waveform_{item}', file_name)

    # 2. 确定通道名称 (根据您生成文件的代码推测)
    channel_name = f"{item}:TEST_INJ"


    # 3. 确定时间 (如果需要读取完整段，可以不指定时间或使用生成时的 st/et)
    # 假设 duration = 2048
    # st = (job_number - 1) * 2048
    # et = st + 2048

    try:
        # 读取整个文件中的指定通道数据
        L_Noise = TimeSeries.read(file_path, channel_name)
        print(f'探测器{item}-jobNumber为{job_number}的信息')
        print(f"数据：{L_Noise}")
        print(f"成功读取文件: {file_path}")
        print(f"数据长度: {len(L_Noise)} 个样本")
        print(f"数据开始时间: {L_Noise.times[0].value}")

    except Exception as e:
        print(f"读取文件 {file_path} 失败，请检查通道名称和文件是否存在。错误: {e}")