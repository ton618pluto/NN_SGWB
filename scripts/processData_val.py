import os
from gwpy.timeseries import TimeSeries
import numpy as np
import pandas as pd


# 获取标签
def getLabel(file_path):
    df = pd.read_csv(file_path)
    first_column = df.columns[0]
    result_dict = {row[0]: list(row[2:]) for row in df.itertuples(index=False)}

    return result_dict

# 获取CBC-SGWB数据
def getSGWBData(d_name,base_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. 获取文件
    files = [f for f in os.listdir(base_path) if f.endswith('.gwf')]
    files.sort()

    # 试读前 10 个原文件
    # test_files = files[:10]
    num_splits = 8

    print(f"开始处理探测器{d_name}，每个文件将切分为 {num_splits} 段...")
    count=0

    for filename in files:
        # 获取标签
        temp_filename=filename.replace(f"{d_name}-", "").replace(".gwf","")
        label=lablesDict[temp_filename]
        file_full_path = os.path.join(base_path, filename)

        try:
            # 读取数据 (使用你确认过的通道名 H1:TEST_INJ)
            data = TimeSeries.read(file_full_path, channel=f"{d_name}:TEST_INJ").value

            # 计算每一段的长度
            total_length = len(data)
            split_length = total_length // num_splits

            for i in range(num_splits):
                start = i * split_length
                end = (i + 1) * split_length
                chunk = data[start:end]
                sample={
                    'data':chunk,
                    'label':label
                }

                # 生成新的文件名，例如: H1-pop00299_sample04_p0.npy
                base_name = os.path.splitext(filename)[0]
                split_filename = f"{base_name}_p{i}.npy"

                # 保存单段数据
                np.save(os.path.join(save_path, split_filename), sample)

            print(f"成功拆分并保存: {filename} -> {num_splits} files")
            count+=1
            if(count%20==0):
                print(f'✅ 处理了{count}个来自探测器{d_name}的CBC-SGWB数据文件')

        except Exception as e:
            print(f"处理 {filename} 失败: {e}")

    print("---")
    print(f"全部完成，切片文件保存在: {save_path}")


if __name__ =="__main__":
    # 1. 路径设置
    detectors=['H1','L1']

    # 获取标签，键为一个SGWB文件名，值为超参数（也就是标签）
    # base_label_path = './training_set/train_idx.csv'
    base_label_path = './val_set/train_idx.csv'
    # base_label_path = './test_set/test_idx.csv'
    lablesDict = getLabel(base_label_path)

    for i in range(len(detectors)):
        d_name=detectors[i]
        # base_path = f'./training_set/{d_name}'
        base_path = f'./val_set/{d_name}'
        # base_path = f'./test_set/{d_name}'
        # save_path = f'./processed_data_train/{d_name}_splits'  # 建议保存到子文件夹
        save_path = f'./processed_data_val/{d_name}_splits'  # 建议保存到子文件夹
        # save_path = f'./processed_data_test/{d_name}_splits'  # 建议保存到子文件夹
        getSGWBData(d_name,base_path,save_path)

