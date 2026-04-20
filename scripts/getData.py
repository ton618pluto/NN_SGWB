import os
import numpy as np

# 设置切片文件保存的路径
load_path = './processed_data/H1_splits'  # 确保路径与保存时一致


def check_split_data():
    # 1. 检查路径是否存在
    if not os.path.exists(load_path):
        print(f"错误：路径 '{load_path}' 不存在。")
        return

    # 2. 获取所有 .npy 切片文件
    all_files = [f for f in os.listdir(load_path) if f.endswith('.npy')]
    all_files.sort()

    total_files = len(all_files)

    for i in range(1):
        file_name = all_files[i]
        file_full_path = os.path.join(load_path, file_name)

        try:
            # 【关键修改】：加载包含字典的对象
            # item() 用于从 numpy array 中提取出原始的 dict
            loaded_obj = np.load(file_full_path, allow_pickle=True).item()
            print(loaded_obj)

            data_chunk = loaded_obj['data']
            label_val = loaded_obj['label']

            # 打印信息
            print(f"{file_name:<40} | {str(data_chunk.shape):<15} | {label_val}")

        except Exception as e:
            print(f"读取 {file_name} 失败: {e}")

    # 4. 详细打印第一个文件的结构预览
    print("-" * 80)
    print("第一个文件的完整数据结构预览:")
    sample_file = all_files[0]
    sample_data = np.load(os.path.join(load_path, sample_file), allow_pickle=True).item()

    print(f"Key值: {sample_data.keys()}")
    print(f"Data (前5位): {sample_data['data'][:5]}")
    print(f"Label: {sample_data['label']}")
    print("=" * 60)


if __name__ == "__main__":
    check_split_data()