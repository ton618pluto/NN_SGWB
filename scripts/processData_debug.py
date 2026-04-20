import os
import time

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries


def log(message):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def getLabel(file_path):
    start = time.perf_counter()
    log(f"开始读取标签文件: {file_path}")
    df = pd.read_csv(file_path)
    result_dict = {row[0]: list(row[2:]) for row in df.itertuples(index=False)}
    elapsed = time.perf_counter() - start
    log(f"标签文件读取完成，共 {len(result_dict)} 条，用时 {elapsed:.3f}s")
    return result_dict


def getSGWBData(d_name, base_path, save_path, labels_dict):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"创建输出目录: {save_path}")

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        log(f"输入目录不存在，已创建空目录: {base_path}")

    files = [f for f in os.listdir(base_path) if f.endswith(".gwf")]
    files.sort()
    num_splits = 8

    log(f"开始处理探测器 {d_name}，共发现 {len(files)} 个 .gwf 文件，每个文件切分为 {num_splits} 段")
    count = 0

    for file_index, filename in enumerate(files, start=1):
        file_start = time.perf_counter()
        temp_filename = filename.replace(f"{d_name}-", "").replace(".gwf", "")
        file_full_path = os.path.join(base_path, filename)
        base_name = os.path.splitext(filename)[0]

        log(f"[{d_name} {file_index}/{len(files)}] 开始处理文件: {temp_filename}")
        log(f"[{d_name} {file_index}/{len(files)}] 文件路径: {file_full_path}")

        if temp_filename not in labels_dict:
            log(f"[{d_name} {file_index}/{len(files)}] 未找到标签，跳过: {temp_filename}")
            continue

        label = labels_dict[temp_filename]

        all_splits_exist = all(
            os.path.exists(os.path.join(save_path, f"{base_name}_p{i}.npy"))
            for i in range(num_splits)
        )
        if all_splits_exist:
            log(f"[{d_name} {file_index}/{len(files)}] 已处理完成，跳过: {filename}")
            continue

        try:
            read_start = time.perf_counter()
            log(f"[{d_name} {file_index}/{len(files)}] 开始读取 GWF")
            data = TimeSeries.read(file_full_path, channel=f"{d_name}:TEST_INJ").value
            read_elapsed = time.perf_counter() - read_start
            log(
                f"[{d_name} {file_index}/{len(files)}] 读取完成，数据长度 {len(data)}，用时 {read_elapsed:.3f}s"
            )

            split_start = time.perf_counter()
            total_length = len(data)
            split_length = total_length // num_splits
            log(
                f"[{d_name} {file_index}/{len(files)}] 开始切片，总长度 {total_length}，每段长度 {split_length}"
            )

            for i in range(num_splits):
                save_one_start = time.perf_counter()
                start = i * split_length
                end = (i + 1) * split_length
                chunk = data[start:end]
                sample = {
                    "data": chunk,
                    "label": label,
                }
                split_filename = f"{base_name}_p{i}.npy"
                split_path = os.path.join(save_path, split_filename)

                log(
                    f"[{d_name} {file_index}/{len(files)}] 保存 split {i + 1}/{num_splits}: {split_filename}"
                )
                np.save(split_path, sample)
                save_one_elapsed = time.perf_counter() - save_one_start
                log(
                    f"[{d_name} {file_index}/{len(files)}] split {i + 1}/{num_splits} 保存完成，用时 {save_one_elapsed:.3f}s"
                )

            split_elapsed = time.perf_counter() - split_start
            file_elapsed = time.perf_counter() - file_start
            log(
                f"[{d_name} {file_index}/{len(files)}] 文件处理完成: {filename}，切片与保存用时 {split_elapsed:.3f}s，总用时 {file_elapsed:.3f}s"
            )

            count += 1
            if count % 20 == 0:
                log(f"探测器 {d_name} 已累计处理 {count} 个文件")

        except Exception as e:
            file_elapsed = time.perf_counter() - file_start
            log(f"[{d_name} {file_index}/{len(files)}] 处理失败: {filename}，用时 {file_elapsed:.3f}s")
            log(f"[{d_name} {file_index}/{len(files)}] 异常信息: {e}")

    log(f"探测器 {d_name} 全部完成，切片文件保存在: {save_path}")


if __name__ == "__main__":
    detectors = ["H1", "L1"]
    version = "v3"
    filename = "training_set0"
    input_path = f"training_set/{version}/{filename}"

    base_label_path = f"./{input_path}/train_idx.csv"
    labels_dict = getLabel(base_label_path)

    for d_name in detectors:
        base_path = f"./{input_path}/{d_name}"
        save_path = f"./processed_data_train/{version}/{filename}/{d_name}_splits"
        getSGWBData(d_name, base_path, save_path, labels_dict)
