import os
from pathlib import Path

import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries

NUM_SPLITS = 8
POP_NUM = 100
FRAME_NUM = 23


def resolve_input_root(scripts_dir: Path, version: str) -> Path:
    candidates = [
        scripts_dir / "nosie_waveform" / version,
        scripts_dir / "noise_waveform" / version,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_output_name(detector: str, pop_id: int, frame_id: int, split_index: int) -> str:
    return f"{detector}-NOISE-pop{pop_id:05d}_sample{frame_id:05d}_p{split_index}.npy"


def get_label(file_path: Path) -> dict[str, list[float]]:
    df = pd.read_csv(file_path)
    return {row[0]: list(row[2:]) for row in df.itertuples(index=False)}


def get_noise_data(
    detector: str,
    base_path: str,
    save_path: str,
    labels_dict: dict[str, list[float]],
    num_splits: int = NUM_SPLITS,
    pop_num: int = POP_NUM,
    frame_num: int = FRAME_NUM,
) -> None:
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(base_path):
        print(f"输入目录不存在: {base_path}")
        return

    files = sorted(f for f in os.listdir(base_path) if f.endswith(".gwf"))[:frame_num]
    print(
        f"开始处理探测器 {detector}，共 {len(files)} 个噪声文件，"
        f"每个文件切分成 {num_splits} 段，复制到 {pop_num} 个 pop"
    )

    count = 0
    for frame_id, filename in enumerate(files):
        file_full_path = os.path.join(base_path, filename)

        all_outputs_exist = all(
            os.path.exists(os.path.join(save_path, build_output_name(detector, pop_id, frame_id, split_idx)))
            for pop_id in range(pop_num)
            for split_idx in range(num_splits)
        )
        if all_outputs_exist:
            print(f"跳过已处理文件: {filename}")
            continue

        try:
            data = TimeSeries.read(file_full_path, channel=f"{detector}:TEST_INJ").value
            total_length = len(data)
            split_length = total_length // num_splits

            for split_idx in range(num_splits):
                start = split_idx * split_length
                end = (split_idx + 1) * split_length
                chunk = data[start:end]

                for pop_id in range(pop_num):
                    label_key = f"pop{pop_id:05d}_sample{frame_id:05d}"
                    if label_key not in labels_dict:
                        raise KeyError(f"Label not found for key: {label_key}")
                    sample = {
                        "data": chunk,
                        "label": labels_dict[label_key],
                        "pop_id": pop_id,
                        "frame_id": frame_id,
                        "source_file": filename,
                    }
                    output_name = build_output_name(detector, pop_id, frame_id, split_idx)
                    np.save(os.path.join(save_path, output_name), sample)

            count += 1
            if count % 5 == 0:
                print(f"已处理 {count} 个来自探测器 {detector} 的噪声文件")

        except Exception as exc:
            print(f"处理 {filename} 失败: {exc}")

    print(f"{detector} 处理完成，切分文件保存在: {save_path}")


if __name__ == "__main__":
    detectors = ["H1", "L1"]
    num_splits = NUM_SPLITS
    pop_num = POP_NUM
    frame_num = FRAME_NUM
    version = "v6"

    scripts_dir = Path(__file__).resolve().parent
    input_root = resolve_input_root(scripts_dir, version)
    save_root = scripts_dir / "processed_data_noise" / version
    label_path = scripts_dir / "training_set" / version / "training_set0" / "train.csv"
    if not label_path.exists():
        fallback_label_path = scripts_dir / "training_set" / version / "training_set0" / "train_idx.csv"
        label_path = fallback_label_path
    labels_dict = get_label(label_path)

    print(f"输入目录: {input_root}")
    print(f"输出目录: {save_root}")
    print(f"标签文件: {label_path}")
    print(f"pop 数量: {pop_num} (pop00000 到 pop{pop_num - 1:05d})")
    print(f"frame 数量: {frame_num} (sample00000 到 sample{frame_num - 1:05d})")
    print(f"切分段数: {num_splits}")

    for detector in detectors:
        base_path = input_root / f"noise_waveform_{detector}"
        save_path = save_root / f"{detector}_splits"
        get_noise_data(
            detector=detector,
            base_path=str(base_path),
            save_path=str(save_path),
            labels_dict=labels_dict,
            num_splits=num_splits,
            pop_num=pop_num,
            frame_num=frame_num,
        )
