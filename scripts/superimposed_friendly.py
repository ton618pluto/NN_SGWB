from pathlib import Path

from gwpy.timeseries import TimeSeries

# 如果要改就改这里
POP_NUMS = 1200   # 超参数组数
FRAME_NUMS = 4    # 每段frame是2048s，frame_nums表示有几段frame
DETECTORS = ["H1", "L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = SCRIPTS_DIR / "training_set" / "v4" / "training_set0"
NOISE_ROOT_TEMPLATE = "./noise_waveform/v1/noise_waveform_{detector}"
OUTPUT_ROOT = SCRIPTS_DIR / "training_set_superimposed" / "v1"


def superimpose(detector_name: str, pop_nums: int = POP_NUMS, frame_nums: int = FRAME_NUMS) -> None:
    signal_dir = SIGNAL_ROOT / detector_name
    noise_dir = SCRIPTS_DIR / NOISE_ROOT_TEMPLATE.format(detector=detector_name)
    output_dir = OUTPUT_ROOT / detector_name
    strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"

    output_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = pop_nums * frame_nums
    success_count = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_mismatch = 0
    failed_count = 0

    print(f"[{detector_name}] 开始叠加，共 {total_tasks} 个文件对。")

    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            noise_path = noise_dir / f"{detector_name}-STRAIN-{frame_num + 1:05d}-duration.gwf"
            output_path = output_dir / f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            label = f"pop{pop_num:05d}_sample{frame_num:05d}"

            if output_path.exists():
                skipped_existing += 1
                print(f"  - 跳过 {label}：叠加结果已存在。")
                continue

            if not signal_path.exists():
                skipped_missing += 1
                print(f"  - 跳过 {label}：没找到信号文件。")
                continue

            if not noise_path.exists():
                skipped_missing += 1
                print(f"  - 跳过 {label}：没找到噪声文件 {noise_path.name}。")
                continue

            try:
                signal_ts = TimeSeries.read(signal_path, strain_channel)
                noise_ts = TimeSeries.read(noise_path, strain_channel)

                if len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt:
                    skipped_mismatch += 1
                    print(f"  - 跳过 {label}：信号和噪声长度或采样间隔不一致。")
                    continue

                superimposed_ts = signal_ts + noise_ts
                superimposed_ts.write(output_path)
                success_count += 1

            except Exception as exc:
                failed_count += 1
                print(f"  - 失败 {label}：{exc}")

        completed_pops = pop_num + 1
        if completed_pops % 10 == 0 or completed_pops == pop_nums:
            print(f"[{detector_name}] 已完成 {completed_pops}/{pop_nums} 个 pop。")

    print(
        f"[{detector_name}] 结束。成功 {success_count}，"
        f"已存在跳过 {skipped_existing}，"
        f"缺文件跳过 {skipped_missing}，"
        f"不匹配跳过 {skipped_mismatch}，"
        f"失败 {failed_count}。"
    )


def main() -> None:
    for detector_name in DETECTORS:
        superimpose(detector_name)


if __name__ == "__main__":
    main()
