from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries


POP_NUMS = 1
FRAME_NUMS = 10
BASE_VERSION = "v61"   # this will be changed
SIGNAL_VERSION = f"fixed_one_test_{BASE_VERSION}"
NOISE_VERSION = BASE_VERSION
OUTPUT_VERSION = f"fixed_one_test_{BASE_VERSION}_random_two_noise_500"   # this will be changed
DETECTORS = ["H1", "L1"]
STRAIN_CHANNEL_SUFFIX = "TEST_INJ"
NOISE_SCALE = 2.0   # this will be changed
END_NOISE_NUM = 100000   # this will be changed
RANDOM_SEED = 42

SCRIPTS_DIR = Path(__file__).resolve().parent
SIGNAL_ROOT = SCRIPTS_DIR / "training_set_test" / SIGNAL_VERSION / "training_set0"
NOISE_ROOT_TEMPLATE = "./noise_waveform/{version}/noise_waveform_{detector}"
OUTPUT_ROOT = SCRIPTS_DIR / "training_set_superimposed_test" / OUTPUT_VERSION


def draw_noise_numbers(frame_nums: int, end_noise_num: int, seed: int) -> list[int]:
    if end_noise_num < frame_nums:
        raise ValueError(
            f"END_NOISE_NUM={end_noise_num} is smaller than FRAME_NUMS={frame_nums}; "
            "cannot sample unique noise numbers."
        )
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(1, end_noise_num + 1), size=frame_nums, replace=False).tolist()


def superimpose(
    detector_name: str,
    pop_nums: int = POP_NUMS,
    frame_nums: int = FRAME_NUMS,
    noise_scale: float = NOISE_SCALE,
    end_noise_num: int = END_NOISE_NUM,
    seed: int = RANDOM_SEED,
) -> None:
    signal_dir = SIGNAL_ROOT / detector_name
    noise_dir = SCRIPTS_DIR / NOISE_ROOT_TEMPLATE.format(version=NOISE_VERSION, detector=detector_name)
    output_dir = OUTPUT_ROOT / detector_name
    strain_channel = f"{detector_name}:{STRAIN_CHANNEL_SUFFIX}"

    output_dir.mkdir(parents=True, exist_ok=True)

    noise_numbers = draw_noise_numbers(frame_nums, end_noise_num, seed)

    total_tasks = pop_nums * frame_nums
    success_count = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_mismatch = 0
    failed_count = 0

    print(
        f"[{detector_name}] start superposition, total_tasks={total_tasks}, "
        f"noise_numbers={noise_numbers}"
    )

    for pop_num in range(pop_nums):
        for frame_num in range(frame_nums):
            noise_number = noise_numbers[frame_num]
            signal_path = signal_dir / f"{detector_name}-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            noise_path = noise_dir / f"{detector_name}-STRAIN-{noise_number:05d}-duration.gwf"
            output_path = output_dir / f"{detector_name}-SUPERIMPOSED-pop{pop_num:05d}_sample{frame_num:05d}.gwf"
            label = f"pop{pop_num:05d}_sample{frame_num:05d}"

            if output_path.exists():
                skipped_existing += 1
                print(f"  - skip {label}: output already exists")
                continue

            if not signal_path.exists():
                skipped_missing += 1
                print(f"  - skip {label}: missing signal file")
                continue

            if not noise_path.exists():
                skipped_missing += 1
                print(f"  - skip {label}: missing noise file {noise_path.name}")
                continue

            try:
                signal_ts = TimeSeries.read(signal_path, strain_channel)
                noise_ts = TimeSeries.read(noise_path, strain_channel)

                if len(signal_ts) != len(noise_ts) or signal_ts.dt != noise_ts.dt:
                    skipped_mismatch += 1
                    print(f"  - skip {label}: signal/noise length or dt mismatch")
                    continue

                superimposed_ts = signal_ts + noise_scale * noise_ts
                superimposed_ts.write(output_path)
                success_count += 1

            except Exception as exc:
                failed_count += 1
                print(f"  - failed {label}: {exc}")

        completed_pops = pop_num + 1
        if completed_pops % 10 == 0 or completed_pops == pop_nums:
            print(f"[{detector_name}] completed {completed_pops}/{pop_nums} pops")

    print(
        f"[{detector_name}] done. success={success_count}, "
        f"skipped_existing={skipped_existing}, skipped_missing={skipped_missing}, "
        f"skipped_mismatch={skipped_mismatch}, failed={failed_count}"
    )


def main() -> None:
    for detector_name in DETECTORS:
        superimpose(detector_name)


if __name__ == "__main__":
    main()
